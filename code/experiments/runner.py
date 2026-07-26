"""Run one reproducible phase-one evaluation shard and emit canonical JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agent.sac_agent import SACAgent
from agent.vanilla_sac_agent import VanillaSACAgent
from env.dynamics import ManipulatorDynamics
from env.manipulator_env import ManipulatorEnv
from env.vanilla_env import VanillaEnv
from env.residual_env import ResidualEnv
from experiment_config import (ALGORITHM, ENVIRONMENT, PHASE1_METHODS,
                               PHASE1_SCENARIOS, TRAINING_PROTOCOL_VERSION)
from experiments.metrics import EpisodeRecorder
from experiments.scenarios import apply_named_scenario
from utils.validation import ValidationSet


TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


def _structured_agent(env, checkpoint: Path, args) -> SACAgent:
    agent = SACAgent(
        state_dim=env.obs_dim, action_dim=env.act_dim, dynamics=env.dyn,
        hidden_dims=ALGORITHM.hidden_dims, lambda_dyn=args.lambda_dyn,
        task_scale=ALGORITHM.task_scale, nullspace_scale=ALGORITHM.nullspace_scale,
        n_critics=ALGORITHM.n_critics, use_safety_critic=args.safety_critic,
        device=args.device,
    )
    metadata = agent.load(str(checkpoint), load_optimizers=False)
    if metadata.get("training_protocol_version") != TRAINING_PROTOCOL_VERSION:
        raise RuntimeError("checkpoint was not trained with the current protocol")
    agent.actor.eval()
    return agent


def _vanilla_agent(env, checkpoint: Path, args) -> VanillaSACAgent:
    agent = VanillaSACAgent(env.obs_dim, env.n, hidden_dims=ALGORITHM.hidden_dims,
                            device=args.device)
    metadata = agent.load(str(checkpoint), load_optimizers=False)
    if metadata.get("training_protocol_version") != TRAINING_PROTOCOL_VERSION:
        raise RuntimeError("checkpoint was not trained with the current protocol")
    agent.actor.eval()
    return agent


def _gradient_projection_action(env) -> np.ndarray:
    """Repel link capsules from spheres and express the joint gradient in B(q)."""
    if env.sdf.n_obs == 0:
        return np.zeros(env.act_dim)
    gradient = np.zeros(env.n)
    eps = 1e-4
    for joint in range(env.n):
        q_plus, q_minus = env.q.copy(), env.q.copy()
        q_plus[joint] += eps
        q_minus[joint] -= eps
        d_plus = env.sdf.min_distance(np.zeros(3), q_plus, kinematics=env.kin)
        d_minus = env.sdf.min_distance(np.zeros(3), q_minus, kinematics=env.kin)
        gradient[joint] = (d_plus - d_minus) / (2 * eps)
    basis = env.kin.null_space_basis_position(env.q)
    action = np.zeros(env.act_dim)
    action[3:] = np.clip(basis.T @ gradient, -ALGORITHM.nullspace_scale,
                         ALGORITHM.nullspace_scale)
    return action


def _joint_residual_to_structured(env, residual: np.ndarray) -> np.ndarray:
    jacobian = env.kin.jacobian_position(env.q)
    basis = env.kin.null_space_basis_position(env.q)
    return np.concatenate([jacobian @ residual, basis.T @ residual])


def evaluate_episode(env, action_fn, method: str, scenario: str,
                     seed: int, scene_id: int) -> dict:
    recorder = EpisodeRecorder(method, scenario, seed, scene_id, env.dt, TAU_MAX)
    obs = env._get_obs()
    previous_dq = env.dq.copy()
    for _ in range(env.episode_len):
        action = np.asarray(action_fn(obs), dtype=float)
        if action.shape != (env.act_dim,):
            raise ValueError(f"{method} returned action {action.shape}, expected {(env.act_dim,)}")
        obs, _, done, info = env.step(action)
        ddq = (env.dq - previous_dq) / env.dt
        torque = env.dyn.compute_torque(env.q, env.dq, ddq)
        previous_dq = env.dq.copy()
        if isinstance(env, VanillaEnv):
            null_norm = ref_norm = gate = 0.0
        elif isinstance(env, ResidualEnv):
            basis = env.kin.null_space_basis_position(env.q)
            jacobian = env.kin.jacobian_position(env.q)
            null_norm = float(np.linalg.norm(basis @ (basis.T @ action)))
            ref_norm = float(np.linalg.norm(jacobian @ action))
            gate = float(env._last_sigma)
        else:
            basis = env.kin.null_space_basis_position(env.q)
            null_norm = float(np.linalg.norm(basis @ action[3:]))
            ref_norm = float(np.linalg.norm(action[:3]))
            gate = float(env._last_sigma)
        recorder.add(info=info, dq=env.dq, torque=torque,
                     nullspace_norm=null_norm, reference_norm=ref_norm, gate=gate)
        if done:
            break
    return recorder.finish().to_dict()


def run(args) -> list[dict]:
    env_class = {
        "sac_joint": VanillaEnv,
        "sac_residual": ResidualEnv,
    }.get(args.method, ManipulatorEnv)
    env = env_class(
        urdf_path=args.urdf, xml_path=args.xml, dt=ENVIRONMENT.dt,
        episode_len=args.steps, trajectory_steps=args.trajectory_steps,
        n_obstacles=14,
        w_track=ENVIRONMENT.w_track, w_obs=ENVIRONMENT.w_obs,
        w_manip=ENVIRONMENT.w_manip, w_energy=ENVIRONMENT.w_energy,
        w_collision=ENVIRONMENT.w_collision, w_action=ENVIRONMENT.w_action,
        d_safe=ENVIRONMENT.d_safe, success_bonus=ENVIRONMENT.success_bonus,
        use_cbf=args.method == "cbf_qp",
        gate_enabled=args.method in ("ours_full", "sac_residual"),
    )
    agent = None
    if args.method.startswith("ours_"):
        agent = _structured_agent(env, args.checkpoint, args)
    elif args.method == "sac_joint":
        agent = _vanilla_agent(env, args.checkpoint, args)
    elif args.method == "sac_residual":
        agent = _vanilla_agent(env, args.checkpoint, args)

    scene_set = ValidationSet(str(args.scene_json)) if args.scene_json else None
    if scene_set is not None and not scene_set.scenes:
        raise ValueError(f"scene set is empty: {args.scene_json}")
    rows = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env.reset(seed=seed)
        if scene_set is None:
            apply_named_scenario(env, args.scenario)
        else:
            scene_set.apply_scene_to_env(
                env, scene_set.scenes[episode % len(scene_set.scenes)]
            )
        if args.method in ("pd", "cbf_qp"):
            action_fn = lambda obs: np.zeros(env.act_dim)
        elif args.method == "gradient_projection":
            action_fn = lambda obs: _gradient_projection_action(env)
        elif args.method == "sac_joint":
            action_fn = lambda obs: agent.select_action(obs, deterministic=True)
        elif args.method == "sac_residual":
            action_fn = lambda obs: agent.select_action(obs, deterministic=True)
        else:
            action_fn = lambda obs: agent.select_action(obs, deterministic=True)
        scene_id = (scene_set.scenes[episode % len(scene_set.scenes)]["scene_id"]
                    if scene_set is not None else episode)
        rows.append(evaluate_episode(env, action_fn, args.method, args.scenario,
                                     args.seed, scene_id))
    return rows


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=PHASE1_METHODS)
    parser.add_argument("--scenario", required=True, choices=PHASE1_SCENARIOS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=ENVIRONMENT.episode_len)
    parser.add_argument("--trajectory-steps", type=int,
                        default=ENVIRONMENT.trajectory_steps)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scene-json", type=Path,
                        help="Fixed certified scenes; cycled in file order")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--urdf", default=str(root / "panda_description/urdf/panda.urdf"))
    parser.add_argument("--xml", default=str(root / "models/panda_scene.xml"))
    parser.add_argument("--device", default="cpu")
    parser.set_defaults(lambda_dyn=ALGORITHM.lambda_dyn, safety_critic=True)
    args = parser.parse_args()
    if args.method in ("ours_no_physics", "ours_physics", "ours_full",
                       "sac_joint", "sac_residual") and args.checkpoint is None:
        parser.error(f"--checkpoint is required for {args.method}")
    if args.method == "ours_no_physics":
        args.lambda_dyn = 0.0
        args.safety_critic = False
    elif args.method == "ours_physics":
        args.safety_critic = False
    return args


def main() -> int:
    args = parse_args()
    rows = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
