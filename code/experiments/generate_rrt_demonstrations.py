#!/usr/bin/env python3
"""Convert stored RRT paths into structured-policy demonstrations.

The planner is used only offline.  The output contains observations and actor
actions; no oracle waypoint is appended to the observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from env.manipulator_env import ManipulatorEnv
from robot_config import DEFAULT_URDF, DEFAULT_XML
from utils.validation import ValidationSet


def _dense_path(path: list[list[float]], max_joint_step: float):
    points = [np.asarray(path[0], dtype=float)]
    for raw_a, raw_b in zip(path[:-1], path[1:]):
        a, b = np.asarray(raw_a, dtype=float), np.asarray(raw_b, dtype=float)
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_joint_step)))
        points.extend((a + (b - a) * (i / count)) for i in range(1, count + 1))
    return points


def _set_configuration(env, q, dq, progress):
    env.q = q.copy()
    env.dq = dq.copy()
    env.path_param = float(progress)
    env._trajectory_phase = float(progress)
    env.x_d = (1.0 - progress) * env.x_start + progress * env.x_goal
    env.dx_d = np.zeros(3)
    env._last_sigma = np.float32(1.0)
    env._integral_err.fill(0.0)
    if env.mj_data is not None:
        env.mj_data.qpos[:env.n] = q
        env.mj_data.qvel[:env.n] = dq
        mujoco.mj_forward(env.mj_model, env.mj_data)
    env._cached_x_ee = None
    env._cached_w = None
    env._cached_capsule_dists = None
    env._cached_capsule_directions = None


def _structured_action(env, q, desired_dq, task_scale, nullspace_scale):
    jacobian = env.kin.jacobian_position(q)
    j_pinv = env.kin.pseudo_inverse(jacobian)
    basis = env.kin.null_space_basis_position(q)
    desired_dx = jacobian @ desired_dq
    dx_nom = env._compute_task_velocity()
    # PhysicsInformedActor.sample() returns actions in physical units after
    # applying task_scale/nullspace_scale.  Demonstrations must use those same
    # units; dividing by the scales here would train against normalized actions
    # that the actor can never emit when a scale is smaller than one.
    task = desired_dx - dx_nom
    task_motion = j_pinv @ desired_dx
    null = basis.T @ (desired_dq - task_motion)
    lower = np.r_[-np.full(3, task_scale),
                  -np.full(env.n - 3, nullspace_scale)]
    upper = -lower
    return np.clip(np.concatenate([task, null]), lower, upper)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-scale", type=float, default=0.2)
    parser.add_argument("--nullspace-scale", type=float, default=0.25)
    parser.add_argument("--max-joint-step", type=float, default=0.01)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--xml", default=DEFAULT_XML)
    args = parser.parse_args()
    if args.task_scale <= 0 or args.nullspace_scale <= 0 or args.max_joint_step <= 0:
        raise ValueError("action scales and max joint step must be positive")

    scenes = json.loads(args.input.read_text())
    if not scenes or any("feasible_q_path" not in scene for scene in scenes):
        raise ValueError("every scene must contain a feasible_q_path")
    env = ManipulatorEnv(
        urdf_path=args.urdf, xml_path=args.xml,
        n_obstacles=max(len(scene["obstacles"]) for scene in scenes),
        use_trajectory_generator=False, gate_enabled=False,
        obs_scene_embed=10, obs_waypoint_steps=[10, 25, 50],
    )
    validation = ValidationSet(args.input)
    states, actions, scene_ids = [], [], []
    for scene in scenes:
        validation.apply_scene_to_env(env, scene)
        path = _dense_path(scene["feasible_q_path"], args.max_joint_step)
        for index, (q, next_q) in enumerate(zip(path[:-1], path[1:])):
            desired_dq = np.clip((next_q - q) / env.dt, -env._dq_max, env._dq_max)
            progress = index / max(len(path) - 1, 1)
            _set_configuration(env, q, desired_dq, progress)
            states.append(env._get_obs().astype(np.float32))
            actions.append(_structured_action(
                env, q, desired_dq, args.task_scale, args.nullspace_scale
            ).astype(np.float32))
            scene_ids.append(scene["scene_id"])
            if len(states) >= args.max_samples:
                break
        if len(states) >= args.max_samples:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, states=np.asarray(states),
                        actions=np.asarray(actions), scene_ids=np.asarray(scene_ids))
    print(f"wrote {len(states)} demonstrations from {len(set(scene_ids))} scenes "
          f"to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
