"""
train.py
--------
SAC (off-policy) training entry point for physics-informed RL on a
7-DOF manipulator with obstacle avoidance.

Usage:
    python train.py --steps 500000 --n_envs 32 \\
        --scene_json results/trajectories_obs.json

    # Resume from checkpoint:
    python train.py --resume checkpoints/run_name/ckpt_best.pt --steps 1000000

    # Render mode (single env):
    python train.py --render --steps 10000 --n_envs 1

Author: xie yang
Date:   2025-06

"""

import json
import torch
import argparse
import sys
import os
import numpy as np
from datetime import datetime
from multiprocessing import Array, Lock

sys.path.insert(0, os.path.dirname(__file__))

from env.manipulator_env import ManipulatorEnv
from env.dynamics import ManipulatorDynamics
from agent.sac_agent import SACAgent
from agent.vanilla_sac_agent import VanillaSACAgent
from env.vanilla_env import VanillaEnv
from env.residual_env import ResidualEnv
from utils.replay_buffer import ReplayBuffer
from utils.logger import (TrainingLogger, REWARD_HEADER, REWARD_FORMAT,
                           reward_accumulators, accumulate_rewards,
                           avg_rewards, reward_print_values)
from utils.validation import ValidationSet, evaluate_on_validation_set
from experiment_config import ALGORITHM, ENVIRONMENT, TRAINING_PROTOCOL_VERSION


def validation_rank(metrics):
    """Return a lexicographic rank: success, safety, then tracking accuracy."""
    return (float(metrics["success_rate"]),
            -float(metrics["collision_rate"]),
            -float(metrics["avg_tracking_error"]))


def is_better_validation(candidate, incumbent):
    """Whether candidate should replace the validation-selected checkpoint."""
    return incumbent is None or validation_rank(candidate) > validation_rank(incumbent)


def scene_sampling_weights(success_ema, uniform_mix=0.20):
    """Prioritize low-success scenes while retaining uniform coverage."""
    values = np.asarray(success_ema, dtype=np.float64)
    if values.size == 0 or not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("invalid scene EMA or uniform mixture")
    difficulty = np.clip(1.0 - values, 0.05, 1.0)
    prioritized = difficulty / difficulty.sum()
    uniform = np.full(values.size, 1.0 / values.size)
    return uniform_mix * uniform + (1.0 - uniform_mix) * prioritized


# ========================================================================
# Argument parsing
# ========================================================================

def parse_args():
    p = argparse.ArgumentParser()

    # --- SAC training ---
    p.add_argument("--steps",        type=int,   default=500_000)
    p.add_argument("--batch_size",   type=int,   default=ALGORITHM.batch_size)
    p.add_argument("--start_steps",  type=int,   default=ALGORITHM.start_steps)
    p.add_argument("--grad_steps",   type=int,   default=ALGORITHM.gradient_steps)
    p.add_argument("--update_every", type=int,   default=1)
    p.add_argument("--buffer_size",  type=int,   default=ALGORITHM.replay_size)
    p.add_argument("--n_envs",       type=int,   default=ALGORITHM.parallel_envs)
    p.add_argument("--episode_len",  type=int,   default=ENVIRONMENT.episode_len)
    p.add_argument("--trajectory_steps", type=int, default=ENVIRONMENT.trajectory_steps)
    p.add_argument("--tracking_full_speed_error", type=float,
                   default=ENVIRONMENT.tracking_full_speed_error)
    p.add_argument("--tracking_stop_error", type=float,
                   default=ENVIRONMENT.tracking_stop_error)
    p.add_argument("--success_tolerance", type=float,
                   default=ENVIRONMENT.success_tolerance)
    p.add_argument("--success_hold_steps", type=int,
                   default=ENVIRONMENT.success_hold_steps)
    p.add_argument("--seed",         type=int,   default=11)

    # --- Policy / action ---
    p.add_argument("--task_scale",       type=float, default=ALGORITHM.task_scale)
    p.add_argument("--nullspace_scale",  type=float, default=ALGORITHM.nullspace_scale)
    p.add_argument("--hidden_dims",      type=str,   default="256,256")
    p.add_argument("--n_critics",        type=int,   default=ALGORITHM.n_critics)
    p.add_argument("--backbone",         type=str,   default="mlp", choices=["mlp", "transformer"])
    p.add_argument("--frame_stack",      type=int,   default=1)
    p.add_argument("--action_horizon",   type=int,   default=1)
    p.add_argument("--d_model",          type=int,   default=128)
    p.add_argument("--n_heads",          type=int,   default=4)
    p.add_argument("--n_enc_layers",     type=int,   default=2)
    p.add_argument("--n_dec_layers",     type=int,   default=2)
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--use_cbf",          action="store_true")
    p.add_argument("--cbf_alpha",        type=float, default=1.0)
    p.add_argument("--no_safety_critic", action="store_true")
    p.add_argument("--disable_gate", action="store_true")
    p.add_argument("--agent_type", choices=["structured", "joint", "residual"],
                   default="structured")

    # --- Physics regularization ---
    p.add_argument("--lambda_dyn",  type=float, default=ALGORITHM.lambda_dyn)

    # --- Learning ---
    p.add_argument("--lr",            type=float, default=ALGORITHM.learning_rate)
    p.add_argument("--tau",           type=float, default=ALGORITHM.tau)
    p.add_argument("--alpha",         type=float, default=ALGORITHM.alpha)
    p.add_argument("--target_entropy",type=float, default=None)
    p.add_argument("--critic_warmup", type=int,   default=5000)
    p.add_argument("--lr_lag",        type=float, default=ALGORITHM.lagrange_learning_rate)
    p.add_argument("--lag_init",      type=float, default=ALGORITHM.lagrange_initial_value)
    p.add_argument("--lag_target",    type=float, default=0.05)
    p.add_argument("--cost_limit",    type=float, default=ALGORITHM.cost_limit)
    p.add_argument("--cost_scale",    type=float, default=1.0)

    # --- Reward ---
    p.add_argument("--w_track",      type=float, default=ENVIRONMENT.w_track)
    p.add_argument("--w_obs",        type=float, default=ENVIRONMENT.w_obs)
    p.add_argument("--w_collision",  type=float, default=ENVIRONMENT.w_collision)
    p.add_argument("--w_manip",      type=float, default=ENVIRONMENT.w_manip)
    p.add_argument("--w_energy",     type=float, default=ENVIRONMENT.w_energy)
    p.add_argument("--w_action",     type=float, default=ENVIRONMENT.w_action)
    p.add_argument("--w_null",       type=float, default=0.0)
    p.add_argument("--d_safe",       type=float, default=ENVIRONMENT.d_safe)
    p.add_argument("--success_bonus",type=float, default=ENVIRONMENT.success_bonus)
    p.add_argument("--reward_min",   type=float, default=None)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--path_deadzone",type=float, default=0.20)

    # --- Data ---
    p.add_argument("--scene_json",   type=str,   default=None)
    p.add_argument("--scene_id",     type=int,   default=-1)
    p.add_argument("--val_json",     type=str,   default=None)
    p.add_argument("--val_every",    type=int,   default=50)
    p.add_argument("--val_every_steps", type=int,
                   default=ALGORITHM.validation_interval_steps)
    p.add_argument("--val_scenes",   type=int,   default=10)
    p.add_argument("--obs_scene_embed", type=int, default=0)
    p.add_argument("--obs_waypoint_steps", type=str, default=None)

    # --- Replay buffer ---
    p.add_argument("--per",         action="store_true")

    # --- Paths ---
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    _venv_data = os.path.join(_here, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                              "/share/example-robot-data/robots/panda_description")
    _default_urdf = os.path.join(_venv_data, "urdf/panda.urdf")
    _default_xml  = os.path.join(_root, "models/panda_scene.xml")

    p.add_argument("--urdf",       type=str, default=_default_urdf)
    p.add_argument("--xml",        type=str, default=_default_xml)
    p.add_argument("--save_path",  type=str, default="checkpoints")
    p.add_argument("--run_name",   type=str, default=None)

    # --- Logging / checkpoint ---
    p.add_argument("--log_every",        type=int, default=10)
    p.add_argument("--checkpoint_every", type=int, default=1000)
    p.add_argument("--checkpoint_every_steps", type=int,
                   default=ALGORITHM.checkpoint_interval_steps)
    p.add_argument("--no_collision_term", action="store_true")

    # --- Render ---
    p.add_argument("--render", action="store_true")

    # --- Resume ---
    p.add_argument("--resume",       type=str, default=None)
    p.add_argument("--allow_legacy_resume", action="store_true")
    p.add_argument("--reset_alpha",  action="store_true")
    p.add_argument("--reset_critic", action="store_true")
    p.add_argument("--reset_actor",  action="store_true")
    p.add_argument("--load_actor",   type=str, default=None)

    return p.parse_args()


# ========================================================================
# Setup helpers
# ========================================================================

def setup_scene_loading(args):
    """
    Load scenes from JSON and initialize prioritized sampling.

    Returns
    -------
    scene_data      : (ValidationSet, scenes) or (ValidationSet, single_scene) or None
    n_obs           : number of obstacles per scene
    scene_weights   : multiprocessing.Array for shared scene sampling weights
    scene_ema       : np.ndarray for per-scene EMA of rewards
    scene_counts    : np.ndarray for per-scene visit counts
    scene_lock      : multiprocessing.Lock for weights access
    obs_waypoint_steps : list of int or None
    """
    scene_data = None
    scene_weights = scene_ema = scene_counts = None
    scene_lock = None
    obs_waypoint_steps = None

    if args.obs_waypoint_steps is not None:
        obs_waypoint_steps = [int(s.strip()) for s in args.obs_waypoint_steps.split(",")]

    if args.scene_json is not None:
        vs = ValidationSet(args.scene_json)
        if args.scene_id >= 0:
            scene_data = (vs, vs.get_scene(args.scene_id))
            n_obs = len(scene_data[1]["obstacles"])
            print(f"[train] Fixed scene mode: scene_id={args.scene_id}, obstacles={n_obs}")
        else:
            scene_data = (vs, vs.scenes)
            n_obs = len(vs.scenes[0]["obstacles"])
            print(f"[train] Scene cycle mode: {len(vs.scenes)} scenes, obs/scene={n_obs}")

        if args.n_envs > 1 and scene_data is not None and args.scene_id < 0:
            n_scenes = len(scene_data[1])
            scene_weights = Array('d', [1.0] * n_scenes)
            scene_lock = Lock()
            scene_ema = np.full(n_scenes, 0.5, dtype=np.float64)
            scene_counts = np.zeros(n_scenes, dtype=np.int32)
    else:
        n_obs = 5

    return scene_data, n_obs, scene_weights, scene_ema, scene_counts, scene_lock, obs_waypoint_steps


def make_env_kwargs(args, n_obs, obs_waypoint_steps):
    return dict(
        urdf_path=args.urdf, xml_path=args.xml, obs_radius=0.03,
        n_obstacles=n_obs,
        use_trajectory_generator=args.scene_json is None,
        collision_term=not args.no_collision_term,
        path_deadzone=args.path_deadzone,
        w_obs=args.w_obs, w_collision=args.w_collision, w_track=args.w_track,
        w_manip=args.w_manip, w_energy=args.w_energy, w_action=args.w_action, w_null=args.w_null,
        d_safe=args.d_safe, success_bonus=args.success_bonus,
        lr_lag=args.lr_lag, lag_target=args.lag_target,
        obs_scene_embed=args.obs_scene_embed,
        obs_waypoint_steps=obs_waypoint_steps,
        frame_stack=args.frame_stack,
        episode_len=args.episode_len,
        trajectory_steps=args.trajectory_steps,
        tracking_full_speed_error=args.tracking_full_speed_error,
        tracking_stop_error=args.tracking_stop_error,
        success_tolerance=args.success_tolerance,
        success_hold_steps=args.success_hold_steps,
        reward_min=args.reward_min,
        reward_scale=args.reward_scale,
        use_cbf=args.use_cbf, cbf_alpha=args.cbf_alpha,
        gate_enabled=not args.disable_gate,
    )


def setup_agent_and_buffer(args, state_dim, action_dim, dyn, ref_env, device):
    """Create SACAgent and ReplayBuffer."""
    if args.agent_type != "structured":
        agent = VanillaSACAgent(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=args.hidden_dims, lr=args.lr, alpha=args.alpha,
            tau=args.tau, device=device,
            critic_warmup=args.critic_warmup,
            total_steps=args.steps, n_critics=2,
        )
        if args.per:
            from utils.replay_buffer import PrioritizedReplayBuffer
            buffer = PrioritizedReplayBuffer(args.buffer_size, state_dim, action_dim)
        else:
            buffer = ReplayBuffer(args.buffer_size, state_dim, action_dim)
        return agent, buffer
    single_dim = getattr(ref_env, '_single_obs_dim', state_dim)
    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim, dynamics=dyn,
        hidden_dims=args.hidden_dims,
        lambda_dyn=args.lambda_dyn,
        task_scale=args.task_scale, nullspace_scale=args.nullspace_scale,
        lr=args.lr, alpha=args.alpha, target_entropy=args.target_entropy,
        device=device,
        critic_warmup=args.critic_warmup,
        total_steps=args.steps,
        n_critics=args.n_critics,
        cost_limit=args.cost_limit,
        cost_scale=args.cost_scale,
        backbone=args.backbone, frame_stack=args.frame_stack,
        action_horizon=args.action_horizon,
        single_dim=single_dim,
        d_model=args.d_model, n_heads=args.n_heads,
        n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
        dropout=args.dropout,
        grad_steps=args.grad_steps,
        lr_lag=args.lr_lag,
        lag_init=args.lag_init,
        use_safety_critic=not args.no_safety_critic,
    )
    if args.per:
        from utils.replay_buffer import PrioritizedReplayBuffer
        buffer = PrioritizedReplayBuffer(args.buffer_size, state_dim, action_dim)
    else:
        buffer = ReplayBuffer(args.buffer_size, state_dim, action_dim)
    return agent, buffer


def replay_path_for_checkpoint(checkpoint_path: str, per: bool) -> str:
    suffix = ".replay.pkl.gz" if per else ".replay.npz"
    return os.path.splitext(checkpoint_path)[0] + suffix


def save_training_checkpoint(agent, buffer, checkpoint_path, metadata, per=False,
                             save_replay=False):
    metadata = dict(metadata)
    if save_replay:
        replay_path = replay_path_for_checkpoint(checkpoint_path, per)
        metadata["replay_path"] = replay_path
    agent.save(checkpoint_path, metadata=metadata)
    if save_replay:
        buffer.save(replay_path)


def handle_resume(args, agent, buffer, scene_ema, scene_counts, scene_weights, logger):
    """Load checkpoint and restore training state. Returns (total_steps, episode, best_reward)."""
    total_steps = 0
    episode = 0
    best_reward = -np.inf
    best_validation = None

    # BC pretrained actor (optional)
    if args.load_actor is not None:
        bc_path = args.load_actor
        if not os.path.isabs(bc_path):
            bc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), bc_path)
        if os.path.exists(bc_path):
            ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
            agent.actor.load_state_dict(ckpt["actor"])
            if "obs_normalizer" in ckpt:
                on = ckpt["obs_normalizer"]
                agent.obs_normalizer.mean = on["mean"]
                agent.obs_normalizer.std = on["std"]
                agent.obs_normalizer.n_samples = 100_000
            print(f"[train] Loaded BC-pretrained actor from {bc_path}, meta: {ckpt.get('metadata', {})}")
        else:
            print(f"[train] WARNING: BC checkpoint not found: {bc_path}")

    # SAC checkpoint resume
    if args.resume is not None:
        ckpt_path = args.resume
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ckpt_path)
        if os.path.exists(ckpt_path):
            raw_meta = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            ).get("metadata", {})
            protocol = raw_meta.get("training_protocol_version")
            if protocol != TRAINING_PROTOCOL_VERSION and not args.allow_legacy_resume:
                raise RuntimeError(
                    f"checkpoint protocol {protocol!r} is incompatible with "
                    f"protocol {TRAINING_PROTOCOL_VERSION}; use --load_actor for transfer "
                    "or --allow_legacy_resume to override"
                )
            meta = agent.load(ckpt_path, reset_alpha=args.reset_alpha,
                              reset_critic=args.reset_critic, reset_actor=args.reset_actor,
                              lr=args.lr)
            total_steps = meta.get("step", 0)
            episode = meta.get("episode", 0)
            best_reward = meta.get("best_reward", -np.inf)
            best_validation = meta.get("best_validation")
            logger.best_reward = best_reward

            # Restore per-scene stats
            if scene_ema is not None and "scene_ema" in meta and meta["scene_ema"] is not None \
                    and len(meta["scene_ema"]) == len(scene_ema):
                scene_ema[:] = meta["scene_ema"]
                scene_counts[:] = meta["scene_counts"]
                weights = scene_sampling_weights(
                    scene_ema, ALGORITHM.uniform_scene_mix
                )
                for s in range(len(weights)):
                    scene_weights[s] = weights[s]
                print(f"[train] Restored scene performance stats for {len(scene_ema)} scenes")

            replay_path = replay_path_for_checkpoint(ckpt_path, args.per)
            if not os.path.exists(replay_path):
                raise FileNotFoundError(
                    f"resume requires replay buffer state: {replay_path}"
                )
            buffer.load(replay_path)
            print(f"[train] Restored {len(buffer)} replay transitions from {replay_path}")

            print(f"[train] Resumed from {ckpt_path}: step={total_steps}, episode={episode}, "
                  f"best_reward={best_reward:.3f}")

            if args.reset_alpha:
                agent.log_alpha.data.fill_(np.log(args.alpha))
                agent.alpha = args.alpha
                agent.alpha_opt = torch.optim.Adam([agent.log_alpha], lr=args.lr)
                print(f"[train] Reset alpha to {args.alpha}")
        else:
            print(f"[train] WARNING: resume checkpoint not found: {ckpt_path}")

    return total_steps, episode, best_reward, best_validation


# ========================================================================
# Single-env render loop
# ========================================================================

def _run_render_loop(agent, env, buffer, args, hyperparams, logger, val_set,
                     total_steps=0, episode=0, best_reward=-np.inf,
                     best_validation=None):
    """Single-environment training loop with MuJoCo viewer."""

    print(f"Run directory: {logger.run_dir}")
    print(f"{'Episode':^8}  {'Steps':^8}  {'Reward':^10}  {REWARD_HEADER}  "
          f"{'L_critic':^8}  {'L_scritic':^8}  "
          f"{'L_actor':^10}  {'L_dyn':^9}  {'L_lag':^9}  {'d_obs':^8}  {'suc':^5}")
    print("-" * 175)

    last_losses = {"actor_rl_loss": 0.0, "physics_loss": 0.0}
    log_success_count = 0
    scene_id = getattr(env, '_current_scene_id', -1)

    next_val_step = ((total_steps // args.val_every_steps) + 1) * args.val_every_steps
    next_checkpoint_step = (
        (total_steps // args.checkpoint_every_steps) + 1
    ) * args.checkpoint_every_steps
    while total_steps < args.steps:
        obs = env.reset()
        agent.obs_normalizer.update(obs)
        ep_reward = 0.0
        ep_d_obs = []
        ep_r_acc = reward_accumulators()
        ep_steps = 0
        ep_success = False
        ep_ever_collided = False
        done = False

        while not done:
            if total_steps < args.start_steps:
                if args.agent_type == "structured":
                    a_task = np.random.uniform(-args.task_scale, args.task_scale, 3)
                    a_null = np.random.uniform(
                        -args.nullspace_scale, args.nullspace_scale, env.n - 3)
                    action = np.concatenate([a_task, a_null])
                else:
                    action = np.random.uniform(-2.175, 2.175, env.n)
            else:
                action = agent.select_action(obs)

            q_prev = env.q.copy()
            dq_prev = env.dq.copy()
            next_obs, reward, done, info = env.step(action)

            if args.render:
                env.render()

            dq_next = env.dq.copy()
            agent.obs_normalizer.update(next_obs)

            buffer.push(obs, action, reward, next_obs, done,
                        q=q_prev, dq=dq_prev, dq_next=dq_next,
                        J=env._last_J, sigma=env._last_sigma, dx_nom=env._last_dx_nom,
                        cost=info.get("cost", 0.0))

            obs = next_obs
            ep_reward += reward
            ep_d_obs.append(info["d_obs"])
            accumulate_rewards(info, ep_r_acc)
            ep_ever_collided = ep_ever_collided or info.get("collision", False)
            ep_success = ep_success or info.get("success", False)
            total_steps += 1
            ep_steps += 1

            if total_steps >= args.start_steps and len(buffer) >= args.batch_size \
                    and total_steps % args.update_every == 0:
                for i in range(args.grad_steps):
                    batch = buffer.sample(args.batch_size)
                    losses, td_errors = agent.update(
                        batch, is_last=(i == args.grad_steps - 1),
                        actor_enabled=total_steps >= args.critic_warmup,
                    )
                    if args.per:
                        buffer.update_priorities(batch["indices"], td_errors)
                    logger.log_update(losses)
                    last_losses = losses

        episode += 1
        min_d_obs = min(ep_d_obs) if ep_d_obs else 0.0
        avg_r = avg_rewards(ep_r_acc)

        if ep_success:
            log_success_count += 1

        if episode % args.log_every == 0:
            rp = reward_print_values(avg_r)
            print(f"{episode:>8d}  {total_steps:>8d}  {ep_reward:>10.3f}  "
                  f"{REWARD_FORMAT.format(**rp)}  "
                  f"{last_losses.get('critic_loss', 0):>8.4f}  {last_losses.get('safety_critic_loss', 0):>8.4f}  "
                  f"{last_losses.get('actor_rl_loss', 0):>10.4f}  {last_losses.get('physics_loss', 0):>9.4f}  "
                  f"{last_losses.get('lag_loss', 0):>9.4f}  {min_d_obs:>8.3f}  "
                  f"s={scene_id}  suc={log_success_count}")
            log_success_count = 0

        logger.log_episode_summary(
            step=total_steps, episode=episode,
            total_reward=ep_reward, min_d_obs=min_d_obs,
            avg_actor_loss=last_losses.get("actor_rl_loss", 0.0),
            avg_physics_loss=last_losses.get("physics_loss", 0.0),
            ep_step=ep_steps,
            alpha=last_losses.get("alpha"),
            avg_critic_loss=last_losses.get("critic_loss"),
            avg_safety_critic_loss=last_losses.get("safety_critic_loss"),
            avg_actor_total_loss=last_losses.get("actor_loss"),
            lag_loss=last_losses.get("lag_loss"),
            lag=last_losses.get("lag"),
            success=int(ep_success),
            ever_collided=int(ep_ever_collided),
            **avg_r,
        )

        ckpt_meta = {"training_protocol_version": TRAINING_PROTOCOL_VERSION,
                     "step": total_steps, "episode": episode, "best_reward": logger.best_reward,
                     "hyperparams": hyperparams, "csv_path": logger.csv_path}

        if total_steps >= next_checkpoint_step:
            save_training_checkpoint(
                agent, buffer, logger.checkpoint_path(f"step{total_steps:09d}"),
                ckpt_meta, per=args.per, save_replay=True,
            )
            while next_checkpoint_step <= total_steps:
                next_checkpoint_step += args.checkpoint_every_steps
        if ep_reward > logger.best_reward:
            logger.best_reward = ep_reward
            best_reward = logger.best_reward
            ckpt_meta["best_reward"] = best_reward
            agent.save(logger.checkpoint_path("best_reward"), metadata=ckpt_meta)

        if val_set is not None and total_steps >= next_val_step:
            print(f"\n{'='*60}\nValidation at episode {episode}\n{'='*60}")
            val_results = evaluate_on_validation_set(
                agent, env, val_set, num_scenes=args.val_scenes, max_steps=env.episode_len
            )
            print(f"Success Rate:      {val_results['success_rate']*100:.1f}%")
            print(f"Avg Reward:        {val_results['avg_reward']:.3f}")
            print(f"Avg Track Error:   {val_results['avg_tracking_error']:.4f}m")
            print(f"Avg Min Distance:  {val_results['avg_min_distance']:.4f}m")
            print(f"Collision Rate:    {val_results['collision_rate']*100:.1f}%")
            print(f"{'='*60}\n")
            logger.log_validation(total_steps, episode, val_results)
            if is_better_validation(val_results, best_validation):
                best_validation = val_results.copy()
                ckpt_meta["best_validation"] = best_validation
                agent.save(logger.checkpoint_path("best"), metadata=ckpt_meta)
            while next_val_step <= total_steps:
                next_val_step += args.val_every_steps

    return {"best_reward": best_reward, "total_steps": total_steps,
            "episode": episode, "best_validation": best_validation}


# ========================================================================
# Parallel SAC training loop
# ========================================================================

def _train_sac_parallel(agent, buffer, pool, ref_env, args, hyperparams, logger,
                         val_set, scene_ema, scene_counts, scene_weights, scene_lock,
                         total_steps=0, episode=0, best_reward=-np.inf,
                         best_validation=None):
    """Multi-environment SAC training with batched action selection."""
    n_envs = args.n_envs
    obs = pool.reset_all()
    for o in obs:
        agent.obs_normalizer.update(o)

    # Per-env tracking (re-initialized when an episode ends)
    env_rewards = np.zeros(n_envs)
    env_d_obs = [[] for _ in range(n_envs)]
    env_w = [[] for _ in range(n_envs)]
    env_r_acc = [reward_accumulators() for _ in range(n_envs)]
    env_collision_penalty = [[] for _ in range(n_envs)]
    env_ever_collided = [False for _ in range(n_envs)]
    env_steps = np.zeros(n_envs, dtype=int)
    log_success_count = 0
    next_val_step = ((total_steps // args.val_every_steps) + 1) * args.val_every_steps
    next_checkpoint_step = (
        (total_steps // args.checkpoint_every_steps) + 1
    ) * args.checkpoint_every_steps
    last_losses = {"actor_rl_loss": 0.0, "physics_loss": 0.0}

    # Force sigma=1 during start_steps so random actions actually explore
    if args.start_steps > 0:
        pool.broadcast_setattr("sigma_override", 1.0)
    sigma_overridden = True

    print(f"Run directory: {logger.run_dir}")
    print(f"{'Episode':^8}  {'Steps':^8}  {'Reward':^10}  {REWARD_HEADER}  "
          f"{'L_critic':^8}  {'L_scritic':^8}  "
          f"{'L_actor':^10}  {'L_dyn':^9}  {'L_lag':^9}  {'d_obs':^8}  {'suc':^5}")
    print("-" * 175)

    while total_steps < args.steps:
        # Clear sigma override once start_steps finishes
        if sigma_overridden and total_steps >= args.start_steps:
            pool.broadcast_setattr("sigma_override", None)
            sigma_overridden = False

        # Collect actions for all envs in parallel (batched)
        if total_steps < args.start_steps:
            if args.agent_type == "structured":
                actions = np.zeros((n_envs, ref_env.act_dim), dtype=np.float32)
                actions[:, 3:] = np.random.uniform(
                    -args.nullspace_scale, args.nullspace_scale,
                    (n_envs, ref_env.n - 3)
                )
            else:
                actions = np.random.uniform(
                    -2.175, 2.175, (n_envs, ref_env.act_dim)
                ).astype(np.float32)
        else:
            actions = agent.select_action_batch(obs)

        # Step all envs in parallel
        result = pool.step_all(actions)

        # Store the whole vector-environment result with one ring-buffer write.
        # PER retains its object-based insertion path.
        costs = np.asarray(
            [item.get("cost", 0.0) for item in result["info"]],
            dtype=np.float64,
        )
        if hasattr(buffer, "push_batch"):
            buffer.push_batch(
                obs, actions, result["reward"], result["obs"], result["done"],
                q=result["q_before"], dq=result["dq_before"],
                dq_next=result["dq_after"], J=result["J"],
                sigma=result["sigma"], dx_nom=result["dx_nom"], cost=costs,
            )

        # Track per-env metrics.
        for i in range(n_envs):
            if not hasattr(buffer, "push_batch"):
                buffer.push(
                    obs[i], actions[i], result["reward"][i],
                    result["obs"][i], result["done"][i],
                    q=result["q_before"][i], dq=result["dq_before"][i],
                    dq_next=result["dq_after"][i],
                    J=result["J"][i], sigma=result["sigma"][i],
                    dx_nom=result["dx_nom"][i], cost=costs[i],
                )
            total_steps += 1
            env_rewards[i] += result["reward"][i]
            env_d_obs[i].append(result["info"][i].get("d_obs", 0.0))
            env_w[i].append(result["info"][i].get("w", 0.0))
            accumulate_rewards(result["info"][i], env_r_acc[i])
            env_collision_penalty[i].append(result["info"][i].get("collision_penalty", 0.0))
            env_ever_collided[i] = env_ever_collided[i] or result["info"][i].get("collision", False)
            env_steps[i] += 1

            if result["done"][i]:
                episode += 1
                ep_success = result["info"][i].get("success", False)
                scene_id = result["scene_id"][i]
                min_d_obs = min(env_d_obs[i]) if env_d_obs[i] else 0.0
                avg_w = (sum(env_w[i]) / len(env_w[i])) if env_w[i] else None
                avg_r = avg_rewards(env_r_acc[i])
                avg_collision_penalty = (sum(env_collision_penalty[i]) / len(env_collision_penalty[i])) \
                    if env_collision_penalty[i] else None

                # Per-scene success EMA for difficulty-aware sampling.  A
                # uniform component prevents starvation and catastrophic
                # forgetting of easier scenes.
                if scene_ema is not None:
                    ema_alpha = 0.3
                    outcome = 1.0 if ep_success else 0.0
                    scene_ema[scene_id] = (ema_alpha * outcome +
                                           (1 - ema_alpha) * scene_ema[scene_id])
                    scene_counts[scene_id] += 1
                    if scene_counts.min() >= 1:
                        weights = scene_sampling_weights(
                            scene_ema, ALGORITHM.uniform_scene_mix
                        )
                        if scene_lock is not None:
                            scene_lock.acquire()
                        for s in range(len(weights)):
                            scene_weights[s] = weights[s]
                        if scene_lock is not None:
                            scene_lock.release()

                if ep_success:
                    log_success_count += 1

                # Logging
                if episode % args.log_every == 0:
                    rp = reward_print_values(avg_r)
                    print(f"{episode:>8d}  {total_steps:>8d}  {env_rewards[i]:>10.3f}  "
                          f"{REWARD_FORMAT.format(**rp)}  "
                          f"{last_losses.get('critic_loss', 0):>8.4f}  {last_losses.get('safety_critic_loss', 0):>8.4f}  "
                          f"{last_losses.get('actor_rl_loss', 0):>10.4f}  {last_losses.get('physics_loss', 0):>9.4f}  "
                          f"{last_losses.get('lag_loss', 0):>9.4f}  {min_d_obs:>8.3f}  "
                          f"s={scene_id}  suc={log_success_count}")
                    log_success_count = 0

                logger.log_episode_summary(
                    step=total_steps, episode=episode,
                    total_reward=env_rewards[i], min_d_obs=min_d_obs,
                    avg_actor_loss=last_losses.get("actor_rl_loss", 0.0),
                    avg_physics_loss=last_losses.get("physics_loss", 0.0),
                    ep_step=env_steps[i],
                    alpha=last_losses.get("alpha"),
                    avg_critic_loss=last_losses.get("critic_loss"),
                    avg_safety_critic_loss=last_losses.get("safety_critic_loss"),
                    avg_actor_total_loss=last_losses.get("actor_loss"),
                    lag_loss=last_losses.get("lag_loss"),
                    lag=last_losses.get("lag"),
                    avg_w=avg_w,
                    collision_penalty=avg_collision_penalty,
                    success=int(ep_success),
                    ever_collided=int(env_ever_collided[i]),
                    **avg_r,
                )

                ckpt_meta = {
                    "training_protocol_version": TRAINING_PROTOCOL_VERSION,
                    "step": total_steps, "episode": episode,
                    "best_reward": best_reward, "hyperparams": hyperparams,
                    "csv_path": logger.csv_path,
                    "scene_ema": scene_ema.tolist() if scene_ema is not None else None,
                    "scene_counts": scene_counts.tolist() if scene_counts is not None else None,
                }

                if total_steps >= next_checkpoint_step:
                    save_training_checkpoint(
                        agent, buffer, logger.checkpoint_path(f"step{total_steps:09d}"),
                        ckpt_meta, per=args.per, save_replay=True,
                    )
                    while next_checkpoint_step <= total_steps:
                        next_checkpoint_step += args.checkpoint_every_steps
                if env_rewards[i] > best_reward:
                    best_reward = env_rewards[i]
                    ckpt_meta["best_reward"] = best_reward
                    agent.save(logger.checkpoint_path("best_reward"), metadata=ckpt_meta)

                # Validation
                if val_set is not None and total_steps >= next_val_step:
                    print(f"\n{'='*60}")
                    print(f"Validation at episode {episode}")
                    print(f"{'='*60}")
                    val_results = evaluate_on_validation_set(
                        agent, ref_env, val_set,
                        num_scenes=args.val_scenes, max_steps=ref_env.episode_len
                    )
                    print(f"Success Rate:      {val_results['success_rate']*100:.1f}%")
                    print(f"Avg Reward:        {val_results['avg_reward']:.3f}")
                    print(f"Avg Track Error:   {val_results['avg_tracking_error']:.4f}m")
                    print(f"Avg Min Distance:  {val_results['avg_min_distance']:.4f}m")
                    print(f"Collision Rate:    {val_results['collision_rate']*100:.1f}%")
                    print(f"{'='*60}\n")
                    logger.log_validation(total_steps, episode, val_results)
                    if is_better_validation(val_results, best_validation):
                        best_validation = val_results.copy()
                        ckpt_meta["best_validation"] = best_validation
                        agent.save(logger.checkpoint_path("best"), metadata=ckpt_meta)
                    while next_val_step <= total_steps:
                        next_val_step += args.val_every_steps

                # Reset per-env tracking
                env_rewards[i] = 0.0
                env_d_obs[i] = []
                env_w[i] = []
                env_r_acc[i] = reward_accumulators()
                env_collision_penalty[i] = []
                env_ever_collided[i] = False
                env_steps[i] = 0

        obs = result["obs"]

        # Batch obs normalizer update
        agent.obs_normalizer.update(result["obs"])

        # SAC training update
        if total_steps >= args.start_steps and len(buffer) >= args.batch_size:
            for i in range(args.grad_steps):
                batch = buffer.sample(args.batch_size)
                losses, td_errors = agent.update(
                    batch, is_last=(i == args.grad_steps - 1),
                    actor_enabled=total_steps >= args.critic_warmup,
                )
                if args.per:
                    buffer.update_priorities(batch["indices"], td_errors)
                logger.log_update(losses)
                last_losses = losses

    return {"best_reward": best_reward, "total_steps": total_steps,
            "episode": episode, "best_validation": best_validation}


# ========================================================================
# Main
# ========================================================================

def main():
    args = parse_args()
    if args.val_every_steps <= 0 or args.checkpoint_every_steps <= 0:
        raise ValueError("step-based validation/checkpoint intervals must be positive")
    if args.grad_steps <= 0:
        raise ValueError("grad_steps must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Parse hidden_dims
    if hasattr(args, 'hidden_dims') and args.hidden_dims:
        args.hidden_dims = [int(x) for x in args.hidden_dims.replace(' ', '').split(',')]
    else:
        args.hidden_dims = [256, 256]

    # ---- Dynamics ----
    dyn = ManipulatorDynamics(args.urdf)

    # ---- Scene loading ----
    scene_data, n_obs, scene_weights, scene_ema, scene_counts, scene_lock, obs_waypoint_steps = \
        setup_scene_loading(args)

    # ---- Environment ----
    env_kwargs = make_env_kwargs(args, n_obs, obs_waypoint_steps)
    env_class = {
        "structured": ManipulatorEnv,
        "joint": VanillaEnv,
        "residual": ResidualEnv,
    }[args.agent_type]
    ref_env = env_class(**env_kwargs)
    state_dim = ref_env.obs_dim
    action_dim = ref_env.act_dim

    # ---- Validation set ----
    val_set = None
    if args.val_json is not None:
        val_path = args.val_json
        if not os.path.isabs(val_path):
            # First resolve relative to the launch directory (matching
            # --scene_json), then fall back to the repository root.
            cwd_path = os.path.abspath(val_path)
            repo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), val_path)
            val_path = cwd_path if os.path.exists(cwd_path) else repo_path
        if os.path.exists(val_path):
            val_set = ValidationSet(val_path)
            print(f"[train] Validation set: {len(val_set.scenes)} scenes available")
        else:
            print(f"Warning: Validation file not found at {val_path}")

    # ---- Device ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Agent + buffer ----
    agent, buffer = setup_agent_and_buffer(args, state_dim, action_dim, dyn, ref_env, device)

    # ---- Logger ----
    run_name = args.run_name or f"sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.save_path, run_name)
    hyperparams = {
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "steps": args.steps, "batch_size": args.batch_size,
        "start_steps": args.start_steps, "update_every": args.update_every,
        "critic_warmup_steps": args.critic_warmup,
        "grad_steps": args.grad_steps,
        "update_to_data_ratio": args.grad_steps / max(1, args.n_envs),
        "buffer_size": args.buffer_size,
        "lambda_dyn": args.lambda_dyn, "lr": args.lr,
        "alpha": args.alpha, "gamma": 0.99, "tau": args.tau,
        "state_dim": state_dim, "action_dim": action_dim,
        "episode_len": args.episode_len,
        "trajectory_steps": args.trajectory_steps,
    }
    logger = TrainingLogger(run_dir=run_dir, hyperparams=hyperparams)
    os.makedirs(args.save_path, exist_ok=True)

    # Save config
    _config = {
        "command": " ".join(sys.argv),
        "cli_args": vars(args),
        "hyperparams": hyperparams,
        "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(_config, f, indent=2, default=str)
    print(f"[train] Config saved to {run_dir}/config.json")

    # ---- Resume ----
    total_steps, episode, best_reward, best_validation = handle_resume(
        args, agent, buffer, scene_ema, scene_counts, scene_weights, logger
    )

    # ---- Launch training ----
    if args.render or args.n_envs <= 1:
        # ---- Single-env mode (render or single-process debug) ----
        env = env_class(**env_kwargs)
        if scene_data is not None:
            vs, scenes = scene_data
            if args.scene_id >= 0:
                vs.apply_scene_to_env(env, scenes)
                env.reset = lambda seed=None: (vs.apply_scene_to_env(env, scenes), env._get_obs())[1]
            else:
                vs.apply_scene_to_env(env, scenes[0])
                env.reset = lambda seed=None: (
                    vs.apply_scene_to_env(env, scenes[np.random.randint(len(scenes))]),
                    env._get_obs()
                )[1]
        print(f"[train] Single-env mode ({'render' if args.render else 'n_envs=1 debug'})")
        train_state = _run_render_loop(
            agent, env, buffer, args, hyperparams, logger, val_set,
            total_steps, episode, best_reward, best_validation
        )
        if hasattr(env, '_viewer'):
            env._viewer.close()
    else:
        # ---- Parallel mode ----
        from utils.parallel_env import ParallelEnvPool

        def _create_env():
            e = env_class(**env_kwargs)
            if scene_data is not None:
                vs, scenes = scene_data
                if args.scene_id >= 0:
                    vs.apply_scene_to_env(e, scenes)

                    def _reset_fixed(seed=None):
                        vs.apply_scene_to_env(e, scenes)
                        e._reset_episode_progress()
                        e._integral_err = np.zeros(3)
                        e._ever_collided = False
                        e.reward_fn._prev_dist_to_goal = None
                        e._last_sigma = 0.0
                        e._lag_lambda = 0.0
                        return e._get_obs()
                    e.reset = _reset_fixed
                else:
                    n_s = len(scenes)

                    def _sample_idx() -> int:
                        if scene_weights is not None:
                            if scene_lock is not None:
                                scene_lock.acquire()
                            raw = np.frombuffer(scene_weights.get_obj(), dtype=np.float64).copy()
                            if scene_lock is not None:
                                scene_lock.release()
                            raw = np.maximum(raw, 0.0)
                            total = raw.sum()
                            if total > 0:
                                probs = raw / total
                            else:
                                probs = np.ones(n_s, dtype=np.float64) / n_s
                            return int(np.random.choice(n_s, p=probs))
                        return int(np.random.randint(n_s))

                    init_idx = _sample_idx()
                    vs.apply_scene_to_env(e, scenes[init_idx])
                    e._current_scene_id = init_idx

                    def _reset(seed=None):
                        new_idx = _sample_idx()
                        vs.apply_scene_to_env(e, scenes[new_idx])
                        e._current_scene_id = new_idx
                        e._reset_episode_progress()
                        e._integral_err = np.zeros(3)
                        e._ever_collided = False
                        e.reward_fn._prev_dist_to_goal = None
                        e._last_sigma = 0.0
                        e._lag_lambda = 0.0
                        return e._get_obs()
                    e.reset = _reset
            return e

        pool = ParallelEnvPool(args.n_envs, _create_env, base_seed=args.seed)
        print(f"[train] Parallel mode: {args.n_envs} env workers")

        # Override total_steps/episode from resume (may differ from scratch)
        # agent and buffer are shared; pool resets envs automatically.
        train_state = _train_sac_parallel(
            agent, buffer, pool, ref_env, args, hyperparams, logger,
            val_set, scene_ema, scene_counts, scene_weights, scene_lock,
            total_steps, episode, best_reward, best_validation
        )
        pool.close()

    final_meta = {
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "step": train_state["total_steps"],
        "episode": train_state["episode"],
        "best_reward": train_state["best_reward"],
        "best_validation": train_state["best_validation"],
        "hyperparams": hyperparams,
        "csv_path": logger.csv_path,
    }
    save_training_checkpoint(
        agent, buffer, logger.checkpoint_path("final"), final_meta,
        per=args.per, save_replay=True,
    )
    best_reward = train_state["best_reward"]
    logger.close()
    print(f"\nTraining done. Best reward: {best_reward:.3f}")
    print(f"Run directory: {run_dir}")
    print(f"CSV log: {logger.csv_path}")


if __name__ == "__main__":
    main()
