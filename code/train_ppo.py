"""
train_ppo.py
------------
PPO (on-policy) training entry point for physics-informed RL.

Usage:
    python train_ppo.py --steps 500000 --n_envs 16 \\
        --rollout_steps 200 --ppo_epochs 10 \\
        --scene_json results/trajectories_obs.json

    # Resume from checkpoint:
    python train_ppo.py --resume checkpoints/run_name/ckpt_best.pt --steps 1000000

    # Load actor from SAC checkpoint:
    python train_ppo.py --load_sac_actor checkpoints/sac_run/ckpt_best.pt ...
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
from agent.ppo_agent import PPOAgent
from utils.logger import (TrainingLogger, REWARD_HEADER, REWARD_FORMAT,
                           reward_accumulators, accumulate_rewards,
                           avg_rewards, reward_print_values)
from utils.validation import ValidationSet, evaluate_on_validation_set


def parse_args():
    p = argparse.ArgumentParser()
    # Training
    p.add_argument("--steps",        type=int,   default=500_000)
    p.add_argument("--rollout_steps",type=int,   default=200)
    p.add_argument("--ppo_epochs",   type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--n_envs",       type=int,   default=16)
    p.add_argument("--episode_len",  type=int,   default=400)

    # Policy / regularization
    p.add_argument("--lambda_dyn",   type=float, default=1.0)
    p.add_argument("--task_scale",   type=float, default=1.0)
    p.add_argument("--nullspace_scale", type=float, default=0.5)
    p.add_argument("--hidden_dims",  type=str,   default="256,256")

    # Reward weights (read by env, not by agent)
    p.add_argument("--w_track",      type=float, default=12.0)
    p.add_argument("--w_obs",        type=float, default=5.0)
    p.add_argument("--w_collision",  type=float, default=100.0)
    p.add_argument("--w_manip",      type=float, default=0.05)
    p.add_argument("--w_energy",     type=float, default=0.001)
    p.add_argument("--w_action",     type=float, default=0.5)
    p.add_argument("--w_null",       type=float, default=0.0)
    p.add_argument("--d_safe",       type=float, default=0.06)
    p.add_argument("--success_bonus",type=float, default=50.0)
    p.add_argument("--reward_min",   type=float, default=None)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--path_deadzone",type=float, default=0.20)

    # Learning
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--tau",          type=float, default=0.005)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--lr_lag",       type=float, default=0.01)
    p.add_argument("--lag_target",   type=float, default=0.05)

    # Data
    p.add_argument("--scene_json",   type=str,   default=None)
    p.add_argument("--scene_id",     type=int,   default=-1)
    p.add_argument("--val_json",     type=str,   default=None)
    p.add_argument("--val_every",    type=int,   default=50)
    p.add_argument("--val_scenes",   type=int,   default=10)
    p.add_argument("--obs_scene_embed", type=int, default=0)
    p.add_argument("--obs_waypoint_steps", type=str, default=None)

    # Paths
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    _venv_data = os.path.join(_here, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                              "/share/example-robot-data/robots/panda_description")
    p.add_argument("--urdf", type=str, default=os.path.join(_venv_data, "urdf/panda.urdf"))
    p.add_argument("--xml",  type=str, default=os.path.join(_root, "models/panda_scene.xml"))
    p.add_argument("--save_path", type=str, default="checkpoints/ppo_pirl.pt")
    p.add_argument("--run_name",  type=str, default=None)

    # Logging / checkpoint
    p.add_argument("--log_every",        type=int, default=10)
    p.add_argument("--checkpoint_every", type=int, default=1000)
    p.add_argument("--no_collision_term", action="store_true")

    # Resume
    p.add_argument("--resume",         type=str, default=None)
    p.add_argument("--load_sac_actor", action="store_true")
    p.add_argument("--reset_actor",    action="store_true")

    return p.parse_args()


def setup_scene_loading(args):
    """Load scenes and set up prioritized sampling (if applicable)."""
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
            scene_ema = np.zeros(n_scenes, dtype=np.float64)
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
        frame_stack=1,  # PPO uses single-frame
        episode_len=args.episode_len,
        reward_min=args.reward_min,
        reward_scale=args.reward_scale,
        use_cbf=False, cbf_alpha=1.0,
    )


def build_parallel_pool(args, env_kwargs, scene_data, scene_weights, scene_lock):
    """Create ParallelEnvPool with scene-scene reset logic."""
    from utils.parallel_env import ParallelEnvPool

    def _create_env():
        e = ManipulatorEnv(**env_kwargs)
        if scene_data is not None:
            vs, scenes = scene_data
            if args.scene_id >= 0:
                vs.apply_scene_to_env(e, scenes)
                def _reset_fixed(seed=None):
                    vs.apply_scene_to_env(e, scenes)
                    e.step_count = 0
                    e._integral_err = np.zeros(3)
                    e._ever_collided = False
                    e.reward_fn._prev_dist_to_goal = None
                    e.path_param = 0.0
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
                    e.step_count = 0
                    e._integral_err = np.zeros(3)
                    e._ever_collided = False
                    e.reward_fn._prev_dist_to_goal = None
                    e.path_param = 0.0
                    e._last_sigma = 0.0
                    e._lag_lambda = 0.0
                    return e._get_obs()
                e.reset = _reset
        return e

    pool = ParallelEnvPool(args.n_envs, _create_env)
    print(f"[train] Parallel mode: {args.n_envs} env workers")
    return pool


def setup_agent(args, state_dim, action_dim, dyn, device):
    agent = PPOAgent(
        state_dim=state_dim, action_dim=action_dim, dynamics=dyn,
        n_envs=args.n_envs, rollout_steps=args.rollout_steps,
        lambda_dyn=args.lambda_dyn,
        task_scale=args.task_scale, nullspace_scale=args.nullspace_scale,
        ppo_epochs=args.ppo_epochs, batch_size=args.batch_size,
        device=device,
    )
    return agent


def main():
    args = parse_args()

    # Parse hidden_dims
    if hasattr(args, 'hidden_dims') and args.hidden_dims:
        args.hidden_dims = [int(x) for x in args.hidden_dims.replace(' ', '').split(',')]
    else:
        args.hidden_dims = [256, 256]

    # Dynamics
    dyn = ManipulatorDynamics(args.urdf)

    # Scene loading
    scene_data, n_obs, scene_weights, scene_ema, scene_counts, scene_lock, obs_waypoint_steps = \
        setup_scene_loading(args)

    # Env
    env_kwargs = make_env_kwargs(args, n_obs, obs_waypoint_steps)
    ref_env = ManipulatorEnv(**env_kwargs)
    state_dim = ref_env.obs_dim
    action_dim = ref_env.act_dim

    # Validation set
    val_set = None
    if args.val_json is not None:
        val_path = args.val_json
        if not os.path.isabs(val_path):
            val_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), val_path)
        if os.path.exists(val_path):
            val_set = ValidationSet(val_path)
            print(f"[train] Validation set: {len(val_set.scenes)} scenes available")
        else:
            print(f"Warning: Validation file not found at {val_path}")

    # Parallel pool
    pool = build_parallel_pool(args, env_kwargs, scene_data, scene_weights, scene_lock)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Agent
    agent = setup_agent(args, state_dim, action_dim, dyn, device)

    # Logger
    run_name = args.run_name or f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(os.path.dirname(args.save_path), run_name)
    hyperparams = {
        "steps": args.steps, "batch_size": args.batch_size,
        "rollout_steps": args.rollout_steps, "ppo_epochs": args.ppo_epochs,
        "lambda_dyn": args.lambda_dyn, "lr": args.lr, "gamma": args.gamma,
        "state_dim": state_dim, "action_dim": action_dim,
    }
    logger = TrainingLogger(run_dir=run_dir, hyperparams=hyperparams)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

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

    # Resume
    total_steps = 0
    episode = 0
    best_reward = -np.inf

    if args.resume is not None:
        ckpt_path = args.resume
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ckpt_path)
        if os.path.exists(ckpt_path):
            if args.load_sac_actor:
                meta = agent.load_actor_from_sac(ckpt_path)
            else:
                meta = agent.load(ckpt_path)
            total_steps = meta.get("step", 0)
            episode = meta.get("episode", 0)
            best_reward = meta.get("best_reward", -np.inf)
            logger.best_reward = best_reward

            # Restore per-scene stats
            if scene_ema is not None and "scene_ema" in meta and meta["scene_ema"] is not None \
                    and len(meta["scene_ema"]) == len(scene_ema):
                scene_ema[:] = meta["scene_ema"]
                scene_counts[:] = meta["scene_counts"]
                ema = scene_ema.copy()
                ema_min = ema.min()
                ema_max = ema.max()
                if ema_max > ema_min:
                    norm = (ema - ema_min) / (ema_max - ema_min + 1e-8)
                else:
                    norm = np.ones_like(ema) * 0.5
                weights = np.maximum(0.01, 1.0 - norm)
                weights = weights / weights.sum()
                for s in range(len(weights)):
                    scene_weights[s] = weights[s]
                print(f"[train] Restored scene performance stats for {len(scene_ema)} scenes")

            print(f"[train] Resumed from {ckpt_path}: step={total_steps}, episode={episode}, "
                  f"best_reward={best_reward:.3f}")

            if args.reset_actor:
                pass  # handled by load
        else:
            print(f"[train] WARNING: resume checkpoint not found: {ckpt_path}")

    # ================================================================
    # Training loop
    # ================================================================
    n_envs = args.n_envs
    obs = pool.reset_all()

    env_rewards = np.zeros(n_envs)
    env_d_obs = [[] for _ in range(n_envs)]
    env_w = [[] for _ in range(n_envs)]
    env_r_acc = [reward_accumulators() for _ in range(n_envs)]
    env_collision_penalty = [[] for _ in range(n_envs)]
    env_ever_collided = [False for _ in range(n_envs)]
    env_steps = np.zeros(n_envs, dtype=int)
    log_success_count = 0
    last_val_ep = -1
    last_losses = {"actor_rl_loss": 0.0, "physics_loss": 0.0}

    print(f"Run directory: {run_dir}")
    print(f"{'Episode':^8}  {'Steps':^8}  {'Reward':^10}  {REWARD_HEADER}  "
          f"{'L_actor':^10}  {'L_dyn':^9}  {'d_obs':^8}  {'suc':^5}")
    print("-" * 140)

    while total_steps < args.steps:
        agent.buffer.clear()

        # --- Rollout collection ---
        for _ in range(args.rollout_steps):
            if total_steps >= args.steps:
                break

            actions = np.zeros((n_envs, action_dim), dtype=np.float32)
            log_probs = np.zeros(n_envs, dtype=np.float32)
            values = np.zeros(n_envs, dtype=np.float32)
            for i in range(n_envs):
                actions[i], log_probs[i], values[i] = agent.act(obs[i])

            result = pool.step_all(actions)

            agent.buffer.push(
                obs, actions, result["reward"], result["done"],
                log_probs, values,
                q=result["q_before"], dq=result["dq_before"],
                dq_next=result["dq_after"],
                J=result["J"], sigma=result["sigma"],
                dx_nom=result["dx_nom"],
            )

            for i in range(n_envs):
                total_steps += 1
                env_rewards[i] += result["reward"][i]
                info_i = result["info"][i]
                env_d_obs[i].append(info_i.get("d_obs", 0.0))
                env_w[i].append(info_i.get("w", 0.0))
                accumulate_rewards(info_i, env_r_acc[i])
                env_collision_penalty[i].append(info_i.get("collision_penalty", 0.0))
                env_ever_collided[i] = env_ever_collided[i] or info_i.get("collision", False)
                env_steps[i] += 1
                agent.obs_normalizer.update(result["obs"][i])

                if result["done"][i]:
                    episode += 1
                    ep_success = info_i.get("success", False)
                    scene_id = result["scene_id"][i]
                    min_d_obs = min(env_d_obs[i]) if env_d_obs[i] else 0.0
                    avg_w = (sum(env_w[i]) / len(env_w[i])) if env_w[i] else None
                    avg_r = avg_rewards(env_r_acc[i])
                    avg_collision_penalty = (sum(env_collision_penalty[i]) / len(env_collision_penalty[i])) \
                        if env_collision_penalty[i] else None

                    # Per-scene EMA
                    if scene_ema is not None:
                        ema_alpha = 0.3
                        scene_ema[scene_id] = ema_alpha * env_rewards[i] + (1 - ema_alpha) * scene_ema[scene_id]
                        scene_counts[scene_id] += 1
                        if scene_counts.min() >= 1:
                            ema = scene_ema.copy()
                            ema_min = ema.min()
                            ema_max = ema.max()
                            if ema_max > ema_min:
                                norm = (ema - ema_min) / (ema_max - ema_min + 1e-8)
                            else:
                                norm = np.ones_like(ema) * 0.5
                            weights = np.maximum(0.01, 1.0 - norm)
                            weights = weights / weights.sum()
                            for s in range(len(weights)):
                                scene_weights[s] = weights[s]

                    if ep_success:
                        log_success_count += 1

                    if episode % args.log_every == 0:
                        rp = reward_print_values(avg_r)
                        avg_l_actor = last_losses.get("actor_rl_loss", 0.0)
                        avg_l_dyn = last_losses.get("physics_loss", 0.0)
                        print(f"{episode:>8d}  {total_steps:>8d}  {env_rewards[i]:>10.3f}  "
                              f"{REWARD_FORMAT.format(**rp)}  "
                              f"{avg_l_actor:>10.4f}  {avg_l_dyn:>9.4f}  {min_d_obs:>8.3f}  "
                              f"s={scene_id}  suc={log_success_count}")
                        log_success_count = 0

                    logger.log_episode_summary(
                        step=total_steps, episode=episode,
                        total_reward=env_rewards[i], min_d_obs=min_d_obs,
                        avg_actor_loss=last_losses.get("actor_rl_loss", 0.0),
                        avg_physics_loss=last_losses.get("physics_loss", 0.0),
                        ep_step=env_steps[i],
                        avg_w=avg_w,
                        collision_penalty=avg_collision_penalty,
                        success=int(ep_success),
                        ever_collided=int(env_ever_collided[i]),
                        **avg_r,
                    )

                    ckpt_meta = {
                        "step": total_steps, "episode": episode,
                        "best_reward": best_reward, "hyperparams": hyperparams,
                        "csv_path": logger.csv_path,
                        "scene_ema": scene_ema.tolist() if scene_ema is not None else None,
                        "scene_counts": scene_counts.tolist() if scene_counts is not None else None,
                    }

                    if episode % args.checkpoint_every == 0:
                        agent.save(logger.checkpoint_path(f"ep{episode:05d}"), metadata=ckpt_meta)

                    if env_rewards[i] > best_reward:
                        best_reward = env_rewards[i]
                        ckpt_meta["best_reward"] = best_reward
                        agent.save(logger.checkpoint_path("best"), metadata=ckpt_meta)

                    # Validation
                    if val_set is not None and episode > 0 and episode % args.val_every == 0 \
                            and episode != last_val_ep:
                        last_val_ep = episode
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
                        logger.log_validation(episode, val_results)

                    # Reset per-env
                    env_rewards[i] = 0.0
                    env_d_obs[i] = []
                    env_w[i] = []
                    env_r_acc[i] = reward_accumulators()
                    env_collision_penalty[i] = []
                    env_ever_collided[i] = False
                    env_steps[i] = 0
            obs = result["obs"]

        # --- GAE + PPO update ---
        if len(agent.buffer) > 0:
            last_values = agent.get_value(obs)
            agent.buffer.compute_advantages(last_values)
            losses = agent.update()
            last_losses = losses

    # Cleanup
    logger.close()
    pool.close()
    print(f"\nTraining done. Best reward: {best_reward:.3f}")
    print(f"Run directory: {run_dir}")
    print(f"CSV log: {logger.csv_path}")


if __name__ == "__main__":
    main()
