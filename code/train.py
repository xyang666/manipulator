"""
train.py
--------
Training entry point for physics-informed SAC on the manipulator env.

Usage:
    cd code/
    python train.py [--steps 50000] [--urdf path/to/panda.urdf]

Prints per-episode:
    episode | reward | L_RL | L_dyn | d_obs_min
"""

import torch
import argparse
import sys
import os
import numpy as np
from datetime import datetime

# Allow imports from code/ root
sys.path.insert(0, os.path.dirname(__file__))

from env.manipulator_env import ManipulatorEnv
from env.dynamics import ManipulatorDynamics
from agent.sac_agent import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.logger import TrainingLogger
from utils.validation import ValidationSet, evaluate_on_validation_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",       type=int,   default=500_000,
                   help="Total environment steps (SAC typically needs 500k-1M)")
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--start_steps", type=int,   default=2_000,
                   help="Random exploration steps before training begins")
    p.add_argument("--update_every",type=int,   default=1)
    p.add_argument("--grad_steps",  type=int,   default=4,
                   help="Number of gradient updates per env step")
    p.add_argument("--buffer_size", type=int,   default=100_000)
    p.add_argument("--lambda_dyn",  type=float, default=1.0,
                   help="Weight of physics regularization loss")
    p.add_argument("--d_critical",  type=float, default=0.05,
                   help="Critical distance for primary task relaxation (m)")
    p.add_argument("--alpha_relax", type=float, default=0.1,
                   help="Minimum tracking weight factor when d_obs < d_critical")
    p.add_argument("--val_json",    type=str,   default=None,
                   help="Path to validation trajectories JSON file")
    p.add_argument("--val_every",   type=int,   default=50,
                   help="Evaluate on validation set every N episodes")
    p.add_argument("--val_scenes",  type=int,   default=10,
                   help="Number of validation scenes to evaluate")
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    _venv_data = os.path.join(_here, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                              "/share/example-robot-data/robots/panda_description")
    _default_urdf = os.path.join(_venv_data, "urdf/panda.urdf")
    _default_xml  = os.path.join(_root, "models/panda_scene.xml")

    p.add_argument("--urdf",        type=str,   default=_default_urdf,
                   help="Path to robot URDF for Pinocchio kinematics/dynamics")
    p.add_argument("--xml",         type=str,   default=_default_xml,
                   help="Path to MuJoCo scene XML (None = kinematics-only mode)")
    p.add_argument("--save_path",   type=str,   default="checkpoints/sac_pirl.pt")
    p.add_argument("--log_every",   type=int,   default=10)
    p.add_argument("--checkpoint_every", type=int, default=50,
                   help="Save a periodic checkpoint every N episodes")
    p.add_argument("--run_name",    type=str,   default=None,
                   help="Run directory name; auto-generated if not set")
    p.add_argument("--scene_json", type=str,   default=None,
                   help="Path to JSON with scenes (for fixed-scene training)")
    p.add_argument("--scene_id",   type=int,   default=0,
                   help="Scene ID to use when --scene_json is set")
    p.add_argument("--render",      action="store_true",
                   help="Render the scene with MuJoCo viewer during training")
    return p.parse_args()


def main():
    args = parse_args()

    # -------- Setup --------
    dyn = ManipulatorDynamics(args.urdf)

    # If fixed scene mode, load scene first to get correct obstacle count
    fixed_scene = None
    if args.scene_json is not None:
        _vs = ValidationSet(args.scene_json)
        fixed_scene = _vs.get_scene(args.scene_id)
        n_obs = len(fixed_scene["obstacles"])
        print(f"[train] Fixed scene mode: scene_id={args.scene_id}, "
              f"start={fixed_scene['start']}, goal={fixed_scene['goal']}, "
              f"obstacles={n_obs}")
    else:
        n_obs = 5

    env = ManipulatorEnv(urdf_path=args.urdf, xml_path=args.xml, obs_radius=0.03,
                         n_obstacles=n_obs,
                         use_trajectory_generator=fixed_scene is None,
                         d_critical=args.d_critical, alpha_relax=args.alpha_relax)

    # Apply fixed scene and patch reset to skip random generation
    if fixed_scene is not None:
        _vs.apply_scene_to_env(env, fixed_scene)
        env.reset = (lambda _vs, _scene:
                     lambda seed=None: (_vs.apply_scene_to_env(env, _scene), env._get_obs())[1]
                     )(_vs, fixed_scene)

    state_dim  = env.obs_dim
    action_dim = env.act_dim

    # -------- Load validation set --------
    val_set = None
    if args.val_json is not None:
        val_json_path = args.val_json
        if not os.path.isabs(val_json_path):
            val_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), val_json_path)
        if os.path.exists(val_json_path):
            val_set = ValidationSet(val_json_path)
            if fixed_scene is not None:
                print(f"[train] Validation set: {len(val_set.scenes)} scenes available")
        else:
            print(f"Warning: Validation file not found at {val_json_path}")

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        dynamics=dyn,
        lambda_dyn=args.lambda_dyn,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        critic_warmup=5000,
        total_steps=args.steps,
    )
    buffer = ReplayBuffer(args.buffer_size, state_dim, action_dim)

    # -------- Logger / run directory --------
    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir  = os.path.join(os.path.dirname(args.save_path), run_name)
    hyperparams = {
        "steps":        args.steps,
        "batch_size":   args.batch_size,
        "start_steps":  args.start_steps,
        "update_every": args.update_every,
        "buffer_size":  args.buffer_size,
        "lambda_dyn":   args.lambda_dyn,
        "d_critical":   args.d_critical,
        "alpha_relax":  args.alpha_relax,
        "lr":           1e-4,
        "gamma":        0.99,
        "tau":          0.005,
        "state_dim":    state_dim,
        "action_dim":   action_dim,
    }
    logger = TrainingLogger(run_dir=run_dir, hyperparams=hyperparams)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # -------- Training loop --------
    total_steps = 0
    episode     = 0
    best_reward = -np.inf
    reward_scale = 2.0  # normalize reward magnitude for stable Q learning

    print(f"Run directory: {run_dir}")
    print(f"{'Episode':>8} {'Steps':>8} {'Reward':>10} "
          f"{'L_actor':>10} {'L_dyn':>10} {'d_obs':>8}")
    print("-" * 60)

    while total_steps < args.steps:
        obs = env.reset()
        agent.obs_normalizer.update(obs)
        ep_reward   = 0.0
        ep_l_actor  = 0.0
        ep_l_dyn    = 0.0
        ep_d_obs    = []
        ep_steps    = 0
        done        = False
        while not done:
            # Action selection
            if total_steps < args.start_steps:
                # Random exploration: separate ranges for task relaxation (3D) and null-space (7D)
                a_task = np.random.uniform(-0.1, 0.1, 3)    # position relaxation (Route A)
                a_null = np.random.uniform(-0.3, 0.3, env.n) # larger null-space motion
                action = np.concatenate([a_task, a_null])
            else:
                action = agent.select_action(obs)

            # Store kinematics for physics loss (7D joint state, not action-dim)
            q_prev  = env.q.copy()
            dq_prev = env.dq.copy()

            next_obs, reward, done, info = env.step(action)

            if args.render:
                env.render()

            logger.log_step(total_steps, episode, ep_steps, reward, info)

            dq_next = env.dq.copy()

            # Update observation normalizer and scale reward
            agent.obs_normalizer.update(next_obs)
            reward_scaled = reward / reward_scale

            buffer.push(
                obs, action, reward_scaled, next_obs, done,
                q=q_prev, dq=dq_prev, dq_next=dq_next,
                J=env._last_J, sigma=env._last_sigma, dx_nom=env._last_dx_nom
            )

            obs = next_obs
            ep_reward += reward
            ep_d_obs.append(info["d_obs"])
            total_steps += 1
            ep_steps    += 1

            # Training update (multiple gradient steps per env step)
            if (total_steps >= args.start_steps and
                    len(buffer) >= args.batch_size and
                    total_steps % args.update_every == 0):

                for _ in range(args.grad_steps):
                    batch = buffer.sample(args.batch_size)
                    losses = agent.update(batch)
                    logger.log_update(losses)
                    ep_l_actor += losses["actor_rl_loss"]
                    ep_l_dyn   += losses["physics_loss"]

        episode += 1
        ep_summary = logger.end_episode(episode, total_steps)
        avg_l_actor = ep_l_actor / max(ep_steps, 1)
        avg_l_dyn   = ep_l_dyn   / max(ep_steps, 1)
        min_d_obs   = min(ep_d_obs) if ep_d_obs else 0.0

        if episode % args.log_every == 0:
            print(f"{episode:>8d} {total_steps:>8d} {ep_reward:>10.3f} "
                  f"{avg_l_actor:>10.4f} {avg_l_dyn:>10.4f} {min_d_obs:>8.3f}")

        ckpt_meta = {
            "step":         total_steps,
            "episode":      episode,
            "best_reward":  logger.best_reward,
            "hyperparams":  hyperparams,
            "csv_path":     logger.csv_path,
        }

        # Periodic checkpoint
        if episode % args.checkpoint_every == 0:
            agent.save(logger.checkpoint_path(f"ep{episode:05d}"), metadata=ckpt_meta)

        # Best checkpoint
        if ep_summary["total_reward"] > logger.best_reward:
            logger.best_reward = ep_summary["total_reward"]
            best_reward = logger.best_reward
            ckpt_meta["best_reward"] = logger.best_reward
            agent.save(logger.checkpoint_path("best"), metadata=ckpt_meta)

        # Validation evaluation
        if val_set is not None and episode % args.val_every == 0:
            print(f"\n{'='*60}")
            print(f"Validation at episode {episode}")
            print(f"{'='*60}")

            val_results = evaluate_on_validation_set(
                agent, env, val_set,
                num_scenes=args.val_scenes,
                max_steps=env.episode_len
            )

            print(f"Success Rate:      {val_results['success_rate']*100:.1f}%")
            print(f"Avg Reward:        {val_results['avg_reward']:.3f}")
            print(f"Avg Track Error:   {val_results['avg_tracking_error']:.4f}m")
            print(f"Avg Min Distance:  {val_results['avg_min_distance']:.4f}m")
            print(f"Collision Rate:    {val_results['collision_rate']*100:.1f}%")
            print(f"{'='*60}\n")

            # Log validation results
            logger.log_validation(episode, val_results)

    logger.close()
    print(f"\nTraining done. Best reward: {best_reward:.3f}")
    print(f"Run directory: {run_dir}")
    print(f"CSV log: {logger.csv_path}")


if __name__ == "__main__":
    main()
