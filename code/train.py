"""
train.py
--------
Training entry point for physics-informed RL on the manipulator env.
Supports SAC (off-policy) and PPO (on-policy) algorithms.

Usage:
    cd code/

    # SAC training (default):
    python train.py --steps 500000 --n_envs 16 --scene_json results/trajectories_obs.json

    # PPO training:
    python train.py --algo ppo --steps 500000 --n_envs 16 --rollout_steps 200 --ppo_epochs 10 \\
                    --scene_json results/trajectories_obs.json

    # Resume from checkpoint:
    python train.py --resume checkpoints/run_name/ckpt_best.pt --steps 1000000

    # Validation only:
    python train.py --resume checkpoints/run_name/ckpt_best.pt --val_json results/trajectories_obs.json

Key arguments:
    --algo sac|ppo              RL algorithm (default: sac)
    --steps N                   Total environment steps (default: 500000)
    --n_envs N                  Parallel environments (default: 16)
    --scene_json path           JSON with training scenes
    --val_json path             JSON with validation scenes (optional)
    --rollout_steps N           PPO: steps per rollout (default: 200)
    --ppo_epochs N              PPO: training epochs per rollout (default: 10)
    --render                    Single-env mode with MuJoCo viewer

Prints per-episode:
    episode | steps | reward | L_actor | L_dyn | d_obs [scene_id]
"""

import json
import torch
import argparse
import sys
import os
import numpy as np
from datetime import datetime
from multiprocessing import Array

# Allow imports from code/ root
sys.path.insert(0, os.path.dirname(__file__))

from env.manipulator_env import ManipulatorEnv
from env.dynamics import ManipulatorDynamics
from agent.sac_agent import SACAgent
from agent.ppo_agent import PPOAgent
from utils.replay_buffer import ReplayBuffer
from utils.logger import TrainingLogger
from utils.validation import ValidationSet, evaluate_on_validation_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",       type=int,   default=500_000,
                   help="Total environment steps (SAC typically needs 500k-1M)")
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--start_steps", type=int,   default=10_000,
                   help="Random exploration steps before training begins")
    p.add_argument("--update_every",type=int,   default=1)
    p.add_argument("--grad_steps",  type=int,   default=4,
                   help="Number of gradient updates per env step")
    p.add_argument("--buffer_size", type=int,   default=500_000)
    p.add_argument("--lambda_dyn",  type=float, default=1.0,
                   help="Weight of physics regularization loss")
    p.add_argument("--d_critical",  type=float, default=0.05,
                   help="Critical distance for primary task relaxation (m)")
    p.add_argument("--alpha_relax", type=float, default=0.1,
                   help="Minimum tracking weight factor when d_obs < d_critical")
    p.add_argument("--w_obs", type=float, default=5.0,
                   help="Obstacle proximity penalty weight")
    p.add_argument("--w_obs_safe", type=float, default=0.1,
                   help="Safe-zone positive reward weight")
    p.add_argument("--w_collision", type=float, default=100.0,
                   help="Collision contact penalty weight")
    p.add_argument("--w_track", type=float, default=3.0,
                   help="Tracking error penalty weight")
    p.add_argument("--d_safe", type=float, default=0.06,
                   help="Safe distance threshold for obstacle reward (m)")
    p.add_argument("--success_bonus", type=float, default=50.0,
                   help="Sparse success bonus upon reaching goal")
    p.add_argument("--w_goal", type=float, default=1.0,
                   help="Dense goal-progress reward weight")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate for actor/critic/alpha optimizers")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Initial SAC entropy coefficient")
    p.add_argument("--critic_warmup", type=int, default=5000,
                   help="Number of critic-only updates before actor starts training")
    p.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"],
                   help="RL algorithm: sac (off-policy) or ppo (on-policy)")
    p.add_argument("--rollout_steps", type=int, default=200,
                   help="PPO: steps per rollout collection (default: episode_len)")
    p.add_argument("--ppo_epochs", type=int, default=10,
                   help="PPO: number of training epochs per rollout")
    p.add_argument("--val_json",       type=str, default=None,
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
    p.add_argument("--checkpoint_every", type=int, default=500,
                   help="Save a periodic checkpoint every N episodes")
    p.add_argument("--run_name",    type=str,   default=None,
                   help="Run directory name; auto-generated if not set")
    p.add_argument("--scene_json", type=str,   default=None,
                   help="Path to JSON with scenes (for fixed-scene training)")
    p.add_argument("--scene_id",   type=int,   default=-1,
                   help="Scene ID (>=0 = fixed scene, -1 = random cycle through all scenes)")
    p.add_argument("--n_envs",      type=int,   default=16,
                   help="Number of parallel environment workers (>>1 = faster GPU utilization)")
    p.add_argument("--n_critics",   type=int,   default=5,
                   help="Number of Q-networks in ensemble critic (default 5, 2=standard SAC)")
    p.add_argument("--hidden_dims", type=str,   default="256,256",
                   help="Hidden layer sizes for actor/critic networks (comma-separated, e.g. '512,512,512')")
    p.add_argument("--per",         action="store_true",
                   help="Use Prioritized Experience Replay instead of uniform sampling")
    p.add_argument("--episode_len", type=int,   default=400,
                   help="Max steps per episode (default: 400; use more for obstacle avoidance)")
    p.add_argument("--reward_scale", type=float, default=1.0,
                   help="Reward scaling factor: rewards are divided by this before storing in buffer. "
                        "Use >1 to compress Q-values for stable SAC training (e.g. 50).")
    p.add_argument("--render",      action="store_true",
                   help="Render the scene with MuJoCo viewer during training")
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--reset_alpha", action="store_true",
                   help="When resuming, reset log_alpha to match --alpha (overrides checkpoint value)")
    p.add_argument("--reset_critic", action="store_true",
                   help="When resuming, reinitialize critic (for architecture changes like LayerNorm)")
    p.add_argument("--reset_actor",  action="store_true",
                   help="When resuming, reinitialize actor (for architecture changes like hidden_dims)")
    p.add_argument("--load_sac_actor", action="store_true",
                   help="PPO: load actor from SAC checkpoint (ignores critic/value weights)")
    p.add_argument("--path_deadzone", type=float, default=0.20,
                   help="Deadzone for path progression (m). Larger = more deviation allowed before stalling")
    p.add_argument("--no_collision_term", action="store_true",
                   help="Disable collision-based episode termination")
    return p.parse_args()


def main():
    args = parse_args()

    # Parse hidden_dims from comma-separated string
    if hasattr(args, 'hidden_dims') and args.hidden_dims:
        args.hidden_dims = [int(x) for x in args.hidden_dims.replace(' ', '').split(',')]
    else:
        args.hidden_dims = [256, 256]

    # -------- Setup --------
    dyn = ManipulatorDynamics(args.urdf)

    # If scene JSON mode, load scenes (fixed or random cycle)
    _scene_data = None  # (ValidationSet, scene_or_scenes)
    if args.scene_json is not None:
        _vs = ValidationSet(args.scene_json)
        if args.scene_id >= 0:
            # Fixed single scene
            _scene_data = (_vs, _vs.get_scene(args.scene_id))
            n_obs = len(_scene_data[1]["obstacles"])
            print(f"[train] Fixed scene mode: scene_id={args.scene_id}, "
                  f"obstacles={n_obs}")
        else:
            # Random cycle through all scenes
            _scene_data = (_vs, _vs.scenes)
            n_obs = len(_vs.scenes[0]["obstacles"])
            print(f"[train] Scene cycle mode: {len(_vs.scenes)} scenes, "
                  f"obs/scene={n_obs}")

        # Prioritized scene sampling: shared weights for parallel workers
        _scene_weights = None
        _scene_ema = None
        _scene_counts = None
        if args.n_envs > 1 and _scene_data is not None and args.scene_id < 0:
            n_scenes = len(_scene_data[1])
            _scene_weights = Array('d', [1.0] * n_scenes)
            _scene_ema = np.zeros(n_scenes, dtype=np.float64)
            _scene_counts = np.zeros(n_scenes, dtype=np.int32)
    else:
        n_obs = 5
        _scene_weights = None
        _scene_ema = None
        _scene_counts = None

    # -------- Environment setup --------
    _env_kwargs = dict(
        urdf_path=args.urdf, xml_path=args.xml, obs_radius=0.03,
        n_obstacles=n_obs,
        use_trajectory_generator=_scene_data is None,
        d_critical=args.d_critical, alpha_relax=args.alpha_relax,
        collision_term=not args.no_collision_term,
        path_deadzone=args.path_deadzone,
        w_obs=args.w_obs, w_obs_safe=args.w_obs_safe,
        w_collision=args.w_collision, w_track=args.w_track,
        w_goal=args.w_goal,
        d_safe=args.d_safe, success_bonus=args.success_bonus,
        episode_len=args.episode_len,
    )

    # Reference env for dimension / attribute access
    ref_env = ManipulatorEnv(**_env_kwargs)
    state_dim = ref_env.obs_dim
    action_dim = ref_env.act_dim

    # -------- Load validation set --------
    val_set = None
    if args.val_json is not None:
        val_json_path = args.val_json
        if not os.path.isabs(val_json_path):
            val_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), val_json_path)
        if os.path.exists(val_json_path):
            val_set = ValidationSet(val_json_path)
            print(f"[train] Validation set: {len(val_set.scenes)} scenes available")
        else:
            print(f"Warning: Validation file not found at {val_json_path}")

    # -------- Environment (single for render, parallel pool otherwise) --------
    if args.render:
        env = ManipulatorEnv(**_env_kwargs)
        if _scene_data is not None:
            _vs, _scenes = _scene_data
            if args.scene_id >= 0:
                _vs.apply_scene_to_env(env, _scenes)
                env.reset = lambda seed=None: (_vs.apply_scene_to_env(env, _scenes), env._get_obs())[1]
            else:
                _vs.apply_scene_to_env(env, _scenes[0])
                env.reset = lambda seed=None: (
                    _vs.apply_scene_to_env(env, _scenes[np.random.randint(len(_scenes))]),
                    env._get_obs()
                )[1]
        pool = None
        print(f"[train] Single-env mode (--render)")
    else:
        from utils.parallel_env import ParallelEnvPool

        def _create_env():
            e = ManipulatorEnv(**_env_kwargs)
            if _scene_data is not None:
                _vs, _scenes = _scene_data
                if args.scene_id >= 0:
                    _vs.apply_scene_to_env(e, _scenes)
                    def _reset_fixed(seed=None):
                        _vs.apply_scene_to_env(e, _scenes)
                        e._reset_state()
                        e.path_param = 0.0
                        return e._get_obs()
                    e.reset = _reset_fixed
                else:
                    n_s = len(_scenes)

                    def _sample_idx() -> int:
                        if _scene_weights is not None:
                            raw = np.frombuffer(
                                _scene_weights.get_obj(), dtype=np.float64
                            ).copy()
                            raw = np.maximum(raw, 0.0)
                            total = raw.sum()
                            if total > 0:
                                probs = raw / total
                            else:
                                probs = np.ones(n_s, dtype=np.float64) / n_s
                            return int(np.random.choice(n_s, p=probs))
                        return int(np.random.randint(n_s))

                    init_idx = _sample_idx()
                    _vs.apply_scene_to_env(e, _scenes[init_idx])
                    e._current_scene_id = init_idx

                    def _reset(seed=None):
                        new_idx = _sample_idx()
                        _vs.apply_scene_to_env(e, _scenes[new_idx])
                        e._current_scene_id = new_idx
                        e._reset_state()
                        e.path_param = 0.0
                        return e._get_obs()
                    e.reset = _reset
            return e

        pool = ParallelEnvPool(args.n_envs, _create_env)
        env = None
        print(f"[train] Parallel mode: {args.n_envs} env workers")

    # -------- Agent, replay buffer, logger (CUDA init after fork) --------
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.algo == "ppo":
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            dynamics=dyn,
            n_envs=args.n_envs,
            rollout_steps=args.rollout_steps,
            lambda_dyn=args.lambda_dyn,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            device=_device,
        )
        buffer = None  # PPO uses internal RolloutBuffer
    else:
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            dynamics=dyn,
            hidden_dims=args.hidden_dims,
            lambda_dyn=args.lambda_dyn,
            lr=args.lr,
            alpha=args.alpha,
            device=_device,
            critic_warmup=max(1, args.critic_warmup // args.n_envs),
            total_steps=args.steps,
            n_critics=args.n_critics,
        )
        if args.per:
            from utils.replay_buffer import PrioritizedReplayBuffer
            buffer = PrioritizedReplayBuffer(args.buffer_size, state_dim, action_dim)
        else:
            buffer = ReplayBuffer(args.buffer_size, state_dim, action_dim)

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
        "lr":           args.lr,
        "alpha":        args.alpha,
        "gamma":        0.99,
        "tau":          0.005,
        "state_dim":    state_dim,
        "action_dim":   action_dim,
    }
    logger = TrainingLogger(run_dir=run_dir, hyperparams=hyperparams)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Save config for reproducibility
    try:
        from agent.reward import RewardFunction
        import inspect
        reward_defaults = {
            k: v.default for k, v in inspect.signature(RewardFunction.__init__).parameters.items()
            if v.default is not inspect.Parameter.empty and k not in ('self', 'collision_detector')
        }
    except Exception:
        reward_defaults = {}
    _config = {
        "command": " ".join(sys.argv),
        "cli_args": vars(args),
        "hyperparams": hyperparams,
        "reward_weights": reward_defaults,
        "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(_config, f, indent=2, default=str)
    print(f"[train] Config saved to {run_dir}/config.json")

    # -------- Resume from checkpoint --------
    total_steps = 0
    episode     = 0
    best_reward = -np.inf
    if args.resume is not None:
        ckpt_path = args.resume
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ckpt_path)
        if os.path.exists(ckpt_path):
            if args.load_sac_actor:
                if args.algo != "ppo":
                    print("[train] WARNING: --load_sac_actor only supported for PPO. Ignoring.")
                    meta = agent.load(ckpt_path, reset_alpha=args.reset_alpha,
                                      reset_critic=args.reset_critic, reset_actor=args.reset_actor)
                else:
                    meta = agent.load_actor_from_sac(ckpt_path)
            else:
                meta = agent.load(ckpt_path, reset_alpha=args.reset_alpha,
                                  reset_critic=args.reset_critic, reset_actor=args.reset_actor)
            total_steps = meta.get("step", 0)
            episode     = meta.get("episode", 0)
            best_reward = meta.get("best_reward", -np.inf)
            logger.best_reward = best_reward

            # Restore per-scene stats (skip if scene count mismatched — e.g. phase upgrade)
            if (_scene_ema is not None and "scene_ema" in meta
                    and meta["scene_ema"] is not None
                    and len(meta["scene_ema"]) == len(_scene_ema)):
                _scene_ema[:] = meta["scene_ema"]
                _scene_counts[:] = meta["scene_counts"]
                ema = _scene_ema.copy()
                ema_min = ema.min()
                ema_max = ema.max()
                if ema_max > ema_min:
                    norm = (ema - ema_min) / (ema_max - ema_min + 1e-8)
                else:
                    norm = np.ones_like(ema) * 0.5
                weights = np.maximum(0.01, 1.0 - norm)
                weights = weights / weights.sum()
                for s in range(len(weights)):
                    _scene_weights[s] = weights[s]
                print(f"[train] Restored scene performance stats for {len(_scene_ema)} scenes")

            print(f"[train] Resumed from {ckpt_path}: step={total_steps}, episode={episode}, "
                  f"best_reward={best_reward:.3f}")

            if args.reset_alpha:
                agent.log_alpha.data.fill_(np.log(args.alpha))
                agent.alpha = args.alpha
                # Reinitialize alpha optimizer to avoid stale state from checkpoint
                agent.alpha_opt = torch.optim.Adam([agent.log_alpha], lr=args.lr)
                print(f"[train] Reset alpha to {args.alpha} (log_alpha={np.log(args.alpha):.4f})")
        else:
            print(f"[train] WARNING: resume checkpoint not found: {ckpt_path}")

    # -------- Training loop --------
    reward_scale = args.reward_scale  # normalize reward magnitude for stable Q learning

    print(f"Run directory: {run_dir}")
    print(f"{'Episode':^8}  {'Steps':^8}  {'Reward':^10}  "
          f"{'r_trk':^9}  {'r_obs':^9}  {'r_manip':^8}  {'r_en':^7}  {'r_coll':^8}  "
          f"{'L_actor':^10}  {'L_dyn':^9}  {'d_obs':^8}  {'suc':^5}")
    print("-" * 140)

    if args.render:
        # ================================================================
        # Single-env training (original loop, supports --render)
        # ================================================================
        while total_steps < args.steps:
            obs = env.reset()
            agent.obs_normalizer.update(obs)
            ep_reward   = 0.0
            ep_l_actor  = 0.0
            ep_l_dyn    = 0.0
            ep_d_obs    = []
            ep_r_track  = []
            ep_r_obs    = []
            ep_r_manip  = []
            ep_r_energy = []
            ep_r_coll   = []
            ep_steps    = 0
            done        = False
            while not done:
                if total_steps < args.start_steps:
                    a_task = np.random.uniform(-0.1, 0.1, 3)
                    a_null = np.random.uniform(-0.3, 0.3, env.n - 3)
                    action = np.concatenate([a_task, a_null])
                else:
                    action = agent.select_action(obs)

                q_prev  = env.q.copy()
                dq_prev = env.dq.copy()
                next_obs, reward, done, info = env.step(action)

                if args.render:
                    env.render()

                logger.log_step(total_steps, episode, ep_steps, reward, info)
                dq_next = env.dq.copy()
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
                ep_r_track.append(info.get("r_track", 0.0))
                ep_r_obs.append(info.get("r_obs", 0.0))
                ep_r_manip.append(info.get("r_manip", 0.0))
                ep_r_energy.append(info.get("r_energy", 0.0))
                ep_r_coll.append(info.get("r_collision", 0.0))
                total_steps += 1
                ep_steps    += 1

                if (total_steps >= args.start_steps and
                        len(buffer) >= args.batch_size and
                        total_steps % args.update_every == 0):
                    for _ in range(args.grad_steps):
                        batch = buffer.sample(args.batch_size)
                        losses, td_errors = agent.update(batch)
                        if args.per:
                            buffer.update_priorities(batch["indices"], td_errors)
                        logger.log_update(losses)
                        ep_l_actor += losses["actor_rl_loss"]
                        ep_l_dyn   += losses["physics_loss"]

            episode += 1
            ep_summary = logger.end_episode(episode, total_steps)
            avg_l_actor = ep_l_actor / max(ep_steps, 1)
            avg_l_dyn   = ep_l_dyn   / max(ep_steps, 1)
            min_d_obs   = min(ep_d_obs) if ep_d_obs else 0.0

            if episode % args.log_every == 0:
                def _avg(lst): return sum(lst)/len(lst) if lst else 0.0
                print(f"{episode:>8d}  {total_steps:>8d}  {ep_reward:>10.3f}  "
                      f"{_avg(ep_r_track):>9.4f}  {_avg(ep_r_obs):>9.4f}  "
                      f"{_avg(ep_r_manip):>8.4f}  {_avg(ep_r_energy):>7.4f}  "
                      f"{_avg(ep_r_coll):>8.4f}  "
                      f"{avg_l_actor:>10.4f}  {avg_l_dyn:>9.4f}  {min_d_obs:>8.3f}")

            ckpt_meta = {
                "step":         total_steps,
                "episode":      episode,
                "best_reward":  logger.best_reward,
                "hyperparams":  hyperparams,
                "csv_path":     logger.csv_path,
            }

            if episode % args.checkpoint_every == 0:
                agent.save(logger.checkpoint_path(f"ep{episode:05d}"), metadata=ckpt_meta)

            if ep_summary["total_reward"] > logger.best_reward:
                logger.best_reward = ep_summary["total_reward"]
                best_reward = logger.best_reward
                ckpt_meta["best_reward"] = logger.best_reward
                agent.save(logger.checkpoint_path("best"), metadata=ckpt_meta)

            if val_set is not None and episode % args.val_every == 0:
                print(f"\n{'='*60}")
                print(f"Validation at episode {episode}")
                print(f"{'='*60}")
                val_results = evaluate_on_validation_set(
                    agent, env, val_set,
                    num_scenes=args.val_scenes, max_steps=env.episode_len
                )
                print(f"Success Rate:      {val_results['success_rate']*100:.1f}%")
                print(f"Avg Reward:        {val_results['avg_reward']:.3f}")
                print(f"Avg Track Error:   {val_results['avg_tracking_error']:.4f}m")
                print(f"Avg Min Distance:  {val_results['avg_min_distance']:.4f}m")
                print(f"Collision Rate:    {val_results['collision_rate']*100:.1f}%")
                print(f"{'='*60}\n")
                logger.log_validation(episode, val_results)

    else:
        # ================================================================
        # Parallel training (pool of n_envs workers)
        # ================================================================
        n_envs = args.n_envs
        obs = pool.reset_all()
        for o in obs:
            agent.obs_normalizer.update(o)

        # Per-env episode tracking
        env_rewards = np.zeros(n_envs)
        env_d_obs   = [[] for _ in range(n_envs)]
        env_w       = [[] for _ in range(n_envs)]
        env_r_track = [[] for _ in range(n_envs)]
        env_r_obs   = [[] for _ in range(n_envs)]
        env_r_manip = [[] for _ in range(n_envs)]
        env_r_energy = [[] for _ in range(n_envs)]
        env_r_collision = [[] for _ in range(n_envs)]
        env_collision_penalty = [[] for _ in range(n_envs)]
        env_ever_collided = [False for _ in range(n_envs)]
        env_steps   = np.zeros(n_envs, dtype=int)
        _log_success_count = 0
        _last_val_ep = -1
        last_losses = {"actor_rl_loss": 0.0, "physics_loss": 0.0}

        if args.algo == "ppo":
            # ============================================================
            # PPO parallel training (on-policy: collect rollout, then update)
            # ============================================================
            while total_steps < args.steps:
                agent.buffer.clear()

                # --- Rollout collection ---
                for step_in_rollout in range(args.rollout_steps):
                    if total_steps >= args.steps:
                        break

                    # Get actions with log_probs and values from current policy
                    actions = np.zeros((n_envs, action_dim), dtype=np.float32)
                    log_probs = np.zeros(n_envs, dtype=np.float32)
                    values = np.zeros(n_envs, dtype=np.float32)
                    for i in range(n_envs):
                        actions[i], log_probs[i], values[i] = agent.act(obs[i])

                    result = pool.step_all(actions)

                    # Push full step (all envs) into rollout buffer
                    agent.buffer.push(
                        obs, actions, result["reward"], result["done"],
                        log_probs, values,
                        q=result["q_before"], dq=result["dq_before"],
                        dq_next=result["dq_after"],
                        J=result["J"], sigma=result["sigma"],
                        dx_nom=result["dx_nom"],
                    )

                    # Per-env tracking
                    for i in range(n_envs):
                        total_steps += 1
                        env_rewards[i] += result["reward"][i]
                        info_i = result["info"][i]
                        env_d_obs[i].append(info_i.get("d_obs", 0.0))
                        env_w[i].append(info_i.get("w", 0.0))
                        env_r_track[i].append(info_i.get("r_track", 0.0))
                        env_r_obs[i].append(info_i.get("r_obs", 0.0))
                        env_r_manip[i].append(info_i.get("r_manip", 0.0))
                        env_r_energy[i].append(info_i.get("r_energy", 0.0))
                        env_r_collision[i].append(info_i.get("r_collision", 0.0))
                        env_collision_penalty[i].append(info_i.get("collision_penalty", 0.0))
                        env_ever_collided[i] = env_ever_collided[i] or info_i.get("collision", False)
                        env_steps[i] += 1
                        agent.obs_normalizer.update(result["obs"][i])

                        if result["done"][i]:
                            episode += 1
                            # Episode-end success: path_complete AND no collision during episode
                            ep_success = info_i.get("success", False)
                            scene_id = result["scene_id"][i]
                            avg_l_actor = last_losses.get("actor_rl_loss", 0.0)
                            avg_l_dyn   = last_losses.get("physics_loss", 0.0)
                            last_critic = last_losses.get("critic_loss", None)
                            last_actor_total = last_losses.get("actor_loss", None)
                            last_alpha  = last_losses.get("alpha", None)
                            min_d_obs   = min(env_d_obs[i]) if env_d_obs[i] else 0.0
                            avg_w       = (sum(env_w[i]) / len(env_w[i])) if env_w[i] else None
                            avg_r_track = (sum(env_r_track[i]) / len(env_r_track[i])) if env_r_track[i] else None
                            avg_r_obs   = (sum(env_r_obs[i]) / len(env_r_obs[i])) if env_r_obs[i] else None
                            avg_r_manip = (sum(env_r_manip[i]) / len(env_r_manip[i])) if env_r_manip[i] else None
                            avg_r_energy = (sum(env_r_energy[i]) / len(env_r_energy[i])) if env_r_energy[i] else None
                            avg_r_collision = (sum(env_r_collision[i]) / len(env_r_collision[i])) if env_r_collision[i] else None
                            avg_collision_penalty = (sum(env_collision_penalty[i]) / len(env_collision_penalty[i])) if env_collision_penalty[i] else None

                            # Per-scene performance tracking
                            if _scene_ema is not None:
                                ema_alpha = 0.3
                                _scene_ema[scene_id] = (ema_alpha * env_rewards[i]
                                                        + (1 - ema_alpha) * _scene_ema[scene_id])
                                _scene_counts[scene_id] += 1
                                if _scene_counts.min() >= 1:
                                    ema = _scene_ema.copy()
                                    ema_min = ema.min()
                                    ema_max = ema.max()
                                    if ema_max > ema_min:
                                        norm = (ema - ema_min) / (ema_max - ema_min + 1e-8)
                                    else:
                                        norm = np.ones_like(ema) * 0.5
                                    weights = np.maximum(0.01, 1.0 - norm)
                                    weights = weights / weights.sum()
                                    for s in range(len(weights)):
                                        _scene_weights[s] = weights[s]

                            # Track success for logging window
                            if ep_success:
                                _log_success_count += 1

                            if episode % args.log_every == 0:
                                print(f"{episode:>8d}  {total_steps:>8d}  {env_rewards[i]:>10.3f}  "
                                      f"{avg_r_track or 0:>9.4f}  {avg_r_obs or 0:>9.4f}  "
                                      f"{avg_r_manip or 0:>8.4f}  {avg_r_energy or 0:>7.4f}  "
                                      f"{avg_r_collision or 0:>8.4f}  "
                                      f"{avg_l_actor:>10.4f}  {avg_l_dyn:>9.4f}  {min_d_obs:>8.3f}  "
                                      f"s={scene_id}  suc={_log_success_count}")
                                _log_success_count = 0

                            logger.log_episode_summary(
                                step=total_steps, episode=episode,
                                total_reward=env_rewards[i], min_d_obs=min_d_obs,
                                avg_actor_loss=avg_l_actor, avg_physics_loss=avg_l_dyn,
                                ep_step=env_steps[i],
                                alpha=last_alpha,
                                avg_critic_loss=last_critic,
                                avg_actor_total_loss=last_actor_total,
                                avg_w=avg_w,
                                avg_r_track=avg_r_track,
                                avg_r_obs=avg_r_obs,
                                avg_r_manip=avg_r_manip,
                                avg_r_energy=avg_r_energy,
                                avg_r_collision=avg_r_collision,
                                avg_collision_penalty=avg_collision_penalty,
                                success=int(ep_success),
                                ever_collided=int(env_ever_collided[i]),
                            )

                            ckpt_meta = {
                                "step":        total_steps,
                                "episode":     episode,
                                "best_reward": best_reward,
                                "hyperparams": hyperparams,
                                "csv_path":    logger.csv_path,
                                "scene_ema":   _scene_ema.tolist() if _scene_ema is not None else None,
                                "scene_counts": _scene_counts.tolist() if _scene_counts is not None else None,
                            }

                            if episode % args.checkpoint_every == 0:
                                agent.save(
                                    logger.checkpoint_path(f"ep{episode:05d}"),
                                    metadata=ckpt_meta
                                )

                            if env_rewards[i] > best_reward:
                                best_reward = env_rewards[i]
                                ckpt_meta["best_reward"] = best_reward
                                agent.save(
                                    logger.checkpoint_path("best"),
                                    metadata=ckpt_meta
                                )

                            # Validation evaluation (only once per val_every boundary)
                            if val_set is not None and episode > 0 and episode % args.val_every == 0 and episode != _last_val_ep:
                                _last_val_ep = episode
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

                            # Reset per-env tracking
                            env_rewards[i] = 0.0
                            env_d_obs[i]   = []
                            env_w[i]       = []
                            env_r_track[i] = []
                            env_r_obs[i]   = []
                            env_r_manip[i] = []
                            env_r_energy[i] = []
                            env_r_collision[i] = []
                            env_collision_penalty[i] = []
                            env_ever_collided[i] = False
                            env_steps[i]   = 0
                    obs = result["obs"]

                # --- GAE computation ---
                if len(agent.buffer) > 0:
                    last_values = agent.get_value(obs)
                    agent.buffer.compute_advantages(last_values)

                    # --- PPO update ---
                    losses = agent.update()
                    last_losses = losses

                # --- Validation (moved inside done block below) ---

        else:
            # ============================================================
            # SAC parallel training (off-policy: store + update per step)
            # ============================================================
            while total_steps < args.steps:
                # Collect actions for all envs in parallel
                actions = np.zeros((n_envs, action_dim), dtype=np.float32)
                for i in range(n_envs):
                    if total_steps < args.start_steps:
                        a_task = np.random.uniform(-0.1, 0.1, 3)
                        a_null = np.random.uniform(-0.3, 0.3, ref_env.n - 3)
                        actions[i] = np.concatenate([a_task, a_null])
                    else:
                        actions[i] = agent.select_action(obs[i])

                # Step all envs in parallel
                result = pool.step_all(actions)

                # Store in buffer and track per-env metrics (episode completion
                # checked inline so each done env sees its own total_steps)
                for i in range(n_envs):
                    buffer.push(
                        obs[i], actions[i], result["reward"][i] / reward_scale,
                        result["obs"][i], result["done"][i],
                        q=result["q_before"][i], dq=result["dq_before"][i],
                        dq_next=result["dq_after"][i],
                        J=result["J"][i], sigma=result["sigma"][i],
                        dx_nom=result["dx_nom"][i],
                    )
                    total_steps += 1
                    env_rewards[i] += result["reward"][i]
                    env_d_obs[i].append(result["info"][i].get("d_obs", 0.0))
                    env_w[i].append(result["info"][i].get("w", 0.0))
                    env_r_track[i].append(result["info"][i].get("r_track", 0.0))
                    env_r_obs[i].append(result["info"][i].get("r_obs", 0.0))
                    env_r_manip[i].append(result["info"][i].get("r_manip", 0.0))
                    env_r_energy[i].append(result["info"][i].get("r_energy", 0.0))
                    env_r_collision[i].append(result["info"][i].get("r_collision", 0.0))
                    env_collision_penalty[i].append(result["info"][i].get("collision_penalty", 0.0))
                    env_ever_collided[i] = env_ever_collided[i] or result["info"][i].get("collision", False)
                    env_steps[i] += 1
                    agent.obs_normalizer.update(result["obs"][i])

                    if result["done"][i]:
                        episode += 1
                        # Episode-end success: path_complete AND no collision during episode
                        ep_success = result["info"][i].get("success", False)
                        scene_id = result["scene_id"][i]
                        avg_l_actor = last_losses.get("actor_rl_loss", 0.0)
                        avg_l_dyn   = last_losses.get("physics_loss", 0.0)
                        last_critic = last_losses.get("critic_loss", None)
                        last_actor_total = last_losses.get("actor_loss", None)
                        last_alpha  = last_losses.get("alpha", None)
                        min_d_obs   = min(env_d_obs[i]) if env_d_obs[i] else 0.0
                        avg_w       = (sum(env_w[i]) / len(env_w[i])) if env_w[i] else None
                        avg_r_track = (sum(env_r_track[i]) / len(env_r_track[i])) if env_r_track[i] else None
                        avg_r_obs   = (sum(env_r_obs[i]) / len(env_r_obs[i])) if env_r_obs[i] else None
                        avg_r_manip = (sum(env_r_manip[i]) / len(env_r_manip[i])) if env_r_manip[i] else None
                        avg_r_energy = (sum(env_r_energy[i]) / len(env_r_energy[i])) if env_r_energy[i] else None
                        avg_r_collision = (sum(env_r_collision[i]) / len(env_r_collision[i])) if env_r_collision[i] else None
                        avg_collision_penalty = (sum(env_collision_penalty[i]) / len(env_collision_penalty[i])) if env_collision_penalty[i] else None

                        # Per-scene performance tracking
                        if _scene_ema is not None:
                            ema_alpha = 0.3
                            _scene_ema[scene_id] = (ema_alpha * env_rewards[i]
                                                    + (1 - ema_alpha) * _scene_ema[scene_id])
                            _scene_counts[scene_id] += 1
                            if _scene_counts.min() >= 1:
                                ema = _scene_ema.copy()
                                ema_min = ema.min()
                                ema_max = ema.max()
                                if ema_max > ema_min:
                                    norm = (ema - ema_min) / (ema_max - ema_min + 1e-8)
                                else:
                                    norm = np.ones_like(ema) * 0.5
                                weights = np.maximum(0.01, 1.0 - norm)
                                weights = weights / weights.sum()
                                for s in range(len(weights)):
                                    _scene_weights[s] = weights[s]

                        # Track success for logging window (every episode)
                        if ep_success:
                            _log_success_count += 1

                        if episode % args.log_every == 0:
                            print(f"{episode:>8d}  {total_steps:>8d}  {env_rewards[i]:>10.3f}  "
                                  f"{avg_r_track or 0:>9.4f}  {avg_r_obs or 0:>9.4f}  "
                                  f"{avg_r_manip or 0:>8.4f}  {avg_r_energy or 0:>7.4f}  "
                                  f"{avg_r_collision or 0:>8.4f}  "
                                  f"{avg_l_actor:>10.4f}  {avg_l_dyn:>9.4f}  {min_d_obs:>8.3f}  "
                                  f"s={scene_id}  suc={_log_success_count}")
                            _log_success_count = 0

                        logger.log_episode_summary(
                            step=total_steps, episode=episode,
                            total_reward=env_rewards[i], min_d_obs=min_d_obs,
                            avg_actor_loss=avg_l_actor, avg_physics_loss=avg_l_dyn,
                            ep_step=env_steps[i],
                            alpha=last_alpha,
                            avg_critic_loss=last_critic,
                            avg_actor_total_loss=last_actor_total,
                            avg_w=avg_w,
                            avg_r_track=avg_r_track,
                            avg_r_obs=avg_r_obs,
                            avg_r_manip=avg_r_manip,
                            avg_r_energy=avg_r_energy,
                            avg_r_collision=avg_r_collision,
                            avg_collision_penalty=avg_collision_penalty,
                            success=int(ep_success),
                            ever_collided=int(env_ever_collided[i]),
                        )

                        ckpt_meta = {
                            "step":        total_steps,
                            "episode":     episode,
                            "best_reward": best_reward,
                            "hyperparams": hyperparams,
                            "csv_path":    logger.csv_path,
                            "scene_ema":   _scene_ema.tolist() if _scene_ema is not None else None,
                            "scene_counts": _scene_counts.tolist() if _scene_counts is not None else None,
                        }

                        if episode % args.checkpoint_every == 0:
                            agent.save(
                                logger.checkpoint_path(f"ep{episode:05d}"),
                                metadata=ckpt_meta
                            )

                        if env_rewards[i] > best_reward:
                            best_reward = env_rewards[i]
                            ckpt_meta["best_reward"] = best_reward
                            agent.save(
                                logger.checkpoint_path("best"),
                                metadata=ckpt_meta
                            )

                        # Validation evaluation (only once per val_every boundary)
                        if val_set is not None and episode > 0 and episode % args.val_every == 0 and episode != _last_val_ep:
                            _last_val_ep = episode
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

                        # Reset per-env tracking
                        env_rewards[i] = 0.0
                        env_d_obs[i]   = []
                        env_w[i]       = []
                        env_r_track[i] = []
                        env_r_obs[i]   = []
                        env_r_manip[i] = []
                        env_r_energy[i] = []
                        env_r_collision[i] = []
                        env_collision_penalty[i] = []
                        env_ever_collided[i] = False
                        env_steps[i]   = 0
                obs = result["obs"]  # auto-reset obs for done envs

                # Training update (one batch update per iteration)
                if total_steps >= args.start_steps and len(buffer) >= args.batch_size:
                    for _ in range(args.grad_steps):
                        batch = buffer.sample(args.batch_size)
                        losses, td_errors = agent.update(batch)
                        if args.per:
                            buffer.update_priorities(batch["indices"], td_errors)
                        last_losses = losses

    # -------- Cleanup --------
    logger.close()
    if args.render and hasattr(env, '_viewer'):
        env._viewer.close()
    if not args.render:
        pool.close()
    print(f"\nTraining done. Best reward: {best_reward:.3f}")
    print(f"Run directory: {run_dir}")
    print(f"CSV log: {logger.csv_path}")


if __name__ == "__main__":
    main()
