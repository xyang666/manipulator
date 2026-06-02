"""
train_mjx.py
------------
MJX-accelerated batched training for the manipulator RL environment.

Uses MJXManipulatorEnv (JAX GPU) for environment stepping, combined
with the existing SAC agent (PyTorch) and replay buffer (numpy).

Usage:
    cd /root/manipulator
    code/.venv/bin/python code/train_mjx.py --n_envs 64 --scene_json <path>

Key difference from train.py:
  - GPU-parallel env stepping (JAX/vmap)
  - Same SAC agent, same replay buffer
  - No Pinocchio dependency for env stepping (JAX FK instead)
"""

import sys, os, json, time, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import torch
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from env.mjx_env import MJXManipulatorEnv
from env.dynamics import ManipulatorDynamics
from agent.sac_agent import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.gpu_buffer import GpuReplayBuffer
from utils.logger import (TrainingLogger, REWARD_COMPONENTS, REWARD_HEADER,
                           REWARD_FORMAT, reward_accumulators,
                           accumulate_rewards, avg_rewards, reward_print_values)
from utils.validation import ValidationSet


def parse_args():
    p = argparse.ArgumentParser()

    # Training
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--start_steps", type=int, default=50_000)
    p.add_argument("--grad_steps", type=int, default=4)
    p.add_argument("--buffer_size", type=int, default=500_000)
    p.add_argument("--update_every", type=int, default=1,
                   help="Update SAC every N outer loops")
    p.add_argument("--episode_len", type=int, default=400)
    p.add_argument("--n_envs", type=int, default=128)
    p.add_argument("--max_obs", type=int, default=10)

    # SAC
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--task_scale", type=float, default=0.2)
    p.add_argument("--nullspace_scale", type=float, default=5.0)
    p.add_argument("--lambda_dyn", type=float, default=1.0)
    p.add_argument("--hidden_dims", type=str, default="512,512,512")
    p.add_argument("--n_critics", type=int, default=5)
    p.add_argument("--cost_limit", type=float, default=0.05)
    p.add_argument("--cost_scale", type=float, default=1.0)
    p.add_argument("--critic_warmup", type=int, default=600,
                   help="Update count before actor training starts. Scaled for batch envs.")
    p.add_argument("--target_entropy", type=float, default=None)
    p.add_argument("--per", action="store_true", default=True)

    # Reward / env params (v3 defaults)
    p.add_argument("--w_track", type=float, default=1.0)
    p.add_argument("--w_obs", type=float, default=50.0)
    p.add_argument("--w_manip", type=float, default=0.0)
    p.add_argument("--w_energy", type=float, default=0.0)
    p.add_argument("--w_action", type=float, default=0.0)
    p.add_argument("--w_collision", type=float, default=100.0)
    p.add_argument("--d_safe", type=float, default=0.06)
    p.add_argument("--lr_lag", type=float, default=0.01)
    p.add_argument("--lag_target", type=float, default=0.05)
    p.add_argument("--path_deadzone", type=float, default=0.10)
    p.add_argument("--success_bonus", type=float, default=500.0)
    p.add_argument("--reward_min", type=float, default=-100.0)
    p.add_argument("--reward_scale", type=float, default=100.0)

    # Observation
    p.add_argument("--obs_scene_embed", type=int, default=5)
    p.add_argument("--obs_waypoint_steps", type=str, default="10,20,50")

    # Validation
    p.add_argument("--val_json", type=str, default=None,
                   help="Validation scenes JSON (optional)")
    p.add_argument("--val_every", type=int, default=1000,
                   help="Validate every N episodes")
    p.add_argument("--val_scenes", type=int, default=10,
                   help="Number of validation scenes to evaluate")

    # Paths
    p.add_argument("--scene_json", type=str, default="results/challenge_stage3.json")
    p.add_argument("--xml", type=str, default="models/panda_scene.xml")
    p.add_argument("--urdf", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_path", type=str, default="checkpoints/mjx/")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--checkpoint_every", type=int, default=1000)

    return p.parse_args()


def main():
    args = parse_args()
    if args.hidden_dims:
        args.hidden_dims = [int(x) for x in args.hidden_dims.replace(' ', '').split(',')]

    # Resolve paths
    _root = "/root/manipulator"
    xml_path = args.xml if os.path.isabs(args.xml) else os.path.join(_root, args.xml)
    scene_json = args.scene_json if os.path.isabs(args.scene_json) else os.path.join(_root, args.scene_json)
    urdf_path = args.urdf if args.urdf else os.path.join(_root, "panda_description/urdf/panda.urdf")

    # Setup save dir
    run_name = args.run_name or f"mjx_{time.strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(_root, args.save_path, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # -------- Parse waypoints --------
    obs_waypoint_steps = [int(s.strip()) for s in args.obs_waypoint_steps.split(",")]

    # -------- Reward params --------
    reward_params = {
        'w_track': args.w_track,
        'w_obs': args.w_obs,
        'w_manip': args.w_manip,
        'w_energy': args.w_energy,
        'w_action': args.w_action,
        'w_collision': args.w_collision,
        'success_bonus': args.success_bonus,
        'reward_min': args.reward_min,
        'reward_scale': args.reward_scale,
    }

    # -------- Create MJX env --------
    print(f"[train_mjx] Creating MJX env with {args.n_envs} parallel instances...")
    env = MJXManipulatorEnv(
        n_envs=args.n_envs,
        xml_path=xml_path,
        scene_json=scene_json,
        dt=0.02,
        episode_len=args.episode_len,
        d_safe=args.d_safe,
        path_deadzone=args.path_deadzone,
        lr_lag=args.lr_lag,
        lag_target=args.lag_target,
        max_obs=args.max_obs,
        reward_params=reward_params,
        n_obs_embed=args.obs_scene_embed,
        obs_waypoint_steps=obs_waypoint_steps,
    )

    # Reset env (loads scenes, triggers JIT warmup)
    obs = env.reset()
    obs_dim = obs.shape[1]
    action_dim = env.n_arm  # 7
    print(f"[train_mjx] obs_dim={obs_dim}, action_dim={action_dim}, "
          f"n_caps={env.n_caps}, n_self={env.n_self}")

    # -------- Dynamics for physics loss --------
    dyn = ManipulatorDynamics(urdf_path)

    # -------- SAC agent --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(
        state_dim=obs_dim,
        action_dim=action_dim,
        dynamics=dyn,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        target_entropy=args.target_entropy,
        lambda_dyn=args.lambda_dyn,
        task_scale=args.task_scale,
        nullspace_scale=args.nullspace_scale,
        hidden_dims=args.hidden_dims,
        n_critics=args.n_critics,
        cost_limit=args.cost_limit,
        cost_scale=args.cost_scale,
        critic_warmup=args.critic_warmup,
        device=device,
    )

    # -------- Replay buffer (GPU-accelerated) --------
    buffer = GpuReplayBuffer(
        capacity=args.buffer_size,
        state_dim=obs_dim,
        action_dim=action_dim,
        joints=7,
        device=device,
    )

    # -------- Validation set --------
    val_set = None
    if args.val_json is not None:
        val_path = args.val_json if os.path.isabs(args.val_json) else os.path.join(_root, args.val_json)
        if os.path.exists(val_path):
            val_set = ValidationSet(val_path)
            print(f"[train_mjx] Validation set: {len(val_set.scenes)} scenes from {val_path}")
        else:
            print(f"[train_mjx] WARNING: validation file not found: {val_path}")

    # -------- Logger --------
    logger = TrainingLogger(save_dir, vars(args))

    # -------- Training loop --------
    total_steps = 0
    episode = 0
    best_reward = -np.inf
    next_obs = obs  # from reset above

    # Per-env accumulators (reset on episode end)
    env_rew = np.zeros(args.n_envs)
    env_len = np.zeros(args.n_envs, dtype=np.int32)
    env_r_acc = [reward_accumulators() for _ in range(args.n_envs)]
    env_d_obs = [[] for _ in range(args.n_envs)]
    env_w = [[] for _ in range(args.n_envs)]
    env_success = [False] * args.n_envs
    env_coll = [False] * args.n_envs
    env_scene = np.zeros(args.n_envs, dtype=np.int32)

    _last_val_episode = 0
    _last_ckpt_episode = 0
    losses = {}  # hold last update losses (empty during start_steps)

    print(f"[train_mjx] Starting training ({args.steps} total steps)...")
    print(f"Run directory: {save_dir}")
    print(f"{'Episode':^8}  {'Steps':^8}  {'Reward':^10}  {REWARD_HEADER}  "
          f"{'L_critic':^8}  {'L_scritic':^8}  "
          f"{'L_actor':^10}  {'L_dyn':^9}  {'d_obs':^8}  {'scene':>4}  {'suc':^5}")
    print("-" * 175)
    t_start = time.time()

    while total_steps < args.steps:
        # Collect batch of actions from agent
        if total_steps < args.start_steps:
            actions = np.random.uniform(-1, 1, (args.n_envs, action_dim)).astype(np.float32)
        else:
            actions = agent.select_action_batch(next_obs, deterministic=False)

        # Step env
        next_obs, rewards, dones, infos = env.step(actions)

        # Store transitions (batched GPU push, reward already scaled by env)
        buffer.push_batch(
            obs, actions, rewards, next_obs, dones, infos,
            keys=['q_before', 'dq_before', 'J_before', 'sigma', 'dx_nom_before', 'cost'],
        )
        # Episode stats (reward already scaled)
        scaled_rewards = rewards
        for i in range(args.n_envs):
            info = infos[i]
            env_rew[i] += scaled_rewards[i]
            env_len[i] += 1
            accumulate_rewards(info, env_r_acc[i])
            env_d_obs[i].append(info.get('d_obs', 0.0))
            env_w[i].append(info.get('w', 0.0) if 'w' in info else 0.0)
            env_success[i] = info.get('success', False)
            env_coll[i] = info.get('collision', False)
            env_scene[i] = env.scene_manager.env_scene_ids[i] if env.scene_manager else 0

        obs = next_obs
        total_steps += args.n_envs

        # Training update
        if (total_steps >= args.start_steps and len(buffer) >= args.batch_size
                and total_steps % args.update_every == 0):
            for _ in range(args.grad_steps):
                batch = buffer.sample(args.batch_size)
                losses, _ = agent.update(batch)
                logger.log_update(losses)

        # Episode bookkeeping
        for i in range(args.n_envs):
            if dones[i]:
                episode += 1
                ep_return = env_rew[i]
                ep_len = env_len[i]
                avg_d_obs = np.mean(env_d_obs[i]) if env_d_obs[i] else 0.0
                avg_w = np.mean(env_w[i]) if env_w[i] else 0.0
                avg_r = avg_rewards(env_r_acc[i])
                succ = env_success[i]
                coll = env_coll[i]

                # Log to CSV with loss data
                logger.log_episode_summary(
                    total_steps, episode, ep_return, avg_d_obs,
                    avg_actor_loss=losses.get('actor_rl_loss', 0.0),
                    avg_physics_loss=losses.get('physics_loss', 0.0),
                    ep_step=ep_len, success=succ,
                    ever_collided=coll, avg_w=avg_w,
                    avg_critic_loss=losses.get('critic_loss', 0.0),
                    avg_safety_critic_loss=losses.get('safety_critic_loss', 0.0),
                    avg_actor_total_loss=losses.get('actor_loss', 0.0),
                    alpha=losses.get('alpha', 0.0) if 'alpha' in losses else None,
                    lag=losses.get('lag', 0.0) if 'lag' in losses else None,
                    **{k: avg_r.get(k, 0.0) for k in [c for _, c, _, _ in REWARD_COMPONENTS]},
                )

                if episode % args.log_every == 0:
                    step_s = total_steps / (time.time() - t_start)
                    status = "OK  " if succ else ("COLL" if coll else "FAIL")
                    rp = reward_print_values(avg_r)
                    print(f"{episode:>8d}  {ep_len:>8d}  {ep_return:>10.2f}  "
                          f"{REWARD_FORMAT.format(**rp)}  "
                          f"{losses.get('critic_loss', 0):>8.4f}  {losses.get('safety_critic_loss', 0):>8.4f}  "
                          f"{losses.get('actor_rl_loss', 0):>10.4f}  {losses.get('physics_loss', 0):>9.4f}  "
                          f"{avg_d_obs:>8.3f}  {env_scene[i]:>4d}  {'OK' if succ else 'COLL' if coll else 'FAIL':>5s}")

                if ep_return > best_reward:
                    best_reward = ep_return

                env_rew[i] = 0.0
                env_len[i] = 0
                env_r_acc[i] = reward_accumulators()
                env_d_obs[i] = []
                env_w[i] = []
                env_success[i] = False
                env_coll[i] = False

        # Periodic checkpoint
        if episode > 0 and episode - _last_ckpt_episode >= args.checkpoint_every:
            ckpt_path = logger.checkpoint_path(f"ep{episode:05d}")
            agent.save(ckpt_path, metadata={
                'step': total_steps, 'episode': episode,
                'best_reward': best_reward, 'args': vars(args),
            })
            print(f"[train_mjx] Saved checkpoint: {ckpt_path}")
            _last_ckpt_episode = episode

        # Periodic validation (uses MJX env + ValidationSet)
        if val_set is not None and episode - _last_val_episode >= args.val_every:
            # Save env state so validation doesn't corrupt training
            saved_obs_c, saved_obs_r = env.obs_centers, env.obs_radii
            saved_state = env.state
            val_results = _run_validation_mjx(agent, env, val_set, args)
            # Restore training env state
            env.obs_centers, env.obs_radii = saved_obs_c, saved_obs_r
            env.state = saved_state
            _last_val_episode = episode
            logger.log_validation(episode, val_results)
            print(f"[val] ep={episode} success={val_results['success_rate']*100:.1f}% "
                  f"coll={val_results['collision_rate']*100:.1f}% "
                  f"rew={val_results['avg_reward']:.1f} "
                  f"track_err={val_results['avg_tracking_error']:.4f} "
                  f"min_dist={val_results['avg_min_distance']:.4f}")

    # Final save
    final_path = logger.checkpoint_path("final")
    agent.save(final_path)
    logger.close()
    print(f"[train_mjx] Done. Total steps: {total_steps}, episodes: {episode}")
    print(f"[train_mjx] Final checkpoint: {final_path}")


def _run_validation_mjx(agent, mjx_env, val_set, args):
    """Validate N scenes on N parallel envs. Saves/restores training env state."""
    from env.mjx_env import init_state
    n_val = min(args.val_scenes, len(val_set.scenes))
    n_env = mjx_env.n_envs
    assert n_env >= n_val, f"Need {n_val} envs but have {n_env}"
    agent.actor.eval()

    home_arm = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
    home_q = np.concatenate([home_arm, np.zeros(mjx_env.nv - mjx_env.n_arm)])
    FAR = 100.0

    # Per-env obstacle/start/goal data (use FAR padding not zeros!)
    obs_c = np.full((n_env, mjx_env.max_obs, 3), FAR)
    obs_r = np.full((n_env, mjx_env.max_obs), 1.0)
    starts = np.zeros((n_env, 3))
    goals = np.zeros((n_env, 3))
    for ei in range(n_val):
        sc = val_set.scenes[ei]
        starts[ei] = sc['start']; goals[ei] = sc['goal']
        for j, ob in enumerate(sc.get('obstacles', [])):
            if j >= mjx_env.max_obs: break
            obs_c[ei, j] = ob[:3]; obs_r[ei, j] = ob[3] if len(ob) > 3 else 0.1

    # Save training env state, override with val data
    saved_oc = mjx_env.obs_centers; saved_or = mjx_env.obs_radii
    saved_s = mjx_env.state
    mjx_env.obs_centers = jnp.array(obs_c)
    mjx_env.obs_radii = jnp.array(obs_r)
    state = init_state(n_env, mjx_env.nv)
    state['q'] = jnp.tile(jnp.array(home_q), (n_env, 1))
    state['x_start'] = jnp.array(starts)
    state['x_goal'] = jnp.array(goals)
    state['x_d'] = jnp.array(starts)
    mjx_env.state = state

    # Run batched validation
    ep_ok = np.zeros(n_val, dtype=bool)
    ep_coll = np.zeros(n_val, dtype=bool)
    ep_rew = np.zeros(n_val)
    ep_te = [[] for _ in range(n_val)]
    ep_d = [[] for _ in range(n_val)]

    for _ in range(args.episode_len):
        act = agent.select_action_batch(mjx_env._compute_obs(mjx_env.state), True)
        _, r, _, info = mjx_env.step(act)
        for i in range(n_val):
            ep_rew[i] += r[i]
            ep_te[i].append(info[i].get('tracking_error', 0.0))
            ep_d[i].append(info[i].get('d_obs', 0.0))
            if info[i].get('path_param', 0.0) >= 0.99: ep_ok[i] = True
            if info[i].get('collision', False): ep_coll[i] = True

    # Restore training env
    mjx_env.obs_centers = saved_oc; mjx_env.obs_radii = saved_or
    mjx_env.state = saved_s
    agent.actor.train()

    ok = int(ep_ok.sum()); coll = int(ep_coll.sum())
    rw = float(ep_rew.mean())
    te = float(np.mean([np.mean(t) for t in ep_te if t])) if ep_te else 0.0
    md = float(np.mean([np.min(d) for d in ep_d if d])) if ep_d else 0.0
    return {"success_rate": ok/n_val, "avg_reward": rw,
            "avg_tracking_error": te, "avg_min_distance": md,
            "collision_rate": coll/n_val, "scene_results": []}


if __name__ == '__main__':
    main()
