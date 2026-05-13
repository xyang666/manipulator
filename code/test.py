"""
test.py
-------
Unified test script for the manipulator. Supports multiple control methods:
  - kp :  Baseline task-space PD (proportional-derivative) tracking
  - mpc:  Model Predictive Control with obstacle avoidance
  - rl :  Trained SAC agent (requires a checkpoint)

Usage:
    cd code/
    python test.py --method mpc --n_obstacles 3 --render
    python test.py --method kp  --steps 500
    python test.py --method sac --checkpoint ../checkpoints/sac_pirl.pt --render
"""

import json
import numpy as np
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from env.manipulator_env import ManipulatorEnv


# ---------------------------------------------------------------------------
# Path defaults
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_VENV_DATA = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                          "/share/example-robot-data/robots/panda_description")
_DEFAULT_URDF = os.path.join(_VENV_DATA, "urdf/panda.urdf")
_DEFAULT_XML  = os.path.join(_ROOT, "models/panda_scene.xml")
_DEFAULT_CKPT = os.path.join(_ROOT, "checkpoints/sac_pirl.pt")
_DEFAULT_SCENE_JSON = os.path.join(_ROOT, "results/trajectories.json")


# ---------------------------------------------------------------------------
# Scene loading from JSON
# ---------------------------------------------------------------------------

def load_scene_from_json(json_path: str, scene_id: int) -> dict:
    """
    Load a scene from a trajectory JSON file.

    JSON format (from TrajectoryGenerator):
        [
          {
            "scene_id": 0,
            "start": [x, y, z],
            "goal": [x, y, z],
            "obstacles": [[x, y, z, r], ...],
            "manipulability_mean": 0.123
          },
          ...
        ]
    """
    with open(json_path) as f:
        scenes = json.load(f)
    for s in scenes:
        if s["scene_id"] == scene_id:
            return s
    available = [s["scene_id"] for s in scenes]
    raise ValueError(f"Scene {scene_id} not found in {json_path}. "
                     f"Available scene IDs: {available}")


def apply_scene(env, scene: dict) -> bool:
    """
    Override env state with scene data (start, goal, obstacles).

    Sets up the environment for tracking from start to goal while
    avoiding the obstacles defined in the scene.

    Returns True on success, False if IK fails.
    """
    start_pos = np.array(scene["start"])
    goal_pos = np.array(scene["goal"])
    obstacles = scene["obstacles"]

    obs_centers = [np.array(o[:3]) for o in obstacles]
    obs_radii = [o[3] for o in obstacles]

    # Set obstacles
    env.sdf.set_static_obstacles(obs_centers, obs_radii)
    env._sync_obstacles_to_mujoco()

    # Use stored IK config if available (guarantees manipulability from generation),
    # otherwise compute IK at runtime
    if "start_q" in scene:
        q_start = np.array(scene["start_q"])
    else:
        q_start = env.kin.inverse_kinematics(start_pos)
        if q_start is None:
            print("[ERROR] IK failed for scene start position "
                  f"[{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
            return False

    # Override environment state
    env.q = q_start.copy()
    env.dq = np.zeros(env.n)
    env.x_start = start_pos.copy()
    env.x_goal = goal_pos.copy()
    env.x_d = start_pos.copy()
    env.dx_d = np.zeros(3)
    env.step_count = 0
    env.path_param = 0.0
    env.ee_trajectory.clear()
    env._last_sigma = 0.0

    # Set desired velocity toward goal
    direction = goal_pos - start_pos
    dist = np.linalg.norm(direction)
    if dist > 1e-6:
        env.dx_d[:3] = (direction / dist) * 0.1  # 0.1 m/s

    # Sync MuJoCo state
    if env.mj_data is not None:
        try:
            import mujoco
        except ImportError:
            return True
        env.mj_data.qpos[:env.n] = q_start
        env.mj_data.qvel[:env.n] = 0.0
        env.mj_data.qpos[env.n:env.n + 2] = 0.0
        env.mj_data.qvel[env.n:env.n + 2] = 0.0
        mujoco.mj_forward(env.mj_model, env.mj_data)

    return True


# ---------------------------------------------------------------------------
# Paper experiment scenario setups
# ---------------------------------------------------------------------------

def _scene1_setup(env):
    """
    Scene 1: 稀疏静态障碍（验证基线跟踪性能）

    Paper: Section 4.1.3, Scenario 1
      - Figure-8 lemniscate trajectory in yz-plane
      - 3 static spherical obstacles placed near but not intersecting the path
      - Purpose: verify EE tracking accuracy and that gate operator
        does NOT activate unnecessarily in safe regions (d_obs > d_critical)
    """
    # Figure-8: y = 0.15·sin(t), z = 0.4 + 0.1·sin(2t), t ∈ [0, 2π]
    x_center = 0.4  # constant x (trajectory in yz-plane)

    def pos_func(t):
        return np.array([x_center,
                         0.15 * np.sin(t),
                         0.4 + 0.1 * np.sin(2.0 * t)])

    def vel_func(t):
        return np.array([0.0,
                         0.15 * np.cos(t),
                         0.2 * np.cos(2.0 * t)])

    env.set_parametric_trajectory(pos_func, vel_func)

    # Three static obstacles, placed well clear of the trajectory path
    # (d_obs > 0.10 m throughout).  The goal in Scene 1 is to verify that
    # the gate operator does NOT activate in safe regions.
    obstacles = [
        ([0.4,  0.0,   0.65], 0.08),  # far above the figure-8 (z-peak≈0.50)
        ([0.4, -0.35,  0.40], 0.08),  # far left of y-extent (±0.15)
        ([0.4,  0.35,  0.30], 0.08),  # far right-lower
    ]
    centers = [np.array(o[0]) for o in obstacles]
    radii   = [o[1] for o in obstacles]
    env.sdf.set_static_obstacles(centers, radii)
    env._sync_obstacles_to_mujoco()

    print("[Scene1] Figure-8 lemniscate  +  3 static obstacles (d_obs > d_critical)")
    return env


def _scene2_setup(env):
    """
    Scene 2: 密集障碍窄通道（考验避障与松弛机制）

    Paper: Section 4.1.3, Scenario 2
      - Linear trajectory through narrow corridor
      - 14 obstacles in staggered formation, 0.14 m gap
    """
    # Linear trajectory along y-axis
    x_start = np.array([0.4, -0.20, 0.4])
    x_goal  = np.array([0.4,  0.20, 0.4])
    env.x_start = x_start
    env.x_goal  = x_goal
    env.x_d     = x_start.copy()
    direction = x_goal - x_start
    env.dx_d[:3] = (direction / np.linalg.norm(direction)) * 0.08

    # 14 obstacles: two staggered rows forming a 0.14 m gap
    obs_radius = 0.06
    spacing = 0.10
    centers = []
    for i in range(7):
        offset = spacing * i - 0.30
        centers.append(np.array([0.4 - 0.07, offset, 0.4]))  # left row
        centers.append(np.array([0.4 + 0.07, offset, 0.4]))  # right row
    radii = [obs_radius] * len(centers)
    env.sdf.set_static_obstacles(centers, radii)
    env._sync_obstacles_to_mujoco()

    print(f"[Scene2] Linear corridor  +  {len(centers)} obstacles (gap=0.14 m)")
    return env


def _scene3_setup(env):
    """
    Scene 3: 动态障碍环境（检验在线自适应能力）

    Paper: Section 4.1.3, Scenario 3
      - Linear trajectory
      - 3 moving obstacles crossing the path
    """
    # Linear trajectory along x-axis
    x_start = np.array([0.5,  0.0, 0.5])
    x_goal  = np.array([0.2,  0.0, 0.5])
    env.x_start = x_start
    env.x_goal  = x_goal
    env.x_d     = x_start.copy()
    direction = x_goal - x_start
    env.dx_d[:3] = (direction / np.linalg.norm(direction)) * 0.10

    # Static initial obstacle placement (dynamics handled by MuJoCo mocap)
    obs_radius = 0.08
    centers = [
        np.array([0.45, -0.10, 0.5]),
        np.array([0.35,  0.10, 0.5]),
        np.array([0.25, -0.10, 0.5]),
    ]
    radii = [obs_radius] * len(centers)
    env.sdf.set_static_obstacles(centers, radii)
    env._sync_obstacles_to_mujoco()

    # Store motion parameters so the stepping loop can update positions
    env._scene3_motion = {
        "amplitude": 0.15,
        "speed": 0.1,
        "centers_init": [c.copy() for c in centers],
    }
    print(f"[Scene3] Linear trajectory  +  3 moving obstacles (v=0.1 m/s)")
    return env

def run_kp(env, args):
    """
    Baseline KP controller: zero action → pure PD tracking in task space
    with adaptive gain and integral term (built into ManipulatorEnv).
    """
    action = np.zeros(env.act_dim)
    return _run_env(env, args, "KP")


def run_mpc(env, args):
    """
    MPC with obstacle avoidance: uses task-space QP with repulsive potential.
    Obstacles are passed from the environment's SDF to the MPC controller.
    """
    if env.controller != "mpc" or env.mpc is None:
        print("[ERROR] MPC not available. Install cvxpy or use controller='mpc'.")
        sys.exit(1)
    action = np.zeros(env.act_dim)
    return _run_env(env, args, "MPC")


def run_rl(env, args, agent):
    """
    Trained SAC agent: actions come from the policy network.
    """
    from agent.sac_agent import SACAgent
    from env.dynamics import ManipulatorDynamics

    print(f"[SAC] Loading agent from {args.checkpoint}")

    # Infer hidden_dims from checkpoint's config.json if available
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, "config.json")
    hidden_dims = (256, 256)  # default fallback
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        cli_hidden = cfg.get("cli_args", {}).get("hidden_dims", None)
        if cli_hidden is not None:
            hidden_dims = tuple(cli_hidden) if isinstance(cli_hidden, list) else (cli_hidden,)
            print(f"[SAC] Using hidden_dims={hidden_dims} from config.json")
        else:
            print(f"[SAC] Using default hidden_dims={hidden_dims}")
        # Sync env params with training config for consistent behavior
        cli = cfg.get("cli_args", {})
        for key, attr in [("sigma_d_safe", "sigma_d_safe"),
                           ("sigma_d_critical", "sigma_d_critical"),
                           ("d_safe", "d_safe"),
                           ("d_critical", "d_critical")]:
            if key in cli and cli[key] is not None:
                setattr(env, attr, cli[key])
        # Also sync reward function params
        for key, attr in [("w_manip", "w_manip"),
                           ("w_track", "w_track"),
                           ("w_goal", "w_goal"),
                           ("w_obs", "w_obs"),
                           ("w_obs_safe", "w_obs_safe"),
                           ("w_collision", "w_collision"),
                           ("w_action", "w_action"),
                           ("d_safe", "d_safe"),
                           ("d_critical", "d_critical")]:
            if key in cli and cli[key] is not None:
                setattr(env.reward_fn, attr, cli[key])
        # Reset sigma filter
        env._last_sigma = 0.0
        # Sync observation format params (obs_k, waypoints, scene_embed)
        if "obs_k" in cli and cli["obs_k"] is not None and cli["obs_k"] > 0:
            env.obs_k = cli["obs_k"]
            env.obs_scene_embed = cli.get("obs_scene_embed", 0) or 0
            env.obs_waypoint_steps = cli.get("obs_waypoint_steps", None)
            if env.obs_waypoint_steps is not None:
                env.obs_waypoint_steps = [int(s.strip()) for s in env.obs_waypoint_steps.split(",")]
            else:
                env.obs_waypoint_steps = []
            # Recompute capsule dimension (env was initialized with obs_scene_embed=0)
            if env.obs_scene_embed > 0 and env._capsule_dists_dim == 0:
                try:
                    zero_q = np.zeros(env.n)
                    env._capsule_dists_dim = len(env.kin.get_link_capsules(zero_q))
                except Exception:
                    env._capsule_dists_dim = 12  # fallback: 12 capsules for Panda
            env.obs_dim = (env.n * 2 + 3 + 3 + 3
                           + env._capsule_dists_dim
                           + env.obs_scene_embed * 4
                           + len(env.obs_waypoint_steps) * 3)
            print(f"[SAC] Using extended observation: obs_k={env.obs_k}, "
                  f"scene_embed={env.obs_scene_embed}, "
                  f"waypoints={env.obs_waypoint_steps}, dim={env.obs_dim}")
        print(f"[SAC] Synced env params from training config")
    else:
        print(f"[SAC] Using default hidden_dims={hidden_dims}")

    dyn = ManipulatorDynamics(args.urdf)
    agent = SACAgent(
        state_dim=env.obs_dim, action_dim=env.act_dim, dynamics=dyn,
        hidden_dims=hidden_dims,
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
    )
    meta = agent.load(args.checkpoint, load_optimizers=False)
    agent.actor.eval()

    print(f"[SAC] Agent loaded. Metadata: {meta}")
    print(f"[SAC] Running policy rollouts...")
    return _run_env(env, args, "SAC",
                    get_action=lambda obs: agent.select_action(obs, deterministic=True))


def run_ppo(env, args):
    """
    Trained PPO agent: actions come from the policy network.
    """
    from agent.ppo_agent import PPOAgent
    from env.dynamics import ManipulatorDynamics

    print(f"[PPO] Loading agent from {args.checkpoint}")
    dyn = ManipulatorDynamics(args.urdf)
    agent = PPOAgent(
        state_dim=env.obs_dim, action_dim=env.act_dim, dynamics=dyn,
        n_envs=1, device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
    )
    meta = agent.load(args.checkpoint)
    agent.actor.eval()

    print(f"[PPO] Agent loaded. Metadata: {meta}")
    print(f"[PPO] Running policy rollouts...")
    return _run_env(env, args, "PPO",
                    get_action=lambda obs: agent.select_action(obs, deterministic=True))


# ---------------------------------------------------------------------------
# RRT* planner
# ---------------------------------------------------------------------------

def _evaluate_rrt_path(env, path, obstacles, start_pos, goal_pos):
    """
    Evaluate a planned RRT* joint-space path at high resolution.

    Returns dict with metrics compatible with the comparison table.
    """
    n_eval = max(200, len(path) * 5)
    alphas = np.linspace(0, 1, n_eval)

    joint_path_length = 0.0
    task_path_length = 0.0
    clearance = float("inf")
    collisions = 0
    tracking_errors = []
    obstacle_distances = []
    prev_ee = None
    prev_q = path[0]

    for i, alpha in enumerate(alphas):
        # Interpolate along path
        t = alpha * (len(path) - 1)
        idx = int(t)
        frac = t - idx
        if idx >= len(path) - 1:
            q = path[-1]
        else:
            q = (1 - frac) * path[idx] + frac * path[idx + 1]

        # Joint-space path length
        if i > 0:
            joint_path_length += np.linalg.norm(q - prev_q)
        prev_q = q

        # FK
        x_ee, _ = env.kin.forward_kinematics(q)
        if prev_ee is not None:
            task_path_length += np.linalg.norm(x_ee - prev_ee)
        prev_ee = x_ee

        # Tracking error vs linear start→goal
        x_d = (1 - alpha) * start_pos + alpha * goal_pos
        track_err = np.linalg.norm(x_ee - x_d)
        tracking_errors.append(track_err)

        # Obstacle clearance
        from planner.rrt_star import capsule_sphere_distance
        capsules = env.kin.get_link_capsules(q)
        min_d = float("inf")
        for p1, p2, cap_r in capsules:
            for obs in obstacles:
                center = np.array(obs[:3], dtype=float)
                r = float(obs[3])
                d = capsule_sphere_distance(p1, p2, cap_r, center, r)
                min_d = min(min_d, d)
        obstacle_distances.append(min_d)
        clearance = min(clearance, min_d)
        if min_d < 0:
            collisions += 1

    # Compute final metrics
    mean_error = float(np.mean(tracking_errors))
    max_error = float(np.max(tracking_errors))
    mean_d_obs = float(np.mean(obstacle_distances))
    min_d_obs = float(np.min(obstacle_distances))

    return {
        "joint_path_length": joint_path_length,
        "task_path_length": task_path_length,
        "clearance": clearance,
        "collisions": collisions,
        "n_steps": n_eval,
        "mean_error": mean_error,
        "max_error": max_error,
        "mean_d_obs": mean_d_obs,
        "min_d_obs": min_d_obs,
    }


def run_rrt_star(env, args):
    """
    RRT* motion planner: finds a collision-free joint-space path
    from start to goal, then evaluates path quality.
    """
    from planner.rrt_star import RRTStar

    # Extract obstacles
    obstacles = []
    if hasattr(env, 'sdf') and hasattr(env.sdf, 'centers') and hasattr(env.sdf, 'radii'):
        for c, r in zip(env.sdf.centers, env.sdf.radii):
            obstacles.append([float(c[0]), float(c[1]), float(c[2]), float(r)])
    if not obstacles and hasattr(args, 'scene_json') and args.scene_json is not None:
        # Fall back to reading from JSON if env doesn't have obstacles set
        sid = max(args.scene_id, 0)
        scene = load_scene_from_json(args.scene_json, sid)
        obstacles = scene.get("obstacles", [])

    start_pos = env.x_start.copy()
    goal_pos = env.x_goal.copy()
    q_start = env.q.copy()

    # Get joint limits and clamp
    q_min = env.kin.q_min
    q_max = env.kin.q_max
    if q_min is not None and q_max is not None:
        q_start = np.clip(q_start, q_min, q_max)

    # Goal IK — prefer stored goal_q from JSON (guarantees good clearance)
    from planner.rrt_star import capsule_sphere_distance
    q_goal = None
    try:
        if args.scene_json is not None and args.scene_id >= 0:
            scene = load_scene_from_json(args.scene_json, args.scene_id)
            if "goal_q" in scene:
                q_stored = np.array(scene["goal_q"])
                if q_min is not None and q_max is not None:
                    q_stored = np.clip(q_stored, q_min, q_max)
                # Verify stored goal is collision-free enough
                caps = env.kin.get_link_capsules(q_stored)
                md = float("inf")
                for p1, p2, cr in caps:
                    for o in obstacles:
                        d = capsule_sphere_distance(p1, p2, cr, np.array(o[:3]), float(o[3]))
                        md = min(md, d)
                print(f"[RRT*] Stored goal_q clearance: {md:.4f}m")
                if md >= -0.01:  # allow tiny penetration
                    q_goal = q_stored
    except Exception:
        pass

    if q_goal is None:
        # Fallback: IK with multiple seeds
        for seed_idx in range(30):
            q_seed = np.random.uniform(q_min, q_max) if seed_idx > 0 else None
            q_ik = env.kin.inverse_kinematics(goal_pos,
                                              q_init=q_seed if seed_idx > 0 else None)
            if q_ik is None:
                continue
            if q_min is not None and q_max is not None:
                q_ik = np.clip(q_ik, q_min, q_max)
            q_goal = q_ik
            # Check clearance
            caps = env.kin.get_link_capsules(q_ik)
            md = float("inf")
            for p1, p2, cr in caps:
                for o in obstacles:
                    d = capsule_sphere_distance(p1, p2, cr, np.array(o[:3]), float(o[3]))
                    md = min(md, d)
            if md >= 0.02:
                print(f"[RRT*] IK seed {seed_idx}: clearance {md:.4f}m")
                break
            q_goal = q_ik  # use best found
    if q_goal is None:
        print("[RRT*] IK failed for goal position — cannot plan")
        return {
            "label": "RRT*",
            "planning_time": 0.0, "n_nodes": 0,
            "joint_path_length": 0.0, "task_path_length": 0.0,
            "clearance": 0.0, "collisions": 0, "n_steps": 0,
            "mean_error": 0.0, "max_error": 0.0,
            "mean_d_obs": 0.0, "min_d_obs": 0.0,
        }
    if q_min is not None and q_max is not None:
        q_goal = np.clip(q_goal, q_min, q_max)

    print(f"[RRT*] Planning from start → goal ({len(obstacles)} obstacles)")
    print(f"[RRT*] Obstacles: {len(obstacles)}")

    planner = RRTStar(
        kin=env.kin,
        q_min=env.kin.q_min if hasattr(env.kin, 'q_min') else env.q_min,
        q_max=env.kin.q_max if hasattr(env.kin, 'q_max') else env.q_max,
        obstacles=obstacles,
        goal_bias=args.rrt_goal_bias,
        max_iterations=args.rrt_max_iter,
        step_size=args.rrt_step_size,
    )

    path, planning_time, n_nodes = planner.plan(q_start, q_goal)

    if not path or len(path) < 2:
        print(f"[RRT*] No path found in {planning_time:.2f}s ({n_nodes} nodes)")
        return {
            "label": "RRT*",
            "planning_time": planning_time, "n_nodes": n_nodes,
            "joint_path_length": 0.0, "task_path_length": 0.0,
            "clearance": 0.0, "collisions": 0, "n_steps": 0,
            "mean_error": 0.0, "max_error": 0.0,
            "mean_d_obs": 0.0, "min_d_obs": 0.0,
        }

    print(f"[RRT*] Path found: {len(path)} waypoints "
          f"in {planning_time:.2f}s ({n_nodes} nodes)")

    # Evaluate path quality
    metrics = _evaluate_rrt_path(env, path, obstacles, start_pos, goal_pos)
    metrics["label"] = "RRT*"
    metrics["planning_time"] = planning_time
    metrics["n_nodes"] = n_nodes

    # Print summary
    print(f"\n[RRT*] Path evaluation:")
    print(f"  Joint path length:     {metrics['joint_path_length']:.4f}")
    print(f"  Task path length:      {metrics['task_path_length']:.4f} m")
    print(f"  Min obstacle clearance: {metrics['clearance']:.4f} m")
    print(f"  Collisions:             {metrics['collisions']}")
    print(f"  Mean tracking error:    {metrics['mean_error']:.4f} m")
    print(f"  Planning time:          {planning_time:.3f}s")
    print(f"  Nodes explored:         {n_nodes}")

    return metrics



# ---------------------------------------------------------------------------
# Shared loop and reporting
# ---------------------------------------------------------------------------

def _run_env(env, args, label, get_action=None):
    """
    Generic environment stepping loop.
    If get_action is None, uses zero action (for KP/MPC modes).
    """
    tracking_errors = []
    obstacle_distances = []
    rewards = []
    collisions = 0

    print(f"\n{'Step':>6} {'Reward':>10} {'r_track':>9} {'r_obs':>9} {'r_manip':>8} "
          f"{'r_energy':>9} {'r_coll':>8} {'Track_err':>10} {'d_obs':>8}")
    print("-" * 90)

    for step in range(args.steps):

        # ------------------------------------------------------------------
        # Scene 3: update dynamic obstacle positions (moving along y-axis)
        # ------------------------------------------------------------------
        if hasattr(env, '_scene3_motion'):
            motion = env._scene3_motion
            t = step * env.dt
            new_centers = []
            for i, c_init in enumerate(motion["centers_init"]):
                # Each obstacle oscillates in y-direction with different phase
                phase = i * (2.0 * np.pi / 3.0)  # 120° phase offset
                y_offset = motion["amplitude"] * np.sin(motion["speed"] * t + phase)
                new_center = c_init.copy()
                new_center[1] = c_init[1] + y_offset
                new_centers.append(new_center)
            env.sdf.set_static_obstacles(new_centers,
                                         [env.sdf.radii[i] for i in range(len(new_centers))])
            env._sync_obstacles_to_mujoco()

        if get_action is not None:
            action = get_action(env._get_obs())
        else:
            action = np.zeros(env.act_dim)

        obs, reward, done, info = env.step(action)

        x_ee, _ = env.kin.forward_kinematics(env.q)
        track_err = np.linalg.norm(x_ee - env.x_d)
        d_obs = info["d_obs"]

        tracking_errors.append(track_err)
        obstacle_distances.append(d_obs)
        rewards.append(reward)
        if info.get("collision", False):
            collisions += 1

        if step % 100 == 0:
            flag = " COLLIDE" if info.get("collision") else ""
            print(f"{step:>6d} {reward:>10.3f} "
                  f"{info.get('r_track', 0):>9.4f} {info.get('r_obs', 0):>9.4f} "
                  f"{info.get('r_manip', 0):>8.4f} {info.get('r_energy', 0):>9.4f} "
                  f"{info.get('r_collision', 0):>8.4f} "
                  f"{track_err:>10.4f} {d_obs:>8.3f}{flag}")

        if args.render:
            env.render()
            time.sleep(env.dt)

        if done:
            # Collision doesn't terminate test — run full trajectory for evaluation
            if info.get("success", False):
                print(f"\nGoal reached at step {step}")
                break
            if step >= args.steps - 1:
                break

    return _summary(label, tracking_errors, obstacle_distances, rewards, collisions)


def _summary(label, tracking_errors, obstacle_distances, rewards, collisions=0):
    print(f"\n{'=' * 60}")
    print(f"{label} Summary")
    print(f"{'=' * 60}")
    n = len(tracking_errors)
    print(f"Steps:               {n}")
    print(f"Mean tracking error:  {np.mean(tracking_errors):.4f} m")
    print(f"Max tracking error:   {np.max(tracking_errors):.4f} m")
    print(f"Mean reward:          {np.mean(rewards):.3f}")
    print(f"Total reward:         {np.sum(rewards):.3f}")
    print(f"Mean d_obs:           {np.mean(obstacle_distances):.4f} m")
    print(f"Min d_obs:            {np.min(obstacle_distances):.4f} m")
    print(f"Collisions:           {collisions}")
    return {
        "label": label,
        "n_steps": n,
        "mean_error": np.mean(tracking_errors),
        "max_error": np.max(tracking_errors),
        "total_reward": np.sum(rewards),
        "mean_reward": np.mean(rewards),
        "mean_d_obs": np.mean(obstacle_distances),
        "min_d_obs": np.min(obstacle_distances),
        "collisions": collisions,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(results_list):
    """Print side-by-side comparison of multiple controller results."""
    if len(results_list) < 2:
        return

    print(f"\n{'=' * 70}")
    print("Controller Comparison")
    print(f"{'=' * 70}")

    n = len(results_list)
    hdr = "{:<28}" + "{:>14}" * n
    row_f = "{:<28}" + "{:>14.4f}" * n
    row_i = "{:<28}" + "{:>14d}" * n
    row_s = "{:<28}" + "{:>14}" * n

    print(hdr.format("Metric", *[r["label"] for r in results_list]))
    print("-" * 70)

    # Common metrics
    if all("mean_error" in r for r in results_list):
        print(row_f.format("Mean tracking error (m)", *[r["mean_error"] for r in results_list]))
    if all("max_error" in r for r in results_list):
        print(row_f.format("Max tracking error (m)", *[r["max_error"] for r in results_list]))
    if all("min_d_obs" in r for r in results_list):
        print(row_f.format("Min d_obs (m)", *[r["min_d_obs"] for r in results_list]))
    if all("mean_d_obs" in r for r in results_list):
        print(row_f.format("Mean d_obs (m)", *[r["mean_d_obs"] for r in results_list]))
    if all("collisions" in r for r in results_list):
        print(row_i.format("Collisions", *[r["collisions"] for r in results_list]))

    # RRT* path planning metrics
    def _fmt(r, key, is_int=False):
        """Format value or '-' if key not in result."""
        if key not in r:
            return "-"
        v = r[key]
        if is_int:
            return f"{v:d}" if isinstance(v, int) else str(v)
        return f"{v:.4f}"

    if any("joint_path_length" in r for r in results_list):
        print(f"\n  --- Path Planning Metrics ---")
        row_any = "{:<28}" + "{:>14}" * n
        print(row_any.format("  Joint path length",
              *[_fmt(r, "joint_path_length") for r in results_list]))
        print(row_any.format("  Task path length (m)",
              *[_fmt(r, "task_path_length") for r in results_list]))
        print(row_any.format("  Min clearance (m)",
              *[_fmt(r, "clearance") for r in results_list]))
        print(row_any.format("  Planning time (s)",
              *[_fmt(r, "planning_time") for r in results_list]))
        print(row_any.format("  Nodes explored",
              *[_fmt(r, "n_nodes", is_int=True) for r in results_list]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Unified manipulator test script")
    p.add_argument("--method", type=str, default="kp",
                   choices=["kp", "mpc", "sac", "ppo", "rrt_star"],
                   help="Control method to test")
    p.add_argument("--render", action="store_true", help="Enable MuJoCo viewer")
    p.add_argument("--steps", type=int, default=1000, help="Max simulation steps")
    p.add_argument("--horizon", type=int, default=10, help="MPC prediction horizon")
    p.add_argument("--n_obstacles", type=int, default=0, help="Number of obstacles")
    p.add_argument("--obs_radius", type=float, default=0.08, help="Obstacle radius")
    p.add_argument("--urdf", type=str, default=_DEFAULT_URDF)
    p.add_argument("--xml", type=str, default=_DEFAULT_XML)
    p.add_argument("--checkpoint", type=str, default=_DEFAULT_CKPT,
                   help="Checkpoint path for RL agent")
    p.add_argument("--compare", action="store_true",
                   help="Run all available methods and compare")

    # Scene generation / loading
    p.add_argument("--use_trajectory_generator", action="store_true",
                   help="Use TrajectoryGenerator for random collision-free scenes")
    p.add_argument("--scene_json", type=str, default=None,
                   help=f"Path to scene JSON (default: {_DEFAULT_SCENE_JSON})")
    p.add_argument("--scene_id", type=int, default=-1,
                   help="Scene ID to load from --scene_json (default: auto-detect)")

    # Paper experiment scenarios
    p.add_argument("--scene", type=str, default=None,
                   choices=["scene1", "scene2", "scene3"],
                   help="Paper evaluation scenario: scene1=sparse+figure8, "
                        "scene2=dense+narrow, scene3=dynamic+obstacles")

    # RRT* parameters
    p.add_argument("--rrt_max_iter", type=int, default=3000,
                   help="Max RRT* iterations")
    p.add_argument("--rrt_step_size", type=float, default=0.25,
                   help="RRT* step size in normalized joint space")
    p.add_argument("--rrt_goal_bias", type=float, default=0.15,
                   help="RRT* goal sampling bias (0-1)")
    return p.parse_args()


def setup_env(args):
    """
    Create and configure ManipulatorEnv.

    Scene resolution order (highest priority first):
      1. --scene_json / --scene_id:  load a specific scene from JSON
      2. --use_trajectory_generator:  random collision-free scene via TrajectoryGenerator
      3. Default:                     fixed trajectory with random obstacles
    """
    # Map method names to controller parameter
    if args.method in ("kp", "sac", "ppo", "rrt_star"):
        ctrl = "rl"
    else:
        ctrl = args.method

    # When loading from JSON, obstacle count is determined by the scene
    n_obstacles = args.n_obstacles
    if args.scene_json is not None:
        sid = max(args.scene_id, 0)
        scene = load_scene_from_json(args.scene_json, sid)
        n_obstacles = len(scene["obstacles"])

    env = ManipulatorEnv(
        urdf_path=args.urdf,
        xml_path=args.xml,
        n_joints=7,
        dt=0.02,
        episode_len=args.steps,
        n_obstacles=n_obstacles,
        obs_radius=args.obs_radius,
        controller=ctrl,
        mpc_horizon=args.horizon,
        use_trajectory_generator=args.use_trajectory_generator,
    )
    env.reset()

    # Paper scenarios: highest priority (overrides JSON / generator / defaults)
    if args.scene is not None:
        if args.scene == "scene1":
            _scene1_setup(env)
        elif args.scene == "scene2":
            _scene2_setup(env)
        elif args.scene == "scene3":
            _scene3_setup(env)

    # Override with JSON scene if specified (takes second priority)
    elif args.scene_json is not None:
        sid = max(args.scene_id, 0)  # default to scene 0 if not set
        scene = load_scene_from_json(args.scene_json, sid)
        if not apply_scene(env, scene):
            sys.exit(1)

    return env


def run_single(args):
    """Run a single method."""
    env = setup_env(args)

    if args.method == "kp":
        results = run_kp(env, args)
    elif args.method == "mpc":
        results = run_mpc(env, args)
    elif args.method == "sac":
        results = run_rl(env, args, agent=None)
    elif args.method == "ppo":
        results = run_ppo(env, args)
    elif args.method == "rrt_star":
        results = run_rrt_star(env, args)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    if env.mj_data is not None and args.render:
        # Keep viewer open after simulation ends
        print("\nClose viewer to exit.")
        if hasattr(env, '_viewer') and env._viewer.is_running():
            while env._viewer.is_running():
                time.sleep(0.1)

    return results


def run_comparison(args):
    """Run all methods and compare."""
    results_list = []

    for method in ["kp", "mpc"]:
        print(f"\n{'#' * 60}")
        print(f"# Running {method.upper()}")
        print(f"{'#' * 60}")
        args.method = method
        env = setup_env(args)

        if method == "kp":
            r = run_kp(env, args)
        elif method == "mpc":
            r = run_mpc(env, args)

        results_list.append(r)

        # Clean up viewer
        if hasattr(env, '_viewer'):
            try:
                env._viewer.close()
            except Exception:
                pass

    # Try SAC if checkpoint exists
    if os.path.exists(args.checkpoint):
        print(f"\n{'#' * 60}")
        print(f"# Running SAC (checkpoint: {args.checkpoint})")
        print(f"{'#' * 60}")
        args.method = "sac"
        env = setup_env(args)
        r = run_rl(env, args, agent=None)
        results_list.append(r)
    else:
        print(f"\n[SKIP] RL checkpoint not found at {args.checkpoint}")

    # RRT* (always runs — no checkpoint needed)
    print(f"\n{'#' * 60}")
    print(f"# Running RRT*")
    print(f"{'#' * 60}")
    args.method = "rrt_star"
    env = setup_env(args)
    r = run_rrt_star(env, args)
    results_list.append(r)

    print_comparison(results_list)


def main():
    args = parse_args()

    # Auto-detect default scene JSON if scene_id is given without explicit path
    if args.scene_json is None and args.scene_id >= 0:
        if os.path.exists(_DEFAULT_SCENE_JSON):
            args.scene_json = _DEFAULT_SCENE_JSON
            print(f"[Auto] Using default scene JSON: {_DEFAULT_SCENE_JSON} (scene {args.scene_id})")
        else:
            print(f"[Warning] --scene_id={args.scene_id} given but no scene JSON found at "
                  f"{_DEFAULT_SCENE_JSON}")

    print("=" * 60)
    print("Unified Manipulator Test")
    print("=" * 60)
    print(f"Method:    {args.method}")
    print(f"Steps:     {args.steps}")
    print(f"Render:    {args.render}")
    print(f"URDF:      {args.urdf}")
    print(f"XML:       {args.xml}")

    # Resolve relative paths relative to project root (_ROOT)
    if args.checkpoint and not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(_ROOT, args.checkpoint)
    if args.scene_json and not os.path.isabs(args.scene_json):
        args.scene_json = os.path.join(_ROOT, args.scene_json)
    if args.scene_json is not None:
        print(f"Scene:     {args.scene_json} (id={args.scene_id})")
    elif args.use_trajectory_generator:
        print(f"Scene:     TrajectoryGenerator (random)")
    else:
        print(f"Obstacles: {args.n_obstacles} (radius={args.obs_radius})")

    if args.compare:
        run_comparison(args)
    else:
        run_single(args)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
