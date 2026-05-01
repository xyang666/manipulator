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
    python test.py --method rl  --checkpoint ../checkpoints/sac_pirl.pt --render
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
    env.dx_d = np.zeros(6)
    env.step_count = 0
    env.path_param = 0.0
    env.ee_trajectory.clear()

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
# Control methods
# ---------------------------------------------------------------------------

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

    print(f"[RL] Loading agent from {args.checkpoint}")
    dyn = ManipulatorDynamics(args.urdf)
    agent = SACAgent(
        state_dim=env.obs_dim, action_dim=env.act_dim, dynamics=dyn,
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
    )
    meta = agent.load(args.checkpoint)
    agent.actor.eval()

    print(f"[RL] Agent loaded. Metadata: {meta}")
    print(f"[RL] Running policy rollouts...")
    return _run_env(env, args, "RL",
                    get_action=lambda obs: agent.select_action(obs, deterministic=True))



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

    print(f"\n{'Step':>6} {'Reward':>10} {'Track_err':>10} {'d_obs':>8}")
    print("-" * 50)

    for step in range(args.steps):
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
            print(f"{step:>6d} {reward:>10.3f} {track_err:>10.4f} {d_obs:>8.3f}{flag}")

        if args.render:
            env.render()
            time.sleep(env.dt)

        if done:
            print(f"\nEpisode terminated at step {step} "
                  f"(success={info.get('success', False)})")
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

    headers = ["Metric"] + [r["label"] for r in results_list]
    col_width = 18
    header_fmt = "{:<22}" + "{:>16}" * (len(results_list))
    row_fmt = "{:<22}" + "{:>16.4f}" * (len(results_list))

    print(header_fmt.format(*headers))
    print("-" * 70)

    print(row_fmt.format("Mean tracking error (m)", *[r["mean_error"] for r in results_list]))
    print(row_fmt.format("Max tracking error (m)", *[r["max_error"] for r in results_list]))
    print(row_fmt.format("Total reward", *[r["total_reward"] for r in results_list]))
    print(row_fmt.format("Mean reward", *[r["mean_reward"] for r in results_list]))
    print(row_fmt.format("Min d_obs (m)", *[r["min_d_obs"] for r in results_list]))
    print(row_fmt.format("Mean d_obs (m)", *[r["mean_d_obs"] for r in results_list]))
    print("{:<22}".format("Collisions") + "".join(f"{r['collisions']:>16d}" for r in results_list))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Unified manipulator test script")
    p.add_argument("--method", type=str, default="kp",
                   choices=["kp", "mpc", "rl"],
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
    if args.method in ("kp", "rl"):
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

    # Override with JSON scene if specified (takes highest priority)
    if args.scene_json is not None:
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
    elif args.method == "rl":
        results = run_rl(env, args, agent=None)
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

    # Try RL if checkpoint exists
    if os.path.exists(args.checkpoint):
        print(f"\n{'#' * 60}")
        print(f"# Running RL (checkpoint: {args.checkpoint})")
        print(f"{'#' * 60}")
        args.method = "rl"
        env = setup_env(args)
        r = run_rl(env, args, agent=None)
        results_list.append(r)
    else:
        print(f"\n[SKIP] RL checkpoint not found at {args.checkpoint}")

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
