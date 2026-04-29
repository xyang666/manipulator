"""
test_scene_vis.py
-----------------
Load a scene from trajectories.json, animate using env's KP controller,
display obstacles/target point, and output collision info at each step.

Usage:
    cd code/
    .venv/bin/python test_scene_vis.py --scene_id 0          # 打开 viewer
    .venv/bin/python test_scene_vis.py --scene_id 0 --headless  # 只输出碰撞信息
"""

import sys, os, json, argparse, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

_TRAJ_JSON = os.path.join(_ROOT, "results/trajectories.json")
_DEFAULT_URDF = os.path.join(_HERE, ".venv/lib/python3.12/site-packages"
                             "/cmeel.prefix/share/example-robot-data"
                             "/robots/panda_description/urdf/panda.urdf")
_DEFAULT_XML = os.path.join(_ROOT, "models/panda_scene.xml")

from env.manipulator_env import ManipulatorEnv


def load_scene(scene_id: int):
    with open(_TRAJ_JSON) as f:
        scenes = json.load(f)
    for s in scenes:
        if s["scene_id"] == scene_id:
            return s
    raise ValueError(f"Scene {scene_id} not found")


def main():
    parser = argparse.ArgumentParser(description="Visualize a scene from trajectories.json")
    parser.add_argument("--scene_id", type=int, default=1)
    parser.add_argument("--headless", action="store_true", help="No viewer, print collision info only")
    parser.add_argument("--urdf", default=_DEFAULT_URDF)
    parser.add_argument("--xml", default=_DEFAULT_XML)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    # Load scene
    scene = load_scene(args.scene_id)
    obstacles = scene["obstacles"]
    start_pos = np.array(scene["start"])
    goal_pos = np.array(scene["goal"])
    n_obs = len(obstacles)

    print(f"\n{'='*60}")
    print(f"Scene {args.scene_id}")
    print(f"{'='*60}")
    print(f"Start:         [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
    print(f"Goal:          [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]")
    print(f"Distance:      {np.linalg.norm(goal_pos - start_pos):.3f} m")
    print(f"Obstacles:     {n_obs}")
    print(f"Manipulability: {scene.get('manipulability_mean', 'N/A'):.4f}")

    n_steps = max(int(np.linalg.norm(goal_pos - start_pos) / 0.001), 100)

    # Create environment (use the scene's obstacle count)
    env = ManipulatorEnv(
        urdf_path=args.urdf,
        xml_path=args.xml,
        n_joints=7,
        dt=args.dt,
        episode_len=n_steps + 10,
        n_obstacles=n_obs,
        obs_radius=0.03,
        use_trajectory_generator=False,
    )

    # Override obstacles from scene
    obs_centers = [np.array(o[:3]) for o in obstacles]
    obs_radii = [o[3] for o in obstacles]
    env.sdf.set_static_obstacles(obs_centers, obs_radii)

    # IK for start configuration
    q_start = env.kin.inverse_kinematics(start_pos)
    if q_start is None:
        print("ERROR: IK failed for start position")
        sys.exit(1)

    # Reset env to start configuration
    env.q = q_start.copy()
    env.dq = np.zeros(7)
    env.x_start = start_pos.copy()
    env.x_goal = goal_pos.copy()
    env.x_d = start_pos.copy()
    env.dx_d = np.zeros(6)
    env.step_count = 0

    # Sync MuJoCo
    if env.mj_data is not None:
        env.mj_data.qpos[:7] = q_start
        env.mj_data.qvel[:7] = 0.0
        env._sync_obstacles_to_mujoco()
        mujoco.mj_forward(env.mj_model, env.mj_data)

    # Set up trajectory tracking
    direction = goal_pos - start_pos
    dist = np.linalg.norm(direction)
    env.dx_d = np.zeros(6)
    env.dx_d[:3] = direction / dist * 0.1  # 0.1 m/s toward goal

    if args.headless:
        # Headless mode: step through with KP controller, print collision
        print(f"\n{'Step':>6} {'d_obs':>8} {'MJ contacts':>12} {'obs_ct':>8} {'penetration':>12}")
        print("-" * 50)

        action = np.zeros(13)
        for i in range(n_steps + 1):
            env.x_d = (1 - i / n_steps) * start_pos + (i / n_steps) * goal_pos
            _, _, done, info = env.step(action)

            d_obs = info.get("d_obs", 0.0)

            if env.collision_detector is not None and env.collision_detector.has_mujoco:
                obs_pen, n_obs_ct = env.collision_detector.detect_obstacle_collisions()
                total_pen, max_pen, n_total = env.collision_detector.detect_collisions()
            else:
                n_total, n_obs_ct, total_pen = 0, 0, 0.0

            flag = " COLLIDE" if (d_obs < 0.02 or n_obs_ct > 0) else ""
            print(f"{i / n_steps:6.2f} {d_obs:8.4f} {n_total:12d} {n_obs_ct:8d} {total_pen:12.4f}{flag}")

        return

    # --- Viewer mode ---
    if env.mj_data is None:
        print("ERROR: MuJoCo not available")
        sys.exit(1)

    # Launch viewer
    mujoco.mj_forward(env.mj_model, env.mj_data)
    env.render()

    # First render
    env._draw_visualizations()
    env._viewer.sync()

    print(f"\nKP controller tracking from start to goal ({n_obs} obstacles)...")
    print("Close viewer to exit.")

    action = np.zeros(13)
    for i in range(n_steps):
        if not env._viewer.is_running():
            break

        # Update target
        progress = min(1.0, i / n_steps)
        env.x_d = (1 - progress) * start_pos + progress * goal_pos

        # Step with KP PD controller (zero action = pure tracking, no relaxation)
        _, _, _, info = env.step(action)

        if not env._viewer.is_running():
            break

        # Draw and sync
        env._draw_visualizations()
        env._viewer.sync()

        d_obs = info.get("d_obs", 0.0)
        collision = info.get("collision", False)
        if d_obs < 0.02 or collision:
            print(f"  step {i:3d} | d_obs={d_obs:+.4f} {'COLLISION' if collision else 'near'}")
        elif i % 20 == 0:
            print(f"  step {i:3d} | d_obs={d_obs:+.4f}")

        time.sleep(env.dt)

    # Keep viewer open
    while env._viewer.is_running():
        time.sleep(0.1)

    print("Done.")


if __name__ == "__main__":
    try:
        import mujoco
    except ImportError:
        pass
    main()
