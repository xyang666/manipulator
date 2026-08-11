#!/usr/bin/env python3
"""
visualize_scene.py
------------------
可视化训练场景：E-Walker 机械臂 + 障碍物 + 起点/终点 + 胶囊体。

用法:
    # 按索引查看场景
    code/.venv/bin/python code/visualize_scene.py 0

    # 按场景 ID 查看
    code/.venv/bin/python code/visualize_scene.py whole_body-train-00000

    # 查看验证集
    code/.venv/bin/python code/visualize_scene.py 0 --val

    # 隐藏胶囊体（只看机器人网格）
    code/.venv/bin/python code/visualize_scene.py 0 --no-capsules

    # 手动指定场景文件
    code/.venv/bin/python code/visualize_scene.py 0 \\
        --scene-json results/ewalker_scenes/confined_space/test.json
"""

import sys, json, numpy as np, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import mujoco
import mujoco.viewer
from env.manipulator_env import ManipulatorEnv
from env.kinematics import ManipulatorKinematics


def make_capsule_geom(scn, p1, p2, radius, color):
    """Draw a capsule in the MuJoCo scene geometry overlay."""
    if scn.ngeom >= scn.maxgeom:
        return
    center = (p1 + p2) / 2
    length = np.linalg.norm(p2 - p1)
    if length < 1e-6:
        return
    direction = (p2 - p1) / length
    z = np.array([0, 0, 1])
    ra = np.cross(z, direction)
    rn = np.linalg.norm(ra)
    if rn > 1e-6:
        ra /= rn
        c = np.dot(z, direction)
        ang = np.arccos(np.clip(c, -1.0, 1.0))
        K = np.array([
            [0, -ra[2], ra[1]],
            [ra[2], 0, -ra[0]],
            [-ra[1], ra[0], 0]
        ])
        R = np.eye(3) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)
    else:
        R = np.eye(3) if np.dot(z, direction) > 0 else np.diag([1, 1, -1])
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, length/2, 0]),
        center, R.flatten(), np.array(color, dtype=float),
    )
    scn.ngeom += 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize E-Walker training scene")
    parser.add_argument("scene_id", type=str,
                        help="Scene index (0-based) or full scene ID string")
    parser.add_argument("--scene-json", type=str,
                        default="results/ewalker_scenes/curriculum/train.json",
                        help="Path to scene JSON file")
    parser.add_argument("--val", action="store_true",
                        help="Use validation.json instead of train.json")
    parser.add_argument("--no-capsules", action="store_true",
                        help="Hide capsule overlay")
    parser.add_argument("--steps", type=int, default=500,
                        help="Max simulation steps")
    parser.add_argument("--slow", action="store_true",
                        help="Run KP tracking instead of static view")
    args = parser.parse_args()

    # Resolve scene file
    scene_json = args.scene_json
    if args.val:
        scene_json = scene_json.replace("train.json", "validation.json")
        if scene_json == args.scene_json:
            scene_json = "results/ewalker_scenes/curriculum/validation.json"

    # Load scene
    with open(scene_json) as f:
        scenes = json.load(f)

    # Try by index first, then by string ID
    try:
        idx = int(args.scene_id)
        scene = scenes[idx]
    except ValueError:
        matches = [s for s in scenes if s["scene_id"] == args.scene_id]
        if not matches:
            print(f"Scene '{args.scene_id}' not found. Available IDs:")
            for s in scenes[:5]:
                print(f"  {s['scene_id']}")
            print(f"  ... ({len(scenes)} total)")
            return 1
        scene = matches[0]

    sid = scene["scene_id"]
    start = np.array(scene["start"])
    goal = np.array(scene["goal"])
    obstacles = scene.get("obstacles", [])
    n_obs = len(obstacles)
    print(f"Scene: {sid}")
    print(f"  start: {start}")
    print(f"  goal:  {goal}")
    print(f"  obstacles: {n_obs}")

    # Create environment
    env = ManipulatorEnv(
        urdf_path="ewalker_description/urdf/ewalker.urdf",
        xml_path="models/ewalker_scene.xml",
        n_joints=7, dt=0.02, episode_len=args.steps,
        n_obstacles=max(n_obs, 1), controller="rl", path_deadzone=0.10,
    )
    env.reset()

    # Apply scene
    env.x_start = start.copy()
    env.x_goal = goal.copy()
    env.x_d = start.copy()

    # Use start_q if available, else IK
    if "start_q" in scene:
        env.q[:] = np.array(scene["start_q"])
    env.dq[:] = 0
    if env.mj_data is not None:
        env.mj_data.qpos[:7] = env.q
        env.mj_data.qvel[:7] = 0
        mujoco.mj_forward(env.mj_model, env.mj_data)

    # Setup obstacles
    centers = [np.array(o[:3]) for o in obstacles]
    radii = [o[3] for o in obstacles]
    env.sdf.set_static_obstacles(centers, radii)
    if hasattr(env, '_sync_obstacles_to_mujoco'):
        env._sync_obstacles_to_mujoco()

    # Pre-compute capsule positions at start q
    capsules = env.kin.get_link_capsules(env.q)

    # Override draw function
    def _my_draw(self):
        scn = self._viewer.user_scn
        scn.ngeom = 0

        if not args.no_capsules:
            # Draw capsules (semi-transparent blue)
            for p1, p2, r in capsules:
                make_capsule_geom(scn, p1, p2, r, [0.3, 0.5, 1.0, 0.35])

        # Draw obstacles (red spheres)
        for c, r in zip(centers, radii):
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([max(r, 0.01), 0, 0]),
                c, np.eye(3).flatten(), np.array([1, 0.2, 0.2, 0.4]),
            )
            scn.ngeom += 1

        # Draw start (green) and goal (yellow) markers
        for pos, color in [(start, [0.2, 1.0, 0.2, 0.6]),
                           (goal, [1.0, 1.0, 0.2, 0.6])]:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.02, 0, 0]),
                pos, np.eye(3).flatten(), np.array(color),
            )
            scn.ngeom += 1

        # Draw target line (dashed approximation via small spheres)
        n_pts = 20
        for i in range(n_pts):
            if scn.ngeom >= scn.maxgeom:
                break
            t = (i + 0.5) / n_pts
            p = (1 - t) * start + t * goal
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.005, 0, 0]),
                p, np.eye(3).flatten(), np.array([0.5, 0.5, 0.5, 0.3]),
            )
            scn.ngeom += 1

    env._draw_visualizations = _my_draw.__get__(env, ManipulatorEnv)

    print("\nViewer opened. Close window to exit.")
    if args.slow:
        print("KP tracking...")
        env.render()
        import time
        try:
            while env._viewer and env._viewer.is_running():
                action = np.zeros(env.act_dim)
                env.step(action)
                mujoco.mj_forward(env.mj_model, env.mj_data)
                env._draw_visualizations()
                env._viewer.sync()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
    else:
        # Static view: sync once, wait for close
        env.render()
        mujoco.mj_forward(env.mj_model, env.mj_data)
        env._draw_visualizations()
        env._viewer.sync()
        try:
            while env._viewer and env._viewer.is_running():
                env._draw_visualizations()
                env._viewer.sync()
                __import__('time').sleep(0.03)
        except KeyboardInterrupt:
            pass
        finally:
            if hasattr(env, '_viewer'):
                env._viewer.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
