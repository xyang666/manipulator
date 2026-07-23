"""
test_capsule_collision.py
-------------------------
Check capsule-capsule self-collisions and optionally render them.
Green capsules = safe, Red capsules = colliding.

Usage:
    code/.venv/bin/python code/test_capsule_collision.py

Author: xie yang
Date:   2025-06

"""
import sys, json, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from env.manipulator_env import ManipulatorEnv
import mujoco

# ── Config ──
SCENE_ID = 3  # scene 22 (0-indexed)
SHOW_RENDER = True

# ── Load scene ──
with open("results/challenge_stage1.json") as f:
    scenes = json.load(f)
scene = scenes[SCENE_ID]
start_q = np.array(scene["start_q"])
print(f"Scene {scene['scene_id']}  (difficulty={scene['difficulty']})")

# ── Create env ──
env = ManipulatorEnv(
    urdf_path="panda_description/urdf/panda.urdf",
    xml_path="models/panda_scene.xml",
    n_joints=7,
    dt=0.02,
    episode_len=500,
    n_obstacles=len(scene["obstacles"]),
    controller="rl",
)
env.sdf.centers = [np.array(o[:3]) for o in scene["obstacles"]]
env.sdf.radii = [float(o[3]) for o in scene["obstacles"]]
env.x_start = np.array(scene["start"])
env.x_goal = np.array(scene["goal"])
env.q = start_q.copy()
env.dq = np.zeros(env.n)
env.mj_data.qpos[:env.n] = start_q
mujoco.mj_forward(env.mj_model, env.mj_data)

# ── Compute self-collision info ──
capsules = env.kin.get_link_capsules(start_q)
pairs = env.kin.get_self_collision_pairs()
dists = env.kin.compute_self_distances(start_q)

link_names = [
    "Link0", "Link1", "Link2", "Link3", "Link4", "Link5",
    "Link6", "Link7", "Hand", "L_finger", "R_finger",
]
n_colliding = 0
print("\nSelf-collision distances (surface-to-surface, ≤0 = penetrating):")
for pi, (i, j) in enumerate(pairs):
    ni = link_names[i] if i < len(link_names) else f"C{i}"
    nj = link_names[j] if j < len(link_names) else f"C{j}"
    tag = " ⛔ COLLISION" if dists[pi] <= 0 else " ✓ safe"
    if dists[pi] <= 0:
        n_colliding += 1
        print(f"  ⛔ {ni:15s} <-> {nj:15s}: {dists[pi]:+.4f}m  ** COLLISION **")
    else:
        print(f"  ✓ {ni:15s} <-> {nj:15s}: {dists[pi]:+.4f}m")

print(f"\n{n_colliding} / {len(pairs)} non-adjacent capsule pairs in collision")

# ── Render if display available ──
if SHOW_RENDER:
    try:
        HAS_DISPLAY = bool(
            __import__("os").environ.get("DISPLAY")
            or __import__("os").environ.get("WAYLAND_DISPLAY")
        )
    except Exception:
        HAS_DISPLAY = False

    if not HAS_DISPLAY:
        print("\nNo display detected. Skipping viewer.")
        print("To render: set DISPLAY=:0 or run with X11 forwarding.")
    else:
        print("\nLaunching viewer... (close window to exit)")

        # Monkey-patch to color capsules by collision status
        def _draw_with_self_collision(self):
            scene = self._viewer.user_scn
            scene.ngeom = 0

            capsules = self.kin.get_link_capsules(self.q)
            pairs = self.kin.get_self_collision_pairs()
            dists = self.kin.compute_self_distances(self.q)

            n_caps = len(capsules)
            cap_coll = np.zeros(n_caps, dtype=bool)
            for pi, (i, j) in enumerate(pairs):
                if dists[pi] <= 0:
                    cap_coll[i] = True
                    cap_coll[j] = True

            for ci, (p1, p2, cap_r) in enumerate(capsules):
                if scene.ngeom >= scene.maxgeom:
                    break
                center = (p1 + p2) / 2
                length = np.linalg.norm(p2 - p1)
                if length < 1e-6:
                    continue
                direction = (p2 - p1) / length
                z_axis = np.array([0, 0, 1])
                rot_ax = np.cross(z_axis, direction)
                rn = np.linalg.norm(rot_ax)
                if rn > 1e-6:
                    rot_ax /= rn
                    c = np.dot(z_axis, direction)
                    ang = np.arccos(np.clip(c, -1.0, 1.0))
                    K = np.array([[0, -rot_ax[2], rot_ax[1]],
                                  [rot_ax[2], 0, -rot_ax[0]],
                                  [-rot_ax[1], rot_ax[0], 0]])
                    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
                else:
                    R = np.eye(3) if np.dot(z_axis, direction) > 0 else np.diag([1, 1, -1])
                size = np.array([cap_r, length / 2, 0])
                color = np.array([1, 0.2, 0.2, 0.6]) if cap_coll[ci] else np.array([0.2, 1, 0.2, 0.4])
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom],
                                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                                    size, center, R.flatten(), color)
                scene.ngeom += 1

            # Obstacles
            for i, c in enumerate(self.sdf.centers):
                if scene.ngeom >= scene.maxgeom:
                    break
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom],
                                    mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([self.sdf.radii[i], 0, 0]),
                                    c, np.eye(3).flatten(),
                                    np.array([1, 0, 0, 0.3]))
                scene.ngeom += 1
            # Target
            if scene.ngeom < scene.maxgeom:
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom],
                                    mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([0.02, 0, 0]),
                                    self.x_goal, np.eye(3).flatten(),
                                    np.array([1, 1, 0, 1]))
                scene.ngeom += 1

        env._draw_visualizations = _draw_with_self_collision.__get__(env, ManipulatorEnv)

        # Hide robot body geoms so capsules are visible
        for i in range(env.mj_model.ngeom):
            body_id = env.mj_model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name and ("panda" in body_name.lower() or "link" in body_name.lower()):
                env.mj_model.geom_rgba[i, 3] = 0.0

        env.render()
        try:
            while env._viewer and env._viewer.is_running():
                mujoco.mj_forward(env.mj_model, env.mj_data)
                env._draw_visualizations()
                env._viewer.sync()
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        finally:
            if hasattr(env, '_viewer'):
                env._viewer.close()
            print("Viewer closed.")
