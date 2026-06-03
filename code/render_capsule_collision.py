"""
Launch MuJoCo viewer highlighting Link1 (red) and Link3 (blue) capsule collision.
"""
import sys, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import mujoco
from env.manipulator_env import ManipulatorEnv
from env.kinematics import ManipulatorKinematics

# Load scene 22
with open("results/challenge_stage1.json") as f:
    scene = json.load(f)[21]
start_q = np.array(scene["start_q"])

env = ManipulatorEnv(urdf_path="panda_description/urdf/panda.urdf",
                      xml_path="models/panda_scene.xml", n_joints=7,
                      dt=0.02, episode_len=500,
                      n_obstacles=len(scene["obstacles"]), controller="rl")
env.sdf.centers = [np.array(o[:3]) for o in scene["obstacles"]]
env.sdf.radii = [float(o[3]) for o in scene["obstacles"]]
env.x_start = np.array(scene["start"])
env.x_goal = np.array(scene["goal"])
env.q = start_q.copy()
env.dq = np.zeros(env.n)
env.mj_data.qpos[:7] = start_q
mujoco.mj_forward(env.mj_model, env.mj_data)

capsules = env.kin.get_link_capsules(start_q)
# Link1=caps[1], Link3=caps[3]

def make_capsule_geom(scn, p1, p2, radius, color):
    center = (p1 + p2) / 2
    length = np.linalg.norm(p2 - p1)
    if length < 1e-6: return
    direction = (p2 - p1) / length
    z = np.array([0, 0, 1])
    ra = np.cross(z, direction)
    rn = np.linalg.norm(ra)
    if rn > 1e-6:
        ra /= rn
        c = np.dot(z, direction)
        ang = np.arccos(np.clip(c, -1.0, 1.0))
        K = np.array([[0, -ra[2], ra[1]],[ra[2], 0, -ra[0]],[-ra[1], ra[0], 0]])
        R = np.eye(3) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)
    else:
        R = np.eye(3) if np.dot(z, direction) > 0 else np.diag([1, 1, -1])
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.array([radius, length/2, 0]),
                        center, R.flatten(), np.array(color, dtype=float))
    scn.ngeom += 1

# Override draw: show robot + highlighted capsules
_orig_draw = env._draw_visualizations
def _my_draw(self):
    scene = self._viewer.user_scn
    scene.ngeom = 0
    # Redraw Link1+L3 capsules
    make_capsule_geom(scene, capsules[1][0], capsules[1][1], capsules[1][2], [1, 0.1, 0.1, 0.6])
    make_capsule_geom(scene, capsules[3][0], capsules[3][1], capsules[3][2], [0.1, 0.4, 1, 0.6])
    # Obstacles
    for i, c in enumerate(env.sdf.centers):
        if scene.ngeom >= scene.maxgeom: break
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([env.sdf.radii[i], 0, 0]),
                            c, np.eye(3).flatten(), np.array([1, 0, 0, 0.3]))
        scene.ngeom += 1
    # Target
    if scene.ngeom < scene.maxgeom:
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.02, 0, 0]),
                            env.x_goal, np.eye(3).flatten(), np.array([1, 1, 0, 1]))
        scene.ngeom += 1

env._draw_visualizations = _my_draw.__get__(env, ManipulatorEnv)

print("Viewer launched. Close window to exit.")
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
    print("Done")
