#!/usr/bin/env python3
"""Test classical baselines on confined_space (10-obstacle) scenes."""
import sys, json, numpy as np
sys.path.insert(0, "code")
from env.manipulator_env import ManipulatorEnv
from env.kinematics import ManipulatorKinematics

# Inline gradient projection (avoids runner.py import)
def _gradient_projection_action(env):
    if env.sdf.n_obs == 0:
        return np.zeros(env.act_dim)
    gradient = np.zeros(env.n)
    eps = 1e-4
    for joint in range(env.n):
        q_plus, q_minus = env.q.copy(), env.q.copy()
        q_plus[joint] += eps; q_minus[joint] -= eps
        d_plus = env.sdf.min_distance(np.zeros(3), q_plus, kinematics=env.kin)
        d_minus = env.sdf.min_distance(np.zeros(3), q_minus, kinematics=env.kin)
        gradient[joint] = (d_plus - d_minus) / (2 * eps)
    basis = env.kin.null_space_basis_position(env.q)
    action = np.zeros(env.act_dim)
    action[3:] = np.clip(basis.T @ gradient, -0.5, 0.5)
    return action

with open("results/ewalker_scenes/curriculum/train.json") as f:
    scenes = json.load(f)
cs = [s for s in scenes if "confined_space" in s["scene_id"]]
method = sys.argv[1] if len(sys.argv) > 1 else "gp"
n = min(len(cs), 30)
print(f"Testing {method} on {n} confined_space scenes\n")

results = {"success": 0, "collision": 0, "timeout": 0}
for i, scene in enumerate(cs[:n]):
    n_obs = len(scene["obstacles"])
    use_cbf = (method == "cbf")
    env = ManipulatorEnv(
        urdf_path="ewalker_description/urdf/ewalker.urdf",
        xml_path="models/ewalker_scene.xml",
        n_joints=7, dt=0.02, episode_len=500, n_obstacles=n_obs,
        controller="rl", path_deadzone=0.10,
        use_cbf=use_cbf, cbf_alpha=1.0)
    env.reset()
    env.x_start = np.array(scene["start"]); env.x_goal = np.array(scene["goal"])
    env.x_d = env.x_start.copy()
    env.q[:] = np.array(scene["start_q"]); env.dq[:] = 0
    import mujoco
    env.mj_data.qpos[:7] = env.q; mujoco.mj_forward(env.mj_model, env.mj_data)
    centers = [np.array(o[:3]) for o in scene["obstacles"]]
    radii = [float(o[3]) for o in scene["obstacles"]]
    env.sdf.set_static_obstacles(centers, radii)
    env._sync_obstacles_to_mujoco()

    collided = False
    for step in range(500):
        if method == "gp":
            a = _gradient_projection_action(env)
        else:
            a = np.zeros(env.act_dim)
        _, _, done, info = env.step(a)
        if info.get("collision"):
            collided = True
        if done:
            break

    path_complete = env.path_param >= 0.99
    if path_complete and not collided:
        results["success"] += 1
    elif collided:
        results["collision"] += 1
    else:
        results["timeout"] += 1

    tag = "OK" if path_complete and not collided else ("COLL" if collided else "TO")
    print(f"  [{i+1:2d}/{n}] {scene['scene_id']}: {tag}  step={step+1}  param={env.path_param:.3f}")

name = "Gradient Projection" if method == "gp" else "CBF-QP"
print(f"\n{name} (confined_space, {n} scenes):")
print(f"  success={results['success']}/{n} ({100*results['success']/n:.0f}%)")
print(f"  collision={results['collision']}  timeout={results['timeout']}")
