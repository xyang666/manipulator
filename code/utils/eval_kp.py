#!/usr/bin/env python3
"""Evaluate KP on all curriculum scenes, with failure analysis."""

import json, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.manipulator_env import ManipulatorEnv

# Resolve URDF/XML from multiple possible project roots
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_ROOTS = [
    os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir)),       # code/
    os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir)),  # manipulator/
    os.getcwd(),
]
_URDF, _XML = None, None
for root in _CANDIDATE_ROOTS:
    u = os.path.join(root, "ewalker_description/urdf/ewalker.urdf")
    x = os.path.join(root, "models/ewalker_scene.xml")
    if os.path.exists(u) and os.path.exists(x):
        _URDF, _XML = u, x
        break
if _URDF is None:
    print("[eval_kp] ERROR: cannot find ewalker URDF/XML. Checked:", file=sys.stderr)
    for root in _CANDIDATE_ROOTS:
        print(f"  {root}", file=sys.stderr)
    sys.exit(1)

def main():
    scene_path = sys.argv[1] if len(sys.argv) > 1 else "results/ewalker_scenes/curriculum/train.json"
    with open(scene_path) as f:
        scenes = json.load(f)

    max_obs = max(len(s["obstacles"]) for s in scenes)
    env = ManipulatorEnv(urdf_path=_URDF, xml_path=_XML, n_joints=7,
                         dt=0.02, episode_len=500, n_obstacles=max_obs,
                         controller="rl", path_deadzone=0.10)
    action = np.zeros(env.act_dim)

    results = {"success": 0, "collision": 0, "timeout": 0, "ik_fail": 0,
               "total": len(scenes), "details": []}

    for i, scene in enumerate(scenes):
        sid = scene["scene_id"]
        start = np.array(scene["start"])
        goal = np.array(scene["goal"])

        # Apply scene
        env.reset()
        env.x_start = start.copy()
        env.x_goal = goal.copy()
        env.x_d = start.copy()

        # IK for start_q
        env.q[:] = np.array(scene["start_q"])
        env.dq[:] = 0.0
        if env.mj_data is not None:
            env.mj_data.qpos[:7] = env.q
            env.mj_data.qvel[:7] = 0.0
            import mujoco
            mujoco.mj_forward(env.mj_model, env.mj_data)

        # Setup obstacles
        centers = [np.array(o[:3]) for o in scene["obstacles"]]
        radii = [o[3] for o in scene["obstacles"]]
        env.sdf.set_static_obstacles(centers, radii)
        if env.mj_model is not None:
            env._sync_obstacles_to_mujoco()

        # Run episode
        done = False
        collided = False
        steps = 0
        while not done and steps < 500:
            obs, reward, done, info = env.step(action)
            collided = collided or info.get("collision", False)
            steps += 1

        path_complete = env.path_param >= 0.99
        success = path_complete and not collided

        if success:
            results["success"] += 1
            reason = "success"
        elif collided:
            results["collision"] += 1
            reason = "collision"
        else:
            results["timeout"] += 1
            reason = "timeout"

        results["details"].append({
            "scene_id": sid, "success": success,
            "collision": collided, "steps": steps,
            "path_param": float(env.path_param),
            "tracking_error": float(info.get("tracking_error", 0)),
            "d_obs": float(info.get("d_obs", 0)),
            "reason": reason,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(scenes)}] success={results['success']} coll={results['collision']} timeout={results['timeout']}")

    print(f"\n{'='*60}")
    print(f"KP 评测结果 ({scene_path})")
    print(f"{'='*60}")
    print(f"总场景:     {results['total']}")
    print(f"成功率:     {results['success']} / {results['total']} ({100*results['success']/results['total']:.1f}%)")
    print(f"碰撞:       {results['collision']} / {results['total']} ({100*results['collision']/results['total']:.1f}%)")
    print(f"超时:       {results['timeout']} / {results['total']} ({100*results['timeout']/results['total']:.1f}%)")

    # Full body vs confined
    wb = [d for d in results["details"] if "whole_body" in d["scene_id"]]
    cs = [d for d in results["details"] if "confined_space" in d["scene_id"]]
    for label, grp in [("whole_body", wb), ("confined_space", cs)]:
        if grp:
            suc = sum(1 for d in grp if d["success"])
            col = sum(1 for d in grp if d["collision"])
            to = sum(1 for d in grp if not d["success"] and not d["collision"])
            print(f"\n  {label}: {suc}/{len(grp)} success, {col} coll, {to} timeout")

    # Save details
    out_path = scene_path.replace(".json", "_kp_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n详细结果已保存: {out_path}")

if __name__ == "__main__":
    main()
