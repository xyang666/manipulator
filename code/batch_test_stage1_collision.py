"""
Quick KP test on stage1 scenes to check collision types.
"""
import sys, os, json, numpy as np, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

SCENE_JSON = os.path.join(_ROOT, "results/challenge_stage1.json")
URDF = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                    "/share/example-robot-data/robots/panda_description/urdf/panda.urdf")
XML = os.path.join(_ROOT, "models/panda_scene.xml")

with open(SCENE_JSON) as f:
    SCENES = json.load(f)

def run_kp_with_debug(scene, max_steps=1000):
    from test import apply_scene
    from env.manipulator_env import ManipulatorEnv

    n_obs = len(scene["obstacles"])
    env = ManipulatorEnv(
        urdf_path=URDF, xml_path=XML, n_joints=7,
        dt=0.02, episode_len=max_steps,
        n_obstacles=n_obs, obs_radius=0.02,
        controller="rl",
    )
    env.reset()
    apply_scene(env, scene)

    n_obs_coll = 0
    n_self_coll = 0
    tracking_errors = []

    for step in range(max_steps):
        action = np.zeros(env.act_dim)
        obs, reward, done, info = env.step(action)
        x_ee, _ = env.kin.forward_kinematics(env.q)
        tracking_errors.append(np.linalg.norm(x_ee - env.x_d))
        if info.get("collision", False):
            n_obs_coll += info.get("n_obstacle_contacts", 0) > 0
            n_self_coll += info.get("n_self_contacts", 0) > 0
        if done:
            break

    x_ee_final, _ = env.kin.forward_kinematics(env.q)
    final_dist = np.linalg.norm(x_ee_final - np.array(env.x_goal))

    return {
        "success": info.get("success", False),
        "n_obs_coll": n_obs_coll,
        "n_self_coll": n_self_coll,
        "any_coll": n_obs_coll > 0 or n_self_coll > 0,
        "final_distance": float(final_dist),
        "mean_track_error": float(np.mean(tracking_errors)),
    }


for sid, scene in enumerate(SCENES):
    n_obs = len(scene["obstacles"])
    t0 = time.time()
    r = run_kp_with_debug(scene)
    elapsed = time.time() - t0

    flag = "✓" if r["success"] else "✗"
    coll_type = ""
    if r["n_obs_coll"] > 0 and r["n_self_coll"] > 0:
        coll_type = " BOTH(obs+self)"
    elif r["n_obs_coll"] > 0:
        coll_type = f" OBS({r['n_obs_coll']})"
    elif r["n_self_coll"] > 0:
        coll_type = f" SELF({r['n_self_coll']})"
    else:
        coll_type = " clear"

    print(f"  scene {sid:2d} ({n_obs}obs): {flag}"
          f" final={r['final_distance']:.3f}{coll_type}"
          f"  ({elapsed:.1f}s)")