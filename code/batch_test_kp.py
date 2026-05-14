"""
Batch test: run KP on all challenge_close scenes to verify they cause collisions.
"""
import sys, os, json, numpy as np, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

SCENE_JSON = os.path.join(_ROOT, "results/challenge_close.json")
URDF = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                    "/share/example-robot-data/robots/panda_description/urdf/panda.urdf")
XML = os.path.join(_ROOT, "models/panda_scene.xml")

with open(SCENE_JSON) as f:
    SCENES = json.load(f)
print(f"Loaded {len(SCENES)} scenes from {SCENE_JSON}")


def run_kp_scene(scene, max_steps=1000):
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
    env.reward_fn.w_collision = 100.0

    apply_scene(env, scene)

    tracking_errors = []
    obstacle_distances = []
    n_collisions = 0
    success = False

    for step in range(max_steps):
        action = np.zeros(env.act_dim)
        obs, reward, done, info = env.step(action)
        x_ee, _ = env.kin.forward_kinematics(env.q)
        track_err = np.linalg.norm(x_ee - env.x_d)
        tracking_errors.append(track_err)
        obstacle_distances.append(info.get("d_obs", 0))
        if info.get("collision", False):
            n_collisions += 1
        if done:
            if info.get("success", False):
                success = True
            break

    x_ee_final, _ = env.kin.forward_kinematics(env.q)
    final_dist = np.linalg.norm(x_ee_final - np.array(env.x_goal))
    return {
        "success": success,
        "collisions": n_collisions,
        "final_distance": float(final_dist),
        "mean_track_error": float(np.mean(tracking_errors)),
        "min_d_obs": float(np.min(obstacle_distances)),
        "steps": len(tracking_errors),
    }


# Run KP on all scenes
results = []
for sid, scene in enumerate(SCENES):
    n_obs = len(scene["obstacles"])
    t0 = time.time()
    r = run_kp_scene(scene)
    elapsed = time.time() - t0
    results.append(r)

    flag = "✓" if r["success"] else "✗"
    coll = f" {r['collisions']} COLL" if r["collisions"] > 0 else " clear"
    print(f"  KP scene {sid:2d} ({n_obs}obs): {flag}"
          f" final={r['final_distance']:.3f} d_obs_min={r['min_d_obs']:.3f}{coll}"
          f"  ({elapsed:.1f}s)")

# Summary
n_success = sum(1 for r in results if r["success"])
n_coll = sum(1 for r in results if r["collisions"] > 0)
print(f"\n{'='*50}")
print(f"KP results on {len(results)} challenge_close scenes")
print(f"{'='*50}")
print(f"  Success:  {n_success}/{len(results)} ({n_success/len(results)*100:.0f}%)")
print(f"  Collided: {n_coll}/{len(results)} ({n_coll/len(results)*100:.0f}%)")
print(f"  Mean min d_obs: {np.mean([r['min_d_obs'] for r in results]):.4f}")
print(f"  Mean final dist: {np.mean([r['final_distance'] for r in results]):.4f}")
print(f"  Mean track error: {np.mean([r['mean_track_error'] for r in results]):.4f}")

# Per-group stats
for n_obs in [1, 2, 3, 4, 5]:
    group = [r for s, r in zip(SCENES, results) if len(s["obstacles"]) == n_obs]
    gc = sum(1 for r in group if r["collisions"] > 0)
    print(f"  [{n_obs}obs] {len(group)} scenes: {gc} collided ({gc/len(group)*100:.0f}%)")
