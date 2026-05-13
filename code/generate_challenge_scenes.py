"""
Generate challenge scenes: small obstacles very close to trajectory path.
Designed to test whether SAC can navigate tight spaces where KP (pure tracking) fails.
"""
import sys, os, json, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from trajectory.generator import TrajectoryGenerator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_VENV_DATA = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                          "/share/example-robot-data/robots/panda_description")
URDF = os.path.join(_VENV_DATA, "urdf/panda.urdf")

# ------------------------------------------------------------
# Scene configuration
# ------------------------------------------------------------
# Key parameters:
#   obstacle_radius_range=(0.01, 0.04)  — small 1-4cm obstacles
#   min_obstacle_distance=0.005         — as close as 5mm from path line
#
# Obstacles are placed at offset_distance in [min_clearance + radius + 0.02, 0.3]
# So with radius 1cm: min offset = 0.5cm + 1cm + 2cm = 3.5cm from path
# With radius 4cm: min offset = 0.5cm + 4cm + 2cm = 6.5cm from path
# Capsule SDF radius ~3cm, so at min offset the gap is 0.5-3.5cm → triggers sigma
# ------------------------------------------------------------

np.random.seed(123)

# Generate challenge scenes: small obstacles, close to path
gen = TrajectoryGenerator(
    urdf_path=URDF,
    n_joints=7,
    obstacle_radius_range=(0.01, 0.04),
    min_obstacle_distance=0.005,
)

# Generate a set of scenes with varying obstacle count
# Scene 0-24: 1 obstacle each (mild)
# Scene 25-49: 2 obstacles each (moderate)
# Scene 50-74: 3 obstacles each (hard)
n_each = 25
configs = [(1, n_each), (2, n_each), (3, n_each)]

scenes = []
failed = 0
total_target = sum(n for _, n in configs)

for n_obs, n_target in configs:
    sid = 0
    while len([s for s in scenes if len(s["obstacles"]) == n_obs]) < n_target and sid < n_target * 10:
        scene = gen.generate_scene(
            scene_id=sid,
            n_obstacles=n_obs,
            max_attempts=500,
            ahead_mode=True,  # Y-parallel lines in front of robot
        )
        if scene is not None:
            scene["scene_id"] = len(scenes)
            scene["difficulty"] = f"{n_obs}obs"
            scene["min_clearance"] = 0.005
            scenes.append(scene)

            # Print geometry info
            start = np.array(scene["start"])
            goal = np.array(scene["goal"])
            d_vec = goal - start
            print(f"  scene {scene['scene_id']:2d} ({n_obs}obs): ", end="")
            for o in scene["obstacles"]:
                obs_c = np.array(o[:3])
                t = np.dot(obs_c - start, d_vec) / (np.dot(d_vec, d_vec) + 1e-8)
                t = np.clip(t, 0, 1)
                closest = start + t * d_vec
                dist = np.linalg.norm(obs_c - closest)
                print(f"[d={dist:.3f}, r={o[3]:.3f}] ", end="")
            print()
        else:
            failed += 1
        sid += 1

# Save all scenes
output = os.path.join(_ROOT, "results/challenge_scenes.json")
with open(output, "w") as f:
    json.dump(scenes, f, indent=2)
print(f"\n→ Saved {len(scenes)} scenes to {output} (failed: {failed})")

# Summary statistics
print(f"\nScene statistics:")
for n_obs in [1, 2, 3]:
    group = [s for s in scenes if len(s["obstacles"]) == n_obs]
    print(f"  {n_obs} obstacle(s): {len(group)} scenes")

# Print full geometry for all scenes
print(f"\nFull scene geometry:")
for scene in scenes:
    start = np.array(scene["start"])
    goal = np.array(scene["goal"])
    d_vec = goal - start
    path_len = np.linalg.norm(d_vec)
    print(f"  scene {scene['scene_id']:2d} ({len(scene['obstacles'])}obs, "
          f"path={path_len:.3f}m):")
    for o in scene["obstacles"]:
        obs_c = np.array(o[:3])
        t = np.dot(obs_c - start, d_vec) / (np.dot(d_vec, d_vec) + 1e-8)
        t = np.clip(t, 0, 1)
        closest = start + t * d_vec
        dist = np.linalg.norm(obs_c - closest)
        print(f"    obs at ({o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}) "
              f"r={o[3]:.3f}  dist_to_path={dist:.3f}")