"""
Generate challenge scenes with obstacles VERY close to the trajectory path.
Obstacles are placed 2-8cm from the path line (center-to-line),
with small radius (1-2cm). Uses ahead_mode for valid, reachable trajectories.

For SAC to outperform KP, obstacles must be close enough that KP's pure
tracking (which doesn't avoid obstacles) collides, while SAC's learned
avoidance navigates around them.
"""
import sys, os, json, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from trajectory.generator import TrajectoryGenerator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_VENV_DATA = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                          "/share/example-robot-data/robots/panda_description")
URDF = os.path.join(_VENV_DATA, "urdf/panda.urdf")

np.random.seed(789)

# Generator with small obstacles and tight clearance
gen = TrajectoryGenerator(
    urdf_path=URDF,
    n_joints=7,
    obstacle_radius_range=(0.01, 0.02),
    min_obstacle_distance=0.002,
)

# Generate scenes: 1-5 obstacles, 20 each = 100 total
n_each = 20
max_obs_dist = 0.08   # obstacles up to 8cm center-to-line
scenes = []

for n_obs in [1, 2, 3, 4, 5]:
    sid = 0
    while len([s for s in scenes if len(s["obstacles"]) == n_obs]) < n_each and sid < 500:
        scene = gen.generate_scene(
            scene_id=len(scenes),
            n_obstacles=n_obs,
            max_attempts=500,
            ahead_mode=True,
            max_obstacle_distance=max_obs_dist,
        )
        if scene is None:
            sid += 1
            continue

        # Validate min obstacle-to-path distance
        start = np.array(scene["start"])
        goal = np.array(scene["goal"])
        d_vec = goal - start
        traj_len = np.linalg.norm(d_vec)
        min_surf_dist = float("inf")
        for o in scene["obstacles"]:
            oc = np.array(o[:3])
            t = np.dot(oc - start, d_vec) / (np.dot(d_vec, d_vec) + 1e-8)
            t = np.clip(t, 0, 1)
            closest = start + t * d_vec
            d_path = np.linalg.norm(oc - closest)
            surf_to_line = d_path - o[3]
            min_surf_dist = min(min_surf_dist, surf_to_line)

        # The path line must clear all obstacles
        if min_surf_dist < 0.0:
            sid += 1
            continue

        scene["difficulty"] = f"{n_obs}obs_close"
        scene["min_clearance"] = 0.002
        scenes.append(scene)

        # Print geometry
        print(f"  scene {scene['scene_id']:2d} ({n_obs}obs): ", end="")
        for o in scene["obstacles"]:
            oc = np.array(o[:3])
            t = np.dot(oc - start, d_vec) / (np.dot(d_vec, d_vec) + 1e-8)
            t = np.clip(t, 0, 1)
            closest = start + t * d_vec
            d_path = np.linalg.norm(oc - closest)
            clearance_path = d_path - o[3]
            print(f"[d_center={d_path:.3f}, r={o[3]:.3f}, clr={clearance_path:.3f}] ", end="")
        print()

        sid += 1

# Save
output = os.path.join(_ROOT, "results/challenge_close.json")
with open(output, "w") as f:
    json.dump(scenes, f, indent=2)
print(f"\nSaved {len(scenes)} scenes to {output}")

# Summary
print(f"\nScene statistics:")
for n_obs in [1, 2, 3, 4, 5]:
    group = [s for s in scenes if len(s["obstacles"]) == n_obs]
    print(f"  {n_obs} obstacle(s): {len(group)} scenes")

if scenes:
    all_surf = []
    for s in scenes:
        start = np.array(s["start"])
        goal = np.array(s["goal"])
        d_vec = goal - start
        for o in s["obstacles"]:
            oc = np.array(o[:3])
            t = np.dot(oc - start, d_vec) / (np.dot(d_vec, d_vec) + 1e-8)
            t = np.clip(t, 0, 1)
            closest = start + t * d_vec
            all_surf.append(np.linalg.norm(oc - closest) - o[3])
    print(f"  Surface-to-pathline: min={np.min(all_surf):.3f}, "
          f"mean={np.mean(all_surf):.3f}, max={np.max(all_surf):.3f}")
