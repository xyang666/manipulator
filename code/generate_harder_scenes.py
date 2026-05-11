"""
Generate challenging scenes with obstacles close to the trajectory path.
Uses the TrajectoryGenerator with reduced min_clearance so obstacles
are close enough to activate the sigma gate (d_obs < d_critical).
"""
import sys, os, json, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from trajectory.generator import TrajectoryGenerator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_VENV_DATA = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                          "/share/example-robot-data/robots/panda_description")
URDF = os.path.join(_VENV_DATA, "urdf/panda.urdf")

# Scene configurations
SCENE_CONFIGS = [
    # (name, n_scenes, n_obs, min_clearance, ahead_mode)
    ("medium", 20, 1, 0.015, False),   # obstacles very close to path → strong sigma activation
    ("hard",   10, 2, 0.005, True),     # multiple obstacles, near-intersecting
]

np.random.seed(42)

for name, n_scenes, n_obs, min_clearance, ahead_mode in SCENE_CONFIGS:
    print(f"\n{'='*60}")
    print(f"Generating {n_scenes} '{name}' scenes: {n_obs} obs, clearance={min_clearance}")
    print(f"{'='*60}")

    gen = TrajectoryGenerator(
        urdf_path=URDF,
        n_joints=7,
        min_obstacle_distance=min_clearance,  # <-- KEY: reduce clearance
    )

    scenes = []
    failed = 0
    sid = 0
    while len(scenes) < n_scenes and sid < n_scenes * 5:
        scene = gen.generate_scene(
            scene_id=sid,
            n_obstacles=n_obs,
            max_attempts=500,
            ahead_mode=ahead_mode,
        )
        if scene is not None:
            # Renumber sequentially
            scene["scene_id"] = len(scenes)
            scene["difficulty"] = name
            scene["min_clearance"] = min_clearance
            scenes.append(scene)
            print(f"  scene {scene['scene_id']:2d}: obs_dist=", end="")
            for o in scene["obstacles"]:
                # Compute distance from obstacle to path line
                start = np.array(scene["start"])
                goal = np.array(scene["goal"])
                obs_c = np.array(o[:3])
                d = goal - start
                t = np.dot(obs_c - start, d) / (np.dot(d, d) + 1e-8)
                t = np.clip(t, 0, 1)
                closest = start + t * d
                dist = np.linalg.norm(obs_c - closest)
                print(f"[d={dist:.3f}, r={o[3]:.3f}] ", end="")
            print()
        else:
            failed += 1
        sid += 1

    # Save
    output = os.path.join(_ROOT, f"results/train_scenes_{name}.json")
    with open(output, "w") as f:
        json.dump(scenes, f, indent=2)
    print(f"  → Saved {len(scenes)} scenes to {output} (failed: {failed})")

# Also create a validation set from medium scenes
print(f"\n{'='*60}")
print("Creating validation split (last 5 medium scenes as validation)")
print(f"{'='*60}")

medium_path = os.path.join(_ROOT, "results/train_scenes_medium.json")
with open(medium_path) as f:
    medium = json.load(f)

n_val = min(5, len(medium) // 3)
val = medium[:n_val]
train = medium[n_val:]

# Save validation set
val_path = os.path.join(_ROOT, "results/val_scenes_medium.json")
with open(val_path, "w") as f:
    json.dump(val, f, indent=2)

# Save remaining as training
train_path = os.path.join(_ROOT, "results/train_scenes_medium.json")
with open(train_path, "w") as f:
    json.dump(train, f, indent=2)

print(f"  Train: {len(train)} scenes → {train_path}")
print(f"  Val:   {len(val)} scenes → {val_path}")

# Print obstacle geometry for validation scenes
print(f"\nValidation scene obstacle geometry:")
for s in val:
    start = np.array(s["start"])
    goal = np.array(s["goal"])
    d_vec = goal - start
    path_len = np.linalg.norm(d_vec)
    for o in s["obstacles"]:
        obs_c = np.array(o[:3])
        t = np.dot(obs_c - start, d_vec) / (np.dot(d_vec, d_vec) + 1e-8)
        t = np.clip(t, 0, 1)
        closest = start + t * d_vec
        dist = np.linalg.norm(obs_c - closest)
        print(f"  scene {s['scene_id']:2d}: obs at ({o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}) "
              f"r={o[3]:.3f}  dist_to_path={dist:.3f}  start=({s['start'][0]:.2f}, {s['start'][1]:.2f}, {s['start'][2]:.2f}) "
              f"goal=({s['goal'][0]:.2f}, {s['goal'][1]:.2f}, {s['goal'][2]:.2f})")