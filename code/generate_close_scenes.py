"""
Generate challenge scenes with obstacles VERY close to the trajectory path.
Obstacles are placed 2-8cm from the path line (center-to-line),
with small radius (1-2cm). This ensures KP baseline collides on most scenes
and SAC must perform real obstacle avoidance.
"""
import sys, os, json, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from trajectory.generator import TrajectoryGenerator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_VENV_DATA = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                          "/share/example-robot-data/robots/panda_description")
URDF = os.path.join(_VENV_DATA, "urdf/panda.urdf")


def line_sphere_check(p1, p2, center, radius, clearance=0.0):
    """Check if line segment p1->p2 comes within radius+clearance of center."""
    d = p2 - p1
    f = p1 - center
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - (radius + clearance) ** 2
    disc = b**2 - 4*a*c
    if disc < 0:
        return False
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)


def place_obstacles_near_path(start_pos, goal_pos, n_obstacles, max_attempts=500):
    """
    Place obstacles very close to the path: 2-8cm center-to-line.
    Obstacle radius: 0.01-0.02m (1-2cm).
    Ensures path line itself doesn't intersect obstacles (with 2mm clearance).
    """
    traj_dir = goal_pos - start_pos
    traj_len = np.linalg.norm(traj_dir)
    if traj_len < 1e-6:
        return None
    traj_dir_u = traj_dir / traj_len

    obstacles = []
    min_offset = 0.02  # 2cm minimum center-to-line (ensures path clears)
    max_offset = 0.08  # 8cm maximum center-to-line

    for _ in range(n_obstacles):
        placed = False
        for attempt in range(max_attempts):
            radius = np.random.uniform(0.01, 0.02)
            t = np.random.uniform(0.15, 0.85)
            point_on_traj = start_pos + t * traj_dir

            # Sample offset distance: biased towards closer values
            offset = min_offset + (max_offset - min_offset) * np.random.beta(1.5, 2.0)

            # Random perpendicular direction
            perp = np.random.randn(3)
            perp -= np.dot(perp, traj_dir_u) * traj_dir_u
            perp_norm = np.linalg.norm(perp)
            if perp_norm < 1e-6:
                continue
            perp /= perp_norm

            pos = point_on_traj + offset * perp

            # Z constraint: stay above floor
            pos[2] = max(pos[2], 0.02 + radius)

            # Verify path line doesn't intersect obstacle (2mm clearance)
            if line_sphere_check(start_pos, goal_pos, pos, radius, clearance=0.002):
                continue

            # Check overlap with existing obstacles
            overlap = False
            for obs in obstacles:
                dist = np.linalg.norm(pos - obs[:3])
                if dist < (radius + obs[3] + 0.01):
                    overlap = True
                    break
            if overlap:
                continue

            obstacles.append(np.array([pos[0], pos[1], pos[2], radius]))
            placed = True
            break
        if not placed:
            return None
    return obstacles


np.random.seed(456)

# Use TrajectoryGenerator for start/goal sampling
gen = TrajectoryGenerator(
    urdf_path=URDF,
    n_joints=7,
    obstacle_radius_range=(0.01, 0.02),
    min_obstacle_distance=0.002,
)

# Generate scenes: 1, 2, 3 obstacles, 15 each
n_each = 15
scenes = []
failed = 0

for n_obs in [1, 2, 3]:
    sid = 0
    while len([s for s in scenes if len(s["obstacles"]) == n_obs]) < n_each and sid < 200:
        # Get a valid start/goal from the generator
        start_pos, start_q = gen.sample_reachable_point()
        goal_pos, goal_q = gen.sample_reachable_point()

        if start_pos[2] < 0.02 or goal_pos[2] < 0.02:
            sid += 1
            continue
        dist = np.linalg.norm(goal_pos - start_pos)
        if dist < 0.3 or dist > 1.2:
            sid += 1
            continue

        # Place obstacles very close to path
        obstacles = place_obstacles_near_path(start_pos, goal_pos, n_obs)
        if obstacles is None:
            sid += 1
            continue

        # Check manipulability
        start_ik = gen.kin.inverse_kinematics(start_pos, q_init=start_q)
        goal_ik = gen.kin.inverse_kinematics(goal_pos, q_init=goal_q)
        if start_ik is None or goal_ik is None:
            sid += 1
            continue

        scene = {
            "scene_id": len(scenes),
            "start": start_pos.tolist(),
            "goal": goal_pos.tolist(),
            "start_q": start_ik.tolist(),
            "goal_q": goal_ik.tolist(),
            "obstacles": [o.tolist() for o in obstacles],
            "difficulty": f"{n_obs}obs_close",
            "min_clearance": 0.002,
        }
        scenes.append(scene)

        # Print geometry
        d_vec = goal_pos - start_pos
        print(f"  scene {scene['scene_id']:2d} ({n_obs}obs): ", end="")
        for o in scene["obstacles"]:
            obs_c = np.array(o[:3])
            t = np.dot(obs_c - start_pos, d_vec) / (np.dot(d_vec, d_vec) + 1e-8)
            t = np.clip(t, 0, 1)
            closest = start_pos + t * d_vec
            d_path = np.linalg.norm(obs_c - closest)
            clearance_path = d_path - o[3]  # obstacle surface to path line
            print(f"[d_center={d_path:.3f}, r={o[3]:.3f}, clr={clearance_path:.3f}] ", end="")
        print()

        sid += 1

# Save
output = os.path.join(_ROOT, "results/challenge_close.json")
with open(output, "w") as f:
    json.dump(scenes, f, indent=2)
print(f"\n→ Saved {len(scenes)} scenes to {output}")

# Summary
print(f"\nScene statistics:")
for n_obs in [1, 2, 3]:
    group = [s for s in scenes if len(s["obstacles"]) == n_obs]
    print(f"  {n_obs} obstacle(s): {len(group)} scenes")