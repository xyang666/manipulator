"""
fix_scenes.py
-------------
Fix scenes with self-collision at start/goal by re-running IK with MuJoCo
self-collision constraints. Keeps start/goal positions and obstacles unchanged.
"""
import sys, json, numpy as np

sys.path.insert(0, 'code/.venv/lib/python3.12/site-packages')
import mujoco
sys.path.insert(0, 'code')
from utils.collision import CollisionDetector
from env.kinematics import ManipulatorKinematics

# Joint limits (Panda default, matches generator)
q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# Init
kin = ManipulatorKinematics(urdf_path='panda_description/urdf/panda.urdf',
                             q_min=q_min, q_max=q_max)
model = mujoco.MjModel.from_xml_path('models/panda_scene.xml')
data = mujoco.MjData(model)
cd = CollisionDetector(model, data)

# Load all scenes
close = json.load(open('results/challenge_close.json'))

# Find bad scenes
bad_ids = []
for s in close:
    sid = s['scene_id']
    for name in ['start_q', 'goal_q']:
        q = np.array(s[name])
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        _, n_self = cd.detect_self_collisions()
        if n_self > 0:
            bad_ids.append(sid)
            break

bad_ids = sorted(set(bad_ids))
print(f"Found {len(bad_ids)} bad scenes: {bad_ids}")
print()

MAX_ATTEMPTS = 500
rng = np.random.RandomState(42)
fixed_count = 0

for sid in bad_ids:
    s = close[sid]
    start_pos = np.array(s['start'])
    goal_pos = np.array(s['goal'])

    print(f"Scene {sid}: fixing...", end=" ")

    # Find clean start_q
    new_start_q = None
    for att in range(MAX_ATTEMPTS):
        seed_q = rng.uniform(q_min, q_max)
        q_ik = kin.inverse_kinematics(start_pos, q_init=seed_q)
        if q_ik is None:
            continue
        if np.any(q_ik < q_min) or np.any(q_ik > q_max):
            continue
        data.qpos[:7] = q_ik
        mujoco.mj_forward(model, data)
        _, n_self = cd.detect_self_collisions()
        if n_self == 0:
            new_start_q = q_ik
            break

    # Find clean goal_q
    new_goal_q = None
    for att in range(MAX_ATTEMPTS):
        seed_q = rng.uniform(q_min, q_max)
        q_ik = kin.inverse_kinematics(goal_pos, q_init=seed_q)
        if q_ik is None:
            continue
        if np.any(q_ik < q_min) or np.any(q_ik > q_max):
            continue
        data.qpos[:7] = q_ik
        mujoco.mj_forward(model, data)
        _, n_self = cd.detect_self_collisions()
        if n_self == 0:
            new_goal_q = q_ik
            break

    if new_start_q is not None and new_goal_q is not None:
        s['start_q'] = new_start_q.tolist()
        s['goal_q'] = new_goal_q.tolist()
        print(f"OK (start={new_start_q[0]:.4f},..., goal={new_goal_q[0]:.4f},...)")
        fixed_count += 1
    elif new_start_q is not None:
        s['start_q'] = new_start_q.tolist()
        print(f"PARTIAL: start fixed, goal still self-collision")
    elif new_goal_q is not None:
        s['goal_q'] = new_goal_q.tolist()
        print(f"PARTIAL: goal fixed, start still self-collision")
    else:
        print(f"FAILED: no clean IK found in {MAX_ATTEMPTS} attempts")

print(f"\nFixed {fixed_count}/{len(bad_ids)} scenes")

# Save updated challenge_close.json
json.dump(close, open('results/challenge_close.json', 'w'), indent=2)
print("Saved challenge_close.json")

# Update subsets (stage1=0-39, stage2=0-59, stage3=0-99)
for name, count in [('challenge_stage1', 40), ('challenge_stage2', 60), ('challenge_stage3', 100)]:
    subset = json.load(open(f'results/{name}.json'))
    for i in range(min(count, len(close))):
        if close[i]['scene_id'] != subset[i]['scene_id']:
            print(f"WARNING: scene_id mismatch at {name}[{i}]")
        subset[i]['start_q'] = close[i]['start_q']
        subset[i]['goal_q'] = close[i]['goal_q']
    json.dump(subset, open(f'results/{name}.json', 'w'), indent=2)
    print(f"Updated {name}.json (scenes 0-{count-1})")

print("\nDone!")
