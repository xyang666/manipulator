#!/usr/bin/env python3
"""Quick feasibility test for corridor widths on E-Walker."""
import sys, numpy as np
sys.path.insert(0, "code")
from trajectory.generator import TrajectoryGenerator
from env.kinematics import ManipulatorKinematics
import mujoco
from utils.collision import CollisionDetector

kin = ManipulatorKinematics("ewalker_description/urdf/ewalker.urdf")
gen = TrajectoryGenerator("ewalker_description/urdf/ewalker.urdf",
                          obstacle_radius_range=(0.025, 0.055))
model = mujoco.MjModel.from_xml_path("models/ewalker_scene.xml")
gen.collision_detector = CollisionDetector(model, mujoco.MjData(model))

home = np.zeros(7)
rng = np.random.default_rng(42)

print("=== Corridor width feasibility ===")
print(f'{"Width":>8} {"IK_OK":>6} {"Oracle_OK":>10}')
print("-" * 30)

for free_width in [0.44, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28, 0.26]:
    ik_ok = 0
    oracle_ok = 0
    attempts = 20  # fewer tries for speed
    for _ in range(attempts):
        x = rng.uniform(0.38, 0.42)
        z = rng.uniform(0.37, 0.43)
        half_span = rng.uniform(0.14, 0.16)
        radius = rng.uniform(0.025, 0.032)
        offset = free_width / 2.0 + radius
        start = np.array([x, -half_span, z])
        goal = np.array([x, half_span, z])
        q0 = kin.inverse_kinematics(start, q_init=home)
        if q0 is None:
            continue
        q1 = kin.inverse_kinematics(goal, q_init=q0)
        if q1 is None:
            continue
        ik_ok += 1
        obstacles = []
        for y in np.linspace(-0.9 * half_span, 0.9 * half_span, 4):
            obstacles.extend([[x - offset, y, z, radius],
                              [x + offset, y, z, radius]])
        obstacles.append([x - offset, 0.0, z + 0.12, radius])
        oracle = gen._task_path_ik_oracle(
            start, goal, q0, q1, obstacles, waypoints=11, candidates=16)
        if oracle is not None:
            oracle_ok += 1
    print(f"{free_width:>8.2f} {ik_ok:>6} {oracle_ok:>10}")
