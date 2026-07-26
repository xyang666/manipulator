"""Generate scene sets matching the phase-one experiments in the paper."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import mujoco

from trajectory.generator import TrajectoryGenerator
from utils.collision import CollisionDetector


def fingerprint(scene):
    payload = {key: scene[key] for key in ("start", "goal", "obstacles")}
    payload["trajectory"] = scene.get("trajectory", {"type": "linear"})
    payload["start_q"] = scene.get("start_q")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def generate_obstacle_set(generator, count, n_obstacles, prefix, used):
    scenes = []
    while len(scenes) < count:
        require_nontrivial = len(scenes) < count // 2
        scene = generator.generate_scene(
            len(scenes), n_obstacles, max_attempts=100, ahead_mode=True,
            require_nontrivial=require_nontrivial,
            oracle_waypoints=11, oracle_candidates=8,
        )
        if scene is None or scene["scene_fingerprint"] in used:
            continue
        used.add(scene["scene_fingerprint"])
        scene["scene_id"] = f"{prefix}-{len(scenes):05d}"
        scene["scenario"] = prefix.rsplit("-", 1)[0]
        scenes.append(scene)
        if len(scenes) % 10 == 0:
            print(f"[{prefix}] {len(scenes)}/{count}", flush=True)
    return scenes


def generate_free_space(generator, count, used):
    center = np.array([0.4, 0.0, 0.4])
    scenes = []
    while len(scenes) < count:
        q = generator.kin.inverse_kinematics(
            center, q_init=np.random.uniform(generator.q_min, generator.q_max)
        )
        if q is None or not generator._configuration_valid(q, [], 0.025):
            continue
        positions = [center + np.array([0.0, 0.15 * np.sin(phase),
                                        0.1 * np.sin(2.0 * phase)])
                     for phase in np.linspace(0.0, 2.0 * np.pi, 41)]
        q_path = [q]
        valid = True
        for position in positions[1:]:
            q_next = generator.kin.inverse_kinematics(position, q_init=q_path[-1])
            if q_next is None or not generator._edge_valid(
                    q_path[-1], q_next, [], x1=positions[len(q_path) - 1],
                    x2=position):
                valid = False
                break
            q_path.append(q_next)
        if not valid:
            continue
        scene = {
            "scene_id": f"free_space-test-{len(scenes):05d}",
            "scenario": "free_space", "start": center.tolist(),
            "goal": center.tolist(), "start_q": q.tolist(), "obstacles": [],
            "trajectory": {"type": "figure_eight", "center": center.tolist(),
                           "y_amplitude": 0.15, "z_amplitude": 0.1,
                           "period": 4.0},
            "feasible": True, "oracle": "figure_eight_ik_continuation",
            "oracle_waypoints": len(positions),
            "feasible_q_path": [item.tolist() for item in q_path],
        }
        scene["scene_fingerprint"] = fingerprint(scene)
        if scene["scene_fingerprint"] in used:
            continue
        used.add(scene["scene_fingerprint"])
        scenes.append(scene)
    return scenes


def generate_corridors(generator, count, prefix, used):
    scenes = []
    home = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
    while len(scenes) < count:
        x = np.random.uniform(0.38, 0.42)
        z = np.random.uniform(0.37, 0.43)
        half_span = np.random.uniform(0.14, 0.16)
        radius = np.random.uniform(0.025, 0.032)
        free_width = np.random.uniform(0.24, 0.28)
        offset = free_width / 2.0 + radius
        start = np.array([x, -half_span, z])
        goal = np.array([x, half_span, z])
        q0 = generator.kin.inverse_kinematics(start, q_init=home)
        if q0 is None:
            continue
        q1 = generator.kin.inverse_kinematics(goal, q_init=q0)
        if q1 is None:
            continue
        obstacles = []
        for y in np.linspace(-0.9 * half_span, 0.9 * half_span, 5):
            obstacles.extend([[x - offset, y, z, radius],
                              [x + offset, y, z, radius]])
        oracle = generator._task_path_ik_oracle(
            start, goal, q0, q1, obstacles, waypoints=11, candidates=16
        )
        if oracle is None:
            continue
        nominal = generator._nominal_path_evidence(q0, q1, obstacles)
        scene = {
            "scene_id": f"{prefix}-{len(scenes):05d}", "scenario": "confined_space",
            "start": start.tolist(), "goal": goal.tolist(),
            "start_q": q0.tolist(), "goal_q": q1.tolist(),
            "obstacles": obstacles, "corridor_free_width": float(free_width),
            "feasible": True, "nontrivial": bool(nominal["collision"] or
                                                   nominal["min_clearance"] < 0.08),
            "nominal_collision": nominal["collision"],
            "nominal_min_clearance": nominal["min_clearance"], **oracle,
        }
        scene["scene_fingerprint"] = fingerprint(scene)
        if scene["scene_fingerprint"] in used:
            continue
        used.add(scene["scene_fingerprint"])
        scenes.append(scene)
        if len(scenes) % 10 == 0:
            print(f"[{prefix}] {len(scenes)}/{count}", flush=True)
    return scenes


def write(path, scenes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scenes, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/paper_scenes")
    parser.add_argument("--train", type=int, default=60)
    parser.add_argument("--validation", type=int, default=20)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root / output
    np.random.seed(args.seed)
    generator = TrajectoryGenerator(
        str(root / "panda_description/urdf/panda.urdf"),
        obstacle_radius_range=(0.025, 0.055),
    )
    model = mujoco.MjModel.from_xml_path(str(root / "models/panda_scene.xml"))
    generator.collision_detector = CollisionDetector(model, mujoco.MjData(model))
    used = set()

    write(output / "free_space/test.json",
          generate_free_space(generator, args.test, used))
    whole_train = generate_obstacle_set(generator, args.train, 3,
                                        "whole_body-train", used)
    whole_validation = generate_obstacle_set(generator, args.validation, 3,
                                             "whole_body-validation", used)
    write(output / "whole_body/train.json", whole_train)
    write(output / "whole_body/validation.json", whole_validation)
    write(output / "whole_body/test.json",
          generate_obstacle_set(generator, args.test, 3, "whole_body-test", used))
    confined_train = generate_corridors(generator, args.train,
                                        "confined_space-train", used)
    confined_validation = generate_corridors(generator, args.validation,
                                             "confined_space-validation", used)
    write(output / "confined_space/train.json", confined_train)
    write(output / "confined_space/validation.json", confined_validation)
    write(output / "confined_space/test.json",
          generate_corridors(generator, args.test, "confined_space-test", used))
    write(output / "generalization/train.json",
          generate_obstacle_set(generator, args.train, 1,
                                "generalization-train", used))
    write(output / "generalization/validation.json",
          generate_obstacle_set(generator, args.validation, 1,
                                "generalization-validation", used))
    write(output / "generalization/test.json",
          generate_obstacle_set(generator, args.test, 3,
                                "generalization-test", used))
    write(output / "curriculum/train.json", whole_train + confined_train)
    write(output / "curriculum/validation.json",
          whole_validation + confined_validation)
    manifest = {"schema_version": 1, "seed": args.seed,
                "train_count_per_scenario": args.train,
                "validation_count_per_scenario": args.validation,
                "test_count_per_scenario": args.test,
                "global_unique_scenes": len(used),
                "protocol": {"free_space": "figure-eight, no obstacles",
                             "whole_body": "linear scan, 3 spheres",
                             "confined_space": "linear scan, 10-sphere corridor",
                             "generalization": "train 1 sphere, test 3 spheres"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
