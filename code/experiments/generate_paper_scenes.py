"""Generate scene sets matching the phase-one experiments in the paper."""

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import mujoco

from env.manipulator_env import ManipulatorEnv
from experiment_config import ENVIRONMENT
from robot_config import DEFAULT_URDF, DEFAULT_XML
from trajectory.generator import TrajectoryGenerator
from utils.collision import CollisionDetector
from utils.validation import ValidationSet

_WORKER_GENERATOR = None
_WORKER_EVALUATOR = None


class PDEvaluator:
    """Run the exact zero-residual controller used by the PD audit."""

    def __init__(self, root, max_obstacles=10):
        self.env = ManipulatorEnv(
            urdf_path=DEFAULT_URDF,
            xml_path=DEFAULT_XML,
            n_obstacles=max_obstacles, use_trajectory_generator=False,
        )
        self.applier = ValidationSet.__new__(ValidationSet)

    def rollout(self, scene, record_path=False):
        self.applier.apply_scene_to_env(self.env, scene)
        q_path, collided, min_clearance = [], False, float("inf")
        info = {}
        for step in range(self.env.episode_len):
            if record_path:
                q_path.append(self.env.q.copy())
            _, _, done, info = self.env.step(np.zeros(self.env.act_dim))
            collided |= bool(info["collision"])
            min_clearance = min(min_clearance, float(info["d_obs"]))
            if done:
                break
        return {
            "success": bool(info.get("success", False)) and not collided,
            "collision": collided,
            "min_clearance": min_clearance,
            "steps": step + 1,
            "q_path": q_path,
        }


def fingerprint(scene):
    payload = {key: scene[key] for key in ("start", "goal", "obstacles")}
    payload["trajectory"] = scene.get("trajectory", {"type": "linear"})
    payload["start_q"] = scene.get("start_q")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _make_pd_blocker(generator, evaluator, base, n_obstacles, rng,
                     attempts=240):
    """Place clutter that blocks PD while retaining a certified IK path."""
    base_obstacles = list(base.get("obstacles", []))
    if len(base_obstacles) != n_obstacles - 1:
        raise ValueError("base scene must contain exactly n_obstacles - 1 spheres")
    pd_path = evaluator.rollout(base, record_path=True)
    if not pd_path["success"] or len(pd_path["q_path"]) < 20:
        return None
    q_path = pd_path["q_path"]
    capsule_motion = []
    for index in range(len(generator.kin.get_link_capsules(q_path[0]))):
        midpoints = []
        for q in q_path:
            p1, p2, _ = generator.kin.get_link_capsules(q)[index]
            midpoints.append((p1 + p2) / 2.0)
        points = np.asarray(midpoints)
        capsule_motion.append(float(np.max(np.linalg.norm(
            points - points[0], axis=1))))
    moving = [index for index, motion in enumerate(capsule_motion)
              if motion > 0.08]
    if not moving:
        return None

    for _ in range(attempts):
        path_index = int(rng.integers(2 * len(q_path) // 5,
                                     3 * len(q_path) // 5))
        q = q_path[path_index]
        capsule_index = int(rng.choice(moving))
        p1, p2, capsule_radius = \
            generator.kin.get_link_capsules(q)[capsule_index]
        axis = p2 - p1
        axis /= max(np.linalg.norm(axis), 1e-9)
        alpha = float(rng.uniform(0.15, 0.85))
        point = (1.0 - alpha) * p1 + alpha * p2
        radius = float(rng.uniform(0.012, 0.022))
        penetration = float(rng.uniform(0.001, 0.004))
        best = None
        for _ in range(32):
            direction = rng.normal(size=3)
            direction -= direction.dot(axis) * axis
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            center = point + direction / norm * (
                capsule_radius + radius - penetration)
            obstacle = [[*center.tolist(), radius]]
            endpoint_clearance = min(
                generator.arm_obstacle_clearance(
                    np.asarray(base["start_q"]), obstacle),
                generator.arm_obstacle_clearance(
                    np.asarray(base["goal_q"]), obstacle),
            )
            if best is None or endpoint_clearance > best[0]:
                best = endpoint_clearance, obstacle[0]
        if best is None or best[0] < 0.025:
            continue
        obstacles = base_obstacles + [best[1]]
        scene = dict(base)
        scene["obstacles"] = obstacles
        # PD failure is cheap to measure and is a necessary condition. Check it
        # before spending most of the generation time on the graph oracle.
        pd = evaluator.rollout(scene)
        if pd["success"]:
            continue
        oracle = generator._task_path_ik_oracle(
            np.asarray(base["start"]), np.asarray(base["goal"]),
            np.asarray(base["start_q"]), np.asarray(base["goal_q"]),
            obstacles, waypoints=11, candidates=16,
        )
        if oracle is None:
            continue
        scene.update(oracle)
        nominal = generator._nominal_path_evidence(
            np.asarray(base["start_q"]), np.asarray(base["goal_q"]), obstacles)
        scene.update({
            "nontrivial": True,
            "difficulty": "hard",
            "difficulty_definition": "zero_residual_pd_fails_or_collides",
            "pd_success": False,
            "pd_collision": pd["collision"],
            "pd_min_clearance": pd["min_clearance"],
            "pd_steps": pd["steps"],
            "blocker_capsule": capsule_index,
            "nominal_collision": nominal["collision"],
            "nominal_min_clearance": nominal["min_clearance"],
            "nominal_min_manipulability": nominal["min_manipulability"],
        })
        return scene
    return None


def generate_obstacle_set(generator, evaluator, count, n_obstacles, prefix,
                          used, rng):
    scenes = []
    while len(scenes) < count:
        base = generator.generate_scene(
            len(scenes), n_obstacles - 1, max_attempts=100, ahead_mode=True,
            require_nontrivial=False, oracle_waypoints=11,
            oracle_candidates=16,
        )
        scene = (None if base is None else _make_pd_blocker(
            generator, evaluator, base, n_obstacles, rng))
        if scene is None:
            continue
        scene["scene_fingerprint"] = fingerprint(scene)
        if scene["scene_fingerprint"] in used:
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


def generate_corridors(generator, evaluator, count, prefix, used, rng):
    scenes = []
    # Use the neutral pose of the loaded robot. The previous Panda-specific
    # seed biased IK after switching the experiment to E-Walker.
    home = np.zeros(generator.kin.n)
    while len(scenes) < count:
        x = np.random.uniform(0.38, 0.42)
        z = np.random.uniform(0.37, 0.43)
        half_span = np.random.uniform(0.14, 0.16)
        radius = np.random.uniform(0.025, 0.032)
        # E-Walker's capsule envelope is wider than the former Panda model.
        # A 0.24--0.28 m corridor collides even at the endpoint IK solutions;
        # 0.44--0.50 m retains a confined passage while allowing at least
        # 2.5 cm endpoint clearance before adding the PD-blocking obstacle.
        free_width = np.random.uniform(0.44, 0.50)
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
        for y in np.linspace(-0.9 * half_span, 0.9 * half_span, 4):
            obstacles.extend([[x - offset, y, z, radius],
                              [x + offset, y, z, radius]])
        # A ninth fixed clutter sphere leaves one slot for the PD blocker.
        obstacles.append([x - offset, 0.0, z + 0.12, radius])
        oracle = generator._task_path_ik_oracle(
            start, goal, q0, q1, obstacles, waypoints=11, candidates=16
        )
        if oracle is None:
            continue
        base = {
            "scene_id": f"{prefix}-{len(scenes):05d}", "scenario": "confined_space",
            "start": start.tolist(), "goal": goal.tolist(),
            "start_q": q0.tolist(), "goal_q": q1.tolist(),
            "obstacles": obstacles, "corridor_free_width": float(free_width),
            "feasible": True, **oracle,
        }
        scene = _make_pd_blocker(generator, evaluator, base, 10, rng)
        if scene is None:
            continue
        scene["scene_id"] = base["scene_id"]
        scene["scenario"] = "confined_space"
        scene["corridor_free_width"] = float(free_width)
        scene["scene_fingerprint"] = fingerprint(scene)
        if scene["scene_fingerprint"] in used:
            continue
        used.add(scene["scene_fingerprint"])
        scenes.append(scene)
        if len(scenes) % 10 == 0:
            print(f"[{prefix}] {len(scenes)}/{count}", flush=True)
    return scenes


def _init_generation_worker(root_text):
    global _WORKER_GENERATOR, _WORKER_EVALUATOR
    root = Path(root_text)
    _WORKER_GENERATOR = TrajectoryGenerator(
        DEFAULT_URDF,
        obstacle_radius_range=(0.025, 0.055),
    )
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    _WORKER_GENERATOR.collision_detector = CollisionDetector(
        model, mujoco.MjData(model))
    _WORKER_EVALUATOR = PDEvaluator(root)


def _generate_one(task):
    kind, index, n_obstacles, prefix, seed = task
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if kind == "corridor":
        scenes = generate_corridors(
            _WORKER_GENERATOR, _WORKER_EVALUATOR, 1, prefix, set(), rng)
    else:
        scenes = generate_obstacle_set(
            _WORKER_GENERATOR, _WORKER_EVALUATOR, 1, n_obstacles,
            prefix, set(), rng)
    scene = scenes[0]
    scene["scene_id"] = f"{prefix}-{index:05d}"
    return index, scene


def generate_parallel(pool, kind, count, n_obstacles, prefix, seed):
    tasks = [(kind, index, n_obstacles, prefix, seed + 10007 * index)
             for index in range(count)]
    scenes = [None] * count
    for completed, (index, scene) in enumerate(
            pool.imap_unordered(_generate_one, tasks), start=1):
        scenes[index] = scene
        print(f"[{prefix}] {completed}/{count}", flush=True)
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
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root / output
    np.random.seed(args.seed)
    generator = TrajectoryGenerator(
        DEFAULT_URDF,
        obstacle_radius_range=(0.025, 0.055),
    )
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    generator.collision_detector = CollisionDetector(model, mujoco.MjData(model))
    used = set()

    write(output / "free_space/test.json",
          generate_free_space(generator, args.test, used))
    context = mp.get_context("spawn")
    with context.Pool(
        args.workers, initializer=_init_generation_worker,
        initargs=(str(root),),
    ) as pool:
        whole_train = generate_parallel(
            pool, "obstacle", args.train, 3, "whole_body-train",
            args.seed + 1_000_000)
        whole_validation = generate_parallel(
            pool, "obstacle", args.validation, 3, "whole_body-validation",
            args.seed + 2_000_000)
        whole_test = generate_parallel(
            pool, "obstacle", args.test, 3, "whole_body-test",
            args.seed + 3_000_000)
        confined_train = generate_parallel(
            pool, "corridor", args.train, 10, "confined_space-train",
            args.seed + 4_000_000)
        confined_validation = generate_parallel(
            pool, "corridor", args.validation, 10,
            "confined_space-validation", args.seed + 5_000_000)
        confined_test = generate_parallel(
            pool, "corridor", args.test, 10, "confined_space-test",
            args.seed + 6_000_000)
        generalization_train = generate_parallel(
            pool, "obstacle", args.train, 1, "generalization-train",
            args.seed + 7_000_000)
        generalization_validation = generate_parallel(
            pool, "obstacle", args.validation, 1,
            "generalization-validation", args.seed + 8_000_000)
        generalization_test = generate_parallel(
            pool, "obstacle", args.test, 3, "generalization-test",
            args.seed + 9_000_000)
    all_obstacle_scenes = (
        whole_train + whole_validation + whole_test + confined_train
        + confined_validation + confined_test + generalization_train
        + generalization_validation + generalization_test)
    for scene in all_obstacle_scenes:
        if scene["scene_fingerprint"] in used:
            raise RuntimeError(
                f"duplicate generated scene: {scene['scene_fingerprint']}")
        used.add(scene["scene_fingerprint"])
    write(output / "whole_body/train.json", whole_train)
    write(output / "whole_body/validation.json", whole_validation)
    write(output / "whole_body/test.json", whole_test)
    write(output / "confined_space/train.json", confined_train)
    write(output / "confined_space/validation.json", confined_validation)
    write(output / "confined_space/test.json", confined_test)
    write(output / "generalization/train.json", generalization_train)
    write(output / "generalization/validation.json", generalization_validation)
    write(output / "generalization/test.json", generalization_test)
    write(output / "curriculum/train.json", whole_train + confined_train)
    write(output / "curriculum/validation.json",
          whole_validation + confined_validation)
    manifest = {"schema_version": 1, "seed": args.seed,
                "robot_model": {
                    "name": "ewalker_inspired_7dof",
                    "status": "research reconstruction; not flight hardware",
                    "urdf": "ewalker_description/urdf/ewalker.urdf",
                    "mjcf": "models/ewalker_scene.xml",
                    "geometry_source": ("E-Walker public-paper topology, "
                                        "tubular form and 1.3 m envelope"),
                    "inertia_status": "modelled values",
                },
                "train_count_per_scenario": args.train,
                "validation_count_per_scenario": args.validation,
                "test_count_per_scenario": args.test,
                "generation_workers": args.workers,
                "global_unique_scenes": len(used),
                "protocol": {"free_space": "figure-eight, no obstacles",
                             "difficulty": ("hard means the zero-residual PD "
                                            "controller fails while the IK-graph "
                                            "oracle certifies a safe path"),
                             "whole_body": "PD-blocking linear scan, 3 spheres",
                             "confined_space": ("PD-blocking linear scan, "
                                                "10-sphere corridor"),
                             "generalization": ("PD-blocking train 1 sphere, "
                                                "test 3 spheres"),
                             "obstacle_geometry": (
                                 "sphere collision proxies; not task-object CAD")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
