"""Generate disjoint scene splits with task-path feasibility certificates.

The saved oracle path is generation evidence only. Environments must not expose
it in observations or use it as a reference trajectory.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from trajectory.generator import TrajectoryGenerator


OBSTACLE_MIX = (1, 2, 3)


def _allocate(total, values=OBSTACLE_MIX):
    return [values[index % len(values)] for index in range(total)]


def generate_split(generator, name, count, used_fingerprints,
                   easy_fraction, max_attempts, waypoints, candidates,
                   obstacle_mix=OBSTACLE_MIX):
    scenes = []
    easy_quota = round(count * easy_fraction)
    for obstacle_count in _allocate(count, obstacle_mix):
        require_nontrivial = len(scenes) >= easy_quota
        batch = 0
        while batch < 10:
            scene = generator.generate_scene(
                scene_id=len(scenes),
                n_obstacles=obstacle_count,
                max_attempts=max_attempts,
                ahead_mode=True,
                require_nontrivial=require_nontrivial,
                oracle_waypoints=waypoints,
                oracle_candidates=candidates,
            )
            if scene is None:
                batch += 1
                print(f"[{name}] retry scene {len(scenes)} batch {batch}/10",
                      flush=True)
                continue
            fingerprint = scene["scene_fingerprint"]
            if fingerprint not in used_fingerprints:
                break
        else:
            raise RuntimeError(
                f"Could not generate {name} scene {len(scenes)} after "
                f"{10 * max_attempts} attempts"
            )
        used_fingerprints.add(fingerprint)
        scene["scene_id"] = f"{name}-{len(scenes):05d}"
        scene["split"] = name
        scenes.append(scene)
        if len(scenes) % 10 == 0 or len(scenes) == count:
            print(f"[{name}] {len(scenes)}/{count}", flush=True)
    return scenes


def main():
    from robot_config import DEFAULT_URDF
    parser = argparse.ArgumentParser(
        description="Generate independently certified train/validation/test scenes"
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--output_dir", default="results/solvable_scenes")
    parser.add_argument("--train", type=int, default=500)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--easy_fraction", type=float, default=0.15)
    parser.add_argument("--oracle_waypoints", type=int, default=21)
    parser.add_argument("--oracle_candidates", type=int, default=10)
    parser.add_argument("--max_attempts", type=int, default=500)
    parser.add_argument("--obstacle_mix", default="1,2,3",
                        help="Comma-separated obstacle counts, cycled per split")
    args = parser.parse_args()

    if not 0.0 <= args.easy_fraction <= 1.0:
        parser.error("--easy_fraction must be between zero and one")
    try:
        obstacle_mix = tuple(int(value) for value in args.obstacle_mix.split(","))
    except ValueError:
        parser.error("--obstacle_mix must contain comma-separated integers")
    if not obstacle_mix or any(value < 0 for value in obstacle_mix):
        parser.error("--obstacle_mix must contain non-negative obstacle counts")
    np.random.seed(args.seed)
    root = Path(__file__).resolve().parents[2]
    urdf = Path(args.urdf)
    if not urdf.is_absolute():
        urdf = root / urdf
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = TrajectoryGenerator(str(urdf))
    used = set()
    split_sizes = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "oracle": "task_path_ik_graph",
        "oracle_waypoints": args.oracle_waypoints,
        "oracle_candidates": args.oracle_candidates,
        "task_edge_tolerance_m": 0.03,
        "collision_clearance_m": 0.02,
        "oracle_clearance_m": 0.025,
        "splits": {},
    }
    for name, count in split_sizes.items():
        scenes = generate_split(
            generator, name, count, used, args.easy_fraction,
            args.max_attempts, args.oracle_waypoints, args.oracle_candidates,
            obstacle_mix,
        )
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(scenes, indent=2), encoding="utf-8")
        manifest["splits"][name] = {
            "path": path.name,
            "count": len(scenes),
            "difficulty": dict(Counter(s["difficulty"] for s in scenes)),
            "obstacle_counts": dict(Counter(len(s["obstacles"]) for s in scenes)),
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Generated {len(used)} unique certified scenes in {output_dir}")


if __name__ == "__main__":
    main()
