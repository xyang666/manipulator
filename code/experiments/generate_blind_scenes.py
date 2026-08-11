"""Generate a held-out obstacle test set without rewriting formal splits."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np

from experiments.generate_paper_scenes import (
    _init_generation_worker,
    generate_free_space,
    generate_parallel,
)
from robot_config import DEFAULT_URDF, DEFAULT_XML
from trajectory.generator import TrajectoryGenerator
from utils.collision import CollisionDetector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True,
                        choices=("free_space", "whole_body", "generalization"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1:
        parser.error("--count and --workers must be positive")

    root = Path(__file__).resolve().parents[2]
    prefix = f"{args.scenario}-blind-{args.seed}"
    if args.scenario == "free_space":
        np.random.seed(args.seed)
        generator = TrajectoryGenerator(
            DEFAULT_URDF, obstacle_radius_range=(0.025, 0.055)
        )
        model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
        generator.collision_detector = CollisionDetector(
            model, mujoco.MjData(model)
        )
        scenes = generate_free_space(generator, args.count, set())
        for index, scene in enumerate(scenes):
            scene["scene_id"] = f"{prefix}-{index:05d}"
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            args.workers,
            initializer=_init_generation_worker,
            initargs=(str(root),),
        ) as pool:
            scenes = generate_parallel(
                pool, "obstacle", args.count, 3, prefix, args.seed
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(scenes, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(scenes)} scenes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
