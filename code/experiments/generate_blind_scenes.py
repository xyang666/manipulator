"""Generate a held-out obstacle test set without rewriting formal splits."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from experiments.generate_paper_scenes import (
    _init_generation_worker,
    generate_parallel,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True,
                        choices=("whole_body", "generalization"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1:
        parser.error("--count and --workers must be positive")

    root = Path(__file__).resolve().parents[2]
    prefix = f"{args.scenario}-blind-{args.seed}"
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
