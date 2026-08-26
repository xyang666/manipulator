"""Merge sharded experiment JSONL after checking IDs, seed, and coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.compare_paper_results import expected_scene_ids, load_rows


def merge(inputs: list[Path], scene_json: Path) -> list[dict]:
    expected = expected_scene_ids(scene_json)
    rows = [row for path in inputs for row in load_rows(path)]
    ids = [str(row["scene_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("shards contain duplicate scene IDs")
    actual = set(ids)
    if actual != expected:
        raise ValueError(
            f"scene coverage mismatch: missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )
    seeds = {int(row["seed"]) for row in rows}
    if len(seeds) != 1:
        raise ValueError(f"shards contain multiple seeds: {sorted(seeds)}")
    return sorted(rows, key=lambda row: str(row["scene_id"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = merge(args.inputs, args.scene_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
