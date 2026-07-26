"""Create deterministic, disjoint train/validation/test scene files."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def scene_fingerprint(scene: dict) -> str:
    payload = {key: value for key, value in scene.items() if key != "scene_id"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def split_scenes(scenes: list[dict], train_count: int, val_count: int,
                 test_count: int, seed: int = 2026):
    unique = {}
    for scene in scenes:
        unique.setdefault(scene_fingerprint(scene), scene)
    required = train_count + val_count + test_count
    if len(unique) < required:
        raise ValueError(f"need {required} unique scenes, found {len(unique)}")
    fingerprints = sorted(unique)
    order = np.random.default_rng(seed).permutation(len(fingerprints))[:required]
    selected = [unique[fingerprints[index]].copy() for index in order]
    cuts = (train_count, train_count + val_count)
    groups = [selected[:cuts[0]], selected[cuts[0]:cuts[1]], selected[cuts[1]:]]
    for group in groups:
        for scene_id, scene in enumerate(group):
            scene["scene_id"] = scene_id
    return tuple(groups), {
        "seed": seed,
        "unique_source_scenes": len(unique),
        "counts": {"train": train_count, "validation": val_count,
                   "test": test_count},
        "fingerprints": {
            name: [scene_fingerprint(scene) for scene in group]
            for name, group in zip(("train", "validation", "test"), groups)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("../results/phase1_splits"))
    parser.add_argument("--train-count", type=int, default=60)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    scenes = []
    source_hashes = {}
    for source in args.sources:
        raw = source.read_bytes()
        source_hashes[str(source)] = hashlib.sha256(raw).hexdigest()
        scenes.extend(json.loads(raw))
    groups, manifest = split_scenes(
        scenes, args.train_count, args.val_count, args.test_count, args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, group in zip(("train", "validation", "test"), groups):
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(group, indent=2) + "\n", encoding="utf-8"
        )
    manifest["source_sha256"] = source_hashes
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
