#!/usr/bin/env python3
"""Build a broader *training-only* predictive-obstacle distribution.

This utility never reads validation or test results and never runs a
controller.  It converts independently generated whole-body training scenes
to timed crossings and augments the original closing-gate training scenes by
sampling motion parameters.  The fixed validation/test files remain untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.generate_rl_challenge_scenes import (
    make_closing_gate,
    make_timed_crossing,
)


def build_training_set(whole_body_json: Path, base_challenge_json: Path,
                       output: Path, seed: int, gate_variants: int,
                       swing_range: tuple[float, float],
                       duration_range: tuple[float, float]) -> dict:
    """Create a controller-independent augmented training set and manifest."""
    rng = np.random.default_rng(seed)
    whole_body = json.loads(whole_body_json.read_text())
    base = json.loads(base_challenge_json.read_text())
    gates = [scene for scene in base
             if scene.get("challenge_type") == "closing_gate"]

    scenes = []
    for index, source in enumerate(whole_body):
        swing = float(rng.uniform(*swing_range))
        duration = float(rng.uniform(*duration_range))
        scene = make_timed_crossing(source, rng, swing, duration)
        scene["scene_id"] = f"rl-crossing-aug-{index:05d}"
        scene["augmentation_swing_m"] = swing
        scene["augmentation_duration_s"] = duration
        scenes.append(scene)

    for variant in range(gate_variants):
        for index, source in enumerate(gates):
            swing = float(rng.uniform(*swing_range))
            duration = float(rng.uniform(*duration_range))
            # Only xyz/r are static geometry; discard the old motion fields.
            static = dict(source)
            static["obstacles"] = [obstacle[:4]
                                   for obstacle in source["obstacles"]]
            scene = make_closing_gate(static, swing, duration)
            scene["scene_id"] = f"rl-gate-aug-{variant:02d}-{index:05d}"
            scene["augmentation_swing_m"] = swing
            scene["augmentation_duration_s"] = duration
            scenes.append(scene)

    rng.shuffle(scenes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scenes, indent=2) + "\n")
    manifest = {
        "protocol": "rl_challenge_train_aug_v2",
        "controller_conditioned_selection": False,
        "seed": seed,
        "whole_body_source": str(whole_body_json),
        "gate_source": str(base_challenge_json),
        "crossing_scenes": len(whole_body),
        "gate_scenes": len(gates) * gate_variants,
        "total_scenes": len(scenes),
        "gate_variants": gate_variants,
        "swing_range_m": list(swing_range),
        "duration_range_s": list(duration_range),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whole-body-json", required=True, type=Path)
    parser.add_argument("--base-challenge-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--gate-variants", type=int, default=3)
    parser.add_argument("--swing-range", nargs=2, type=float,
                        default=(0.04, 0.07), metavar=("MIN", "MAX"))
    parser.add_argument("--duration-range", nargs=2, type=float,
                        default=(6.0, 8.0), metavar=("MIN", "MAX"))
    args = parser.parse_args()
    if args.gate_variants <= 0:
        parser.error("--gate-variants must be positive")
    if not 0 < args.swing_range[0] <= args.swing_range[1]:
        parser.error("--swing-range must satisfy 0 < MIN <= MAX")
    if not 0 < args.duration_range[0] <= args.duration_range[1]:
        parser.error("--duration-range must satisfy 0 < MIN <= MAX")
    manifest = build_training_set(
        args.whole_body_json, args.base_challenge_json, args.output,
        args.seed, args.gate_variants, tuple(args.swing_range),
        tuple(args.duration_range))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
