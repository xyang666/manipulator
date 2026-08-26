"""Compare matched control/treatment checkpoints across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import binomtest, wilcoxon

from experiments.compare_paper_results import (
    QUALITY_METRICS, expected_scene_ids, index_complete, load_rows,
)


METRICS = ("success", "collision", *QUALITY_METRICS)


def analyze(scene_json: Path, controls: list[Path], treatments: list[Path]) -> dict:
    if len(controls) != len(treatments) or not controls:
        raise ValueError("control and treatment must contain the same non-zero number of seeds")
    expected = expected_scene_ids(scene_json)
    seed_results = []
    seen: set[int] = set()
    for control_path, treatment_path in zip(controls, treatments, strict=True):
        control_rows, treatment_rows = load_rows(control_path), load_rows(treatment_path)
        control_seed = {int(row["seed"]) for row in control_rows}
        treatment_seed = {int(row["seed"]) for row in treatment_rows}
        if len(control_seed) != 1 or control_seed != treatment_seed:
            raise ValueError(
                f"seed mismatch: {control_path}={sorted(control_seed)}, "
                f"{treatment_path}={sorted(treatment_seed)}"
            )
        seed = next(iter(control_seed))
        if seed in seen:
            raise ValueError(f"duplicate seed pair: {seed}")
        seen.add(seed)
        control = index_complete(control_rows, expected, f"control seed {seed}")
        treatment = index_complete(treatment_rows, expected, f"treatment seed {seed}")
        ids = sorted(expected)
        result = {"seed": seed, "metrics": {}}
        for metric in METRICS:
            x = np.asarray([float(control[i][metric]) for i in ids])
            y = np.asarray([float(treatment[i][metric]) for i in ids])
            delta = y - x
            item = {
                "control_mean": float(np.mean(x)),
                "treatment_mean": float(np.mean(y)),
                "mean_delta": float(np.mean(delta)),
            }
            if metric in ("success", "collision"):
                treatment_only = int(np.sum(delta == 1))
                control_only = int(np.sum(delta == -1))
                discordant = treatment_only + control_only
                item.update({
                    "treatment_only": treatment_only,
                    "control_only": control_only,
                    "mcnemar_exact_p": (
                        float(binomtest(min(treatment_only, control_only),
                                      discordant, 0.5).pvalue)
                        if discordant else 1.0
                    ),
                })
            else:
                item.update({
                    "relative_difference": float(np.mean(y) / np.mean(x) - 1.0),
                    "wilcoxon_p": (float(wilcoxon(delta).pvalue)
                                   if np.any(delta) else 1.0),
                })
            result["metrics"][metric] = item
        seed_results.append(result)
    seed_results.sort(key=lambda item: item["seed"])
    across = {}
    for metric in METRICS:
        controls_for_metric = [item["metrics"][metric]["control_mean"]
                               for item in seed_results]
        treatments_for_metric = [item["metrics"][metric]["treatment_mean"]
                                 for item in seed_results]
        deltas = [item["metrics"][metric]["mean_delta"] for item in seed_results]
        across[metric] = {
            "control_mean": mean(controls_for_metric),
            "treatment_mean": mean(treatments_for_metric),
            "mean_delta": mean(deltas),
            "delta_sample_std": stdev(deltas) if len(deltas) > 1 else 0.0,
            "relative_difference_of_means": (
                mean(treatments_for_metric) / mean(controls_for_metric) - 1.0
            ),
            "seed_deltas": deltas,
        }
    return {"scene_json": str(scene_json), "scene_count": len(expected),
            "seed_results": seed_results, "across_seeds": across}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--control", type=Path, nargs="+", required=True)
    parser.add_argument("--treatment", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.scene_json, args.control, args.treatment)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
