"""Validate frozen-scene results and produce paired paper statistics.

The script refuses duplicate/missing scene IDs, reports seed-level variation for
the learned method, and compares every RL seed with one deterministic baseline
on the same scenes.  It never pools episodes across seeds as independent runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import binomtest, wilcoxon


QUALITY_METRICS = (
    "completion_time_s",
    "tracking_rms_m",
    "joint_velocity_smoothness",
    "torque_rate_nm_s",
    "energy",
)


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"empty result file: {path}")
    return rows


def expected_scene_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenes = payload["scenes"] if isinstance(payload, dict) else payload
    return {str(scene["scene_id"]) for scene in scenes}


def index_complete(rows: list[dict], expected: set[str], label: str) -> dict[str, dict]:
    ids = [str(row["scene_id"]) for row in rows]
    duplicates = sorted({scene_id for scene_id in ids if ids.count(scene_id) > 1})
    actual = set(ids)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if duplicates or missing or extra or len(rows) != len(expected):
        raise ValueError(
            f"{label}: invalid coverage: rows={len(rows)}, expected={len(expected)}, "
            f"duplicates={duplicates[:5]}, missing={missing[:5]}, extra={extra[:5]}"
        )
    return {str(row["scene_id"]): row for row in rows}


def seed_summary(rows: list[dict]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"seed": int(rows[0]["seed"]), "n": len(rows)}
    for metric in ("success", "collision", *QUALITY_METRICS):
        result[metric] = float(np.mean([float(row[metric]) for row in rows]))
    return result


def paired_comparison(rl: dict[str, dict], baseline: dict[str, dict],
                      rng: np.random.Generator, bootstrap_samples: int) -> dict:
    ids = sorted(rl)
    success_delta = np.asarray(
        [float(rl[i]["success"]) - float(baseline[i]["success"]) for i in ids]
    )
    bootstrap = np.mean(
        rng.choice(success_delta, size=(bootstrap_samples, len(ids)), replace=True), axis=1
    )
    rl_only = int(np.sum(success_delta == 1))
    baseline_only = int(np.sum(success_delta == -1))
    discordant = rl_only + baseline_only
    result = {
        "success_difference": float(np.mean(success_delta)),
        "success_bootstrap_95_ci": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "rl_only_successes": rl_only,
        "baseline_only_successes": baseline_only,
        "mcnemar_exact_p": (float(binomtest(min(rl_only, baseline_only), discordant, 0.5).pvalue)
                            if discordant else 1.0),
        "continuous_metrics": {},
    }
    for metric in QUALITY_METRICS:
        rl_values = np.asarray([float(rl[i][metric]) for i in ids])
        baseline_values = np.asarray([float(baseline[i][metric]) for i in ids])
        delta = rl_values - baseline_values
        try:
            p_value = float(wilcoxon(delta, alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
        result["continuous_metrics"][metric] = {
            "rl_mean": float(np.mean(rl_values)),
            "baseline_mean": float(np.mean(baseline_values)),
            "relative_difference": float(np.mean(rl_values) / np.mean(baseline_values) - 1.0),
            "wilcoxon_p": p_value,
        }
    return result


def analyze(scene_json: Path, rl_paths: list[Path], baseline_path: Path,
            bootstrap_samples: int = 20_000, random_seed: int = 20260826) -> dict:
    expected = expected_scene_ids(scene_json)
    baseline_rows = load_rows(baseline_path)
    baseline = index_complete(baseline_rows, expected, "baseline")
    rng = np.random.default_rng(random_seed)
    seed_summaries, comparisons = [], []
    seen_seeds: set[int] = set()
    for path in rl_paths:
        rows = load_rows(path)
        seed = int(rows[0]["seed"])
        if any(int(row["seed"]) != seed for row in rows):
            raise ValueError(f"{path}: contains more than one seed")
        if seed in seen_seeds:
            raise ValueError(f"duplicate RL seed: {seed}")
        seen_seeds.add(seed)
        indexed = index_complete(rows, expected, f"RL seed {seed}")
        seed_summaries.append(seed_summary(rows))
        comparisons.append({"seed": seed, **paired_comparison(
            indexed, baseline, rng, bootstrap_samples)})
    seed_summaries.sort(key=lambda item: int(item["seed"]))
    aggregate = {}
    for metric in ("success", "collision", *QUALITY_METRICS):
        values = [float(item[metric]) for item in seed_summaries]
        aggregate[metric] = {
            "mean": mean(values),
            "sample_std": stdev(values) if len(values) > 1 else 0.0,
        }
    return {
        "scene_json": str(scene_json),
        "scene_count": len(expected),
        "baseline": seed_summary(baseline_rows),
        "rl_seeds": seed_summaries,
        "rl_across_seeds": aggregate,
        "paired_vs_baseline": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--rl", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    report = analyze(args.scene_json, args.rl, args.baseline, args.bootstrap_samples)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
