import json

import pytest

from experiments.compare_paper_results import analyze


def _row(scene_id, seed, success, value):
    return {
        "method": "method", "scenario": "scenario", "seed": seed,
        "scene_id": scene_id, "success": success, "collision": not success,
        "completion_time_s": value, "tracking_rms_m": value,
        "joint_velocity_smoothness": value, "torque_rate_nm_s": value,
        "energy": value,
    }


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_analyze_checks_coverage_and_uses_seed_level_statistics(tmp_path):
    scenes = tmp_path / "scenes.json"
    scenes.write_text(json.dumps([{"scene_id": "a"}, {"scene_id": "b"}]))
    baseline, seed_1, seed_2 = (tmp_path / name for name in ("b.jsonl", "1.jsonl", "2.jsonl"))
    _jsonl(baseline, [_row("a", 11, True, 2.0), _row("b", 11, False, 2.0)])
    _jsonl(seed_1, [_row("a", 1, True, 1.0), _row("b", 1, True, 1.0)])
    _jsonl(seed_2, [_row("a", 2, False, 3.0), _row("b", 2, False, 3.0)])

    report = analyze(scenes, [seed_1, seed_2], baseline, bootstrap_samples=100)
    assert report["scene_count"] == 2
    assert report["rl_across_seeds"]["success"] == {"mean": 0.5, "sample_std": pytest.approx(2 ** -0.5)}
    assert report["paired_vs_baseline"][0]["success_difference"] == 0.5


def test_analyze_rejects_incomplete_results(tmp_path):
    scenes = tmp_path / "scenes.json"
    scenes.write_text(json.dumps([{"scene_id": "a"}, {"scene_id": "b"}]))
    baseline, rl = tmp_path / "b.jsonl", tmp_path / "rl.jsonl"
    _jsonl(baseline, [_row("a", 11, True, 1.0), _row("b", 11, True, 1.0)])
    _jsonl(rl, [_row("a", 1, True, 1.0)])
    with pytest.raises(ValueError, match="invalid coverage"):
        analyze(scenes, [rl], baseline, bootstrap_samples=10)
