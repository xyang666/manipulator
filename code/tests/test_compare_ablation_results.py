import json

import pytest

from experiments.compare_ablation_results import analyze


def _row(scene, seed, success, value):
    return {
        "scene_id": scene, "seed": seed, "success": success,
        "collision": not success, "completion_time_s": value,
        "tracking_rms_m": value, "joint_velocity_smoothness": value,
        "torque_rate_nm_s": value, "energy": value,
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_matched_seed_ablation(tmp_path):
    scenes = tmp_path / "scenes.json"
    scenes.write_text(json.dumps([{"scene_id": "a"}, {"scene_id": "b"}]))
    controls, treatments = [], []
    for seed, control_value, treatment_value in ((1, 2.0, 1.0), (2, 4.0, 2.0)):
        control, treatment = tmp_path / f"c{seed}", tmp_path / f"t{seed}"
        _write(control, [_row("a", seed, True, control_value),
                         _row("b", seed, False, control_value)])
        _write(treatment, [_row("a", seed, True, treatment_value),
                           _row("b", seed, True, treatment_value)])
        controls.append(control); treatments.append(treatment)
    report = analyze(scenes, controls, treatments)
    smooth = report["across_seeds"]["joint_velocity_smoothness"]
    assert smooth["control_mean"] == 3.0
    assert smooth["treatment_mean"] == 1.5
    assert smooth["relative_difference_of_means"] == -0.5
    assert report["seed_results"][0]["metrics"]["success"]["treatment_only"] == 1


def test_rejects_seed_mismatch(tmp_path):
    scenes = tmp_path / "scenes.json"
    scenes.write_text(json.dumps([{"scene_id": "a"}]))
    control, treatment = tmp_path / "c", tmp_path / "t"
    _write(control, [_row("a", 1, True, 1.0)])
    _write(treatment, [_row("a", 2, True, 1.0)])
    with pytest.raises(ValueError, match="seed mismatch"):
        analyze(scenes, [control], [treatment])
