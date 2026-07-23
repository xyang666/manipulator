import json

import numpy as np

from experiments.metrics import EpisodeRecorder, summarize_episodes
from experiments.report import aggregate, write_outputs


def test_episode_metrics_from_synthetic_trajectory():
    recorder = EpisodeRecorder(
        "pd", "free_space", seed=11, scene_id=0, dt=0.1,
        torque_limits=np.array([1.0, 1.0]),
    )
    recorder.add(info={"tracking_error": 0.03, "d_obs": 0.20},
                 dq=np.array([0.0, 0.0]), torque=np.array([0.0, 0.0]))
    recorder.add(info={"tracking_error": 0.04, "d_obs": 0.10, "success": True},
                 dq=np.array([0.1, 0.0]), torque=np.array([2.0, 0.0]),
                 nullspace_norm=0.2, reference_norm=0.1, gate=0.5)

    result = recorder.finish()

    assert result.success
    assert result.tracking_rms_m == 0.035355339059327376
    assert result.tracking_peak_m == 0.04
    assert result.min_clearance_m == 0.10
    assert result.joint_velocity_smoothness == 1.0
    assert result.torque_rate_nm_s == 20.0
    assert result.gate_trigger_rate == 0.5
    assert result.torque_violation_count == 1


def test_collision_invalidates_success():
    recorder = EpisodeRecorder("pd", "whole_body", 11, 0, 0.02)
    recorder.add(info={"success": True, "collision": True}, dq=np.zeros(2))
    assert recorder.finish().success is False


def test_summary_and_generated_outputs(tmp_path):
    rows = []
    for seed, success in [(11, True), (23, False)]:
        recorder = EpisodeRecorder("pd", "free_space", seed, seed, 0.02)
        recorder.add(info={"success": success, "tracking_error": 0.01,
                           "d_obs": 0.2}, dq=np.zeros(2))
        rows.append(recorder.finish().to_dict())

    summary = summarize_episodes(rows)
    assert summary["n"] == 2
    assert summary["metrics"]["success"]["mean"] == 0.5
    assert summary["metrics"]["success"]["ci_low"] < 0.5

    write_outputs(aggregate(rows), tmp_path)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
    latex = (tmp_path / "phase1_results.tex").read_text()
    assert "pd & free_space" in latex
    assert json.loads((tmp_path / "summary.json").read_text())[0]["n"] == 2
