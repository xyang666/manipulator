import json

from experiments.augment_rl_challenge_training import build_training_set


def _scene(scene_id, scenario, obstacles):
    return {
        "scene_id": scene_id,
        "scenario": scenario,
        "start": [0.4, -0.2, 0.3],
        "goal": [0.4, 0.2, 0.3],
        "start_q": [0.0] * 7,
        "goal_q": [0.1] * 7,
        "obstacles": obstacles,
    }


def test_build_training_set_only_uses_training_sources(tmp_path):
    whole = [_scene("whole_body-train-0", "whole_body",
                    [[0.45, 0.0, 0.3, 0.03]])]
    gate = _scene("rl-gate-train-0", "rl_challenge_gate", [
        [0.35, 0.0, 0.3, 0.02, 0.01, 0.0, 0.0],
        [0.45, 0.0, 0.3, 0.02, -0.01, 0.0, 0.0],
    ])
    gate["challenge_type"] = "closing_gate"
    whole_path = tmp_path / "whole.json"
    base_path = tmp_path / "base.json"
    output = tmp_path / "augmented.json"
    whole_path.write_text(json.dumps(whole))
    base_path.write_text(json.dumps([gate]))

    manifest = build_training_set(
        whole_path, base_path, output, seed=7, gate_variants=2,
        swing_range=(0.04, 0.05), duration_range=(6.0, 7.0))

    scenes = json.loads(output.read_text())
    assert manifest["controller_conditioned_selection"] is False
    assert manifest["total_scenes"] == 3
    assert {scene["challenge_type"] for scene in scenes} == {
        "timed_crossing", "closing_gate"
    }
    assert len({scene["scene_id"] for scene in scenes}) == 3
    assert all("augmentation_duration_s" in scene for scene in scenes)
