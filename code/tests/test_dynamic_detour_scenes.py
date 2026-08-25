import numpy as np

from experiments.generate_dynamic_detour_scenes import make_dynamic_detour


class FakeKinematics:
    def get_link_capsules(self, q):
        x = float(np.asarray(q)[0])
        return [(np.array([x, 0.0, 0.0]),
                 np.array([x, 0.0, 0.1]), 0.01)]


def test_dynamic_detour_preserves_oracle_and_adds_velocity():
    scene = {
        "scene_id": "rl-detour-source",
        "start_q": [-0.2], "goal_q": [0.2],
        "start": [-0.2, 0.0, 0.05], "goal": [0.2, 0.0, 0.05],
        "obstacles": [[0.0, 0.0, 0.05, 0.02]],
        "added_blocker_count": 1,
        "rrt_connect_feasible": True,
        "feasible_q_path": [[-0.2], [0.2]],
    }
    result = make_dynamic_detour(
        scene, FakeKinematics(), np.random.default_rng(3),
        swing_range=(0.1, 0.1), duration=7.0,
        endpoint_clearance=0.01)

    assert result is not None
    assert result["challenge_type"] == "predictive_dynamic_detour"
    assert result["rrt_connect_feasible"] is True
    assert result["moving_obstacle_indices"] == [0]
    assert len(result["obstacles"][0]) == 7
    assert np.linalg.norm(result["obstacles"][0][4:]) > 0.0
    assert result["dynamic_generation_controller_conditioned"] is False
