import numpy as np
import torch

from agent.physics_policy import PhysicsInformedActor
from agent.sac_agent import normalized_discounted_cost, scaled_sigmoid_inverse
from agent.vanilla_sac_agent import VanillaSACAgent
from experiment_config import (ALGORITHM, ENVIRONMENT, EVALUATION,
                               PHASE1_METHODS, PHASE1_SCENARIOS)
from experiments.manifest import build_manifest
from experiments.runner import (gradient_control_defaults,
                                self_safety_distance_default)
from experiments.split_scenes import scene_fingerprint, split_scenes
from env.manipulator_env import ManipulatorEnv, dense_safety_cost
from robot_config import DEFAULT_URDF, DEFAULT_XML
from train import (is_better_validation, prune_step_checkpoints,
                   scene_sampling_weights)
from utils.cbf import CBFController
from utils.replay_buffer import ReplayBuffer


def test_actor_has_three_plus_four_action_contract():
    actor = PhysicsInformedActor(state_dim=25, action_dim=7)
    action, _, mean = actor.sample(torch.zeros(3, 25))
    assert action.shape == (3, 7)
    assert mean.shape == (3, 7)


def test_vanilla_agent_supports_parallel_action_selection():
    agent = VanillaSACAgent(state_dim=25, action_dim=7)
    actions = agent.select_action_batch(np.zeros((4, 25), dtype=np.float32))
    assert actions.shape == (4, 7)


def test_svd_nullspace_coordinates_are_orthogonal():
    rng = np.random.default_rng(7)
    jacobian = rng.normal(size=(3, 7))
    _, _, vh = np.linalg.svd(jacobian, full_matrices=True)
    basis = vh[3:].T
    z = rng.normal(size=4)
    assert np.linalg.norm(jacobian @ (basis @ z)) < 1e-10


def test_manifest_covers_all_methods_scenarios_and_seeds(tmp_path):
    manifest = build_manifest(tmp_path / "checkpoints", tmp_path / "results")
    expected = len(PHASE1_METHODS) * len(PHASE1_SCENARIOS) * len(EVALUATION.train_seeds)
    assert len(manifest["evaluation_jobs"]) == expected
    assert len(manifest["training_jobs"]) == 10 * len(EVALUATION.train_seeds)
    assert all(job["seed"] in EVALUATION.train_seeds
               for job in manifest["evaluation_jobs"])
    assert all("--episodes" in job["command"] for job in manifest["evaluation_jobs"])
    assert all("--scene_json" in job["command"] and "--val_json" in job["command"]
               for job in manifest["training_jobs"])
    assert all(job["command"][job["command"].index("--val_scenes") + 1]
               == ("40" if job["protocol"] == "curriculum" else "20")
               for job in manifest["training_jobs"])
    assert all("--scene-json" in job["command"]
               for job in manifest["evaluation_jobs"])
    assert all("--val_every_steps" in job["command"]
               for job in manifest["training_jobs"])
    assert len(manifest["diagnostic_jobs"]) == 1
    assert "whole_body/train.json" in " ".join(
        manifest["diagnostic_jobs"][0]["command"]
    )


def test_adaptive_gradient_cbf_uses_scenario_specific_smoothing():
    assert gradient_control_defaults(
        "adaptive_gradient_cbf", "whole_body", None, None
    ) == (0.3, 0.8)
    assert gradient_control_defaults(
        "adaptive_gradient_cbf", "confined_space", None, None
    ) == (0.3, 0.9)
    assert gradient_control_defaults(
        "gradient_projection", "confined_space", None, None
    ) == (ALGORITHM.nullspace_scale, 0.0)


def test_dense_safety_cost_tracks_margin_violation():
    assert dense_safety_cost(0.2, 0.1) == 0.0
    assert np.isclose(dense_safety_cost(0.05, 0.1), 0.5)
    assert dense_safety_cost(0.0, 0.1) == 1.0
    assert dense_safety_cost(-1.0, 0.1) == 2.0


def test_validation_selection_prioritizes_success_then_safety():
    incumbent = {"success_rate": 0.5, "collision_rate": 0.3,
                 "avg_tracking_error": 0.02}
    assert is_better_validation(
        {"success_rate": 0.6, "collision_rate": 0.9,
         "avg_tracking_error": 0.5}, incumbent)
    assert is_better_validation(
        {"success_rate": 0.5, "collision_rate": 0.2,
         "avg_tracking_error": 0.5}, incumbent)
    assert not is_better_validation(
        {"success_rate": 0.5, "collision_rate": 0.4,
         "avg_tracking_error": 0.001}, incumbent)


def test_lagrange_raw_parameter_represents_paper_initial_value():
    raw = scaled_sigmoid_inverse(0.1, ALGORITHM.lagrange_maximum)
    actual = ALGORITHM.lagrange_maximum / (1.0 + np.exp(-raw))
    assert np.isclose(actual, 0.1)


def test_discounted_safety_value_is_normalized_to_per_step_scale():
    assert np.isclose(normalized_discounted_cost(5.0, 0.99), 0.05)


def test_cbf_filters_motion_toward_self_collision_without_obstacles():
    class EmptySDF:
        def min_distance(self, *args, **kwargs):
            return np.inf

    class OneJointKinematics:
        n = 1

        def forward_kinematics(self, q):
            return np.zeros(3), np.eye(3)

        def compute_self_distances(self, q):
            return np.array([q[0]])

    cbf = CBFController(EmptySDF(), OneJointKinematics(), self_d_safe=0.02)
    filtered, info = cbf.filter(np.array([-1.0]), np.array([0.03]))
    assert info["self_active"]
    assert not info["obstacle_active"]
    assert filtered[0] >= -cbf.alpha * info["self_h"] - 1e-9


def test_minimum_jerk_progress_has_stationary_endpoints():
    fn = ManipulatorEnv._minimum_jerk_progress
    eps = 1e-5
    assert fn(0.0) == 0.0
    assert fn(1.0) == 1.0
    assert np.isclose((fn(eps) - fn(0.0)) / eps, 0.0, atol=1e-8)
    assert np.isclose((fn(1.0) - fn(1.0 - eps)) / eps, 0.0, atol=1e-8)


def test_tracking_progress_gate_uses_configured_error_band():
    env = ManipulatorEnv.__new__(ManipulatorEnv)
    env.tracking_full_speed_error = 0.03
    env.tracking_stop_error = 0.08
    assert env._tracking_progress_gate(0.02) == 1.0
    assert env._tracking_progress_gate(0.08) == 0.0
    assert np.isclose(env._tracking_progress_gate(0.055), 0.5)


def test_scene_sampling_keeps_uniform_probability_floor():
    weights = scene_sampling_weights(
        np.array([1.0, 0.0]), uniform_mix=0.5, max_ratio=3.0
    )
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 1.0 / 2.0 / 3.0)
    assert weights[1] > weights[0]


def test_adaptive_free_space_uses_predictive_self_collision_margin():
    assert self_safety_distance_default("adaptive_gradient_cbf", "free_space") == 0.03
    assert self_safety_distance_default("cbf_qp", "free_space") == 0.02
    assert self_safety_distance_default("adaptive_gradient_cbf", "whole_body") == 0.02


def test_scene_sampling_caps_extreme_imbalance():
    weights = scene_sampling_weights(
        np.array([1.0] * 99 + [0.0]), uniform_mix=0.5, max_ratio=3.0
    )
    uniform = 1.0 / weights.size
    assert weights.max() <= uniform * 3.0 + 1e-12


def test_periodic_checkpoint_retention_keeps_only_newest(tmp_path):
    for step in (100, 200, 300):
        (tmp_path / f"ckpt_step{step:09d}.pt").write_bytes(b"model")
        (tmp_path / f"ckpt_step{step:09d}.replay.npz").write_bytes(b"replay")
    (tmp_path / "ckpt_best.pt").write_bytes(b"best")
    prune_step_checkpoints(str(tmp_path), keep=1)
    assert sorted(p.name for p in tmp_path.glob("ckpt_step*.pt")) == [
        "ckpt_step000000300.pt"
    ]
    assert (tmp_path / "ckpt_step000000300.replay.npz").exists()
    assert (tmp_path / "ckpt_best.pt").exists()
    prune_step_checkpoints(str(tmp_path), keep=0)
    assert not list(tmp_path.glob("ckpt_step*"))


def test_v5_observation_contains_direction_scene_mask_and_waypoints():
    env = ManipulatorEnv(
        urdf_path=DEFAULT_URDF, xml_path=DEFAULT_XML, n_obstacles=3
    )
    env.reset(seed=3)
    obs = env._get_obs()
    expected = (
        env.n * 2 + 3 + 3
        + len(ENVIRONMENT.obs_waypoint_steps) * 3
        + env._capsule_dists_dim * 4
        + env._self_dists_dim
        + ENVIRONMENT.obs_scene_embed * 5
        + 2
    )
    assert env.obs_dim == expected
    assert obs.shape == (expected,)
    distances, directions = env._mujoco_per_capsule_obstacle_features()
    assert distances.shape == (env._capsule_dists_dim,)
    assert directions.shape == (env._capsule_dists_dim, 3)
    assert np.all(np.isfinite(obs))


def test_scene_split_is_disjoint_and_deterministic():
    scenes = [{"scene_id": i, "start": [i, 0, 0], "goal": [i, 1, 0],
               "obstacles": []} for i in range(10)]
    groups_a, _ = split_scenes(scenes, 6, 2, 2, seed=7)
    groups_b, _ = split_scenes(scenes, 6, 2, 2, seed=7)
    fingerprints = [{scene_fingerprint(s) for s in group} for group in groups_a]
    assert groups_a == groups_b
    assert not (fingerprints[0] & fingerprints[1])
    assert not (fingerprints[0] & fingerprints[2])
    assert not (fingerprints[1] & fingerprints[2])


def test_uniform_replay_buffer_round_trip(tmp_path):
    buffer = ReplayBuffer(capacity=8, state_dim=2, action_dim=1, joints=1)
    for i in range(5):
        buffer.push([i, i + 1], [i], i, [i + 1, i + 2], i == 4,
                    q=[i], dq=[i], dq_next=[i + 1],
                    J=np.ones((3, 1)) * i, sigma=i / 10, dx_nom=np.ones(3) * i,
                    cost=i / 5)
    path = tmp_path / "buffer.replay.npz"
    buffer.save(path)
    restored = ReplayBuffer(capacity=8, state_dim=2, action_dim=1, joints=1)
    restored.load(path)
    assert restored.size == buffer.size and restored.ptr == buffer.ptr
    assert np.array_equal(restored.states[:5], buffer.states[:5])
    assert np.array_equal(restored.costs[:5], buffer.costs[:5])


def test_uniform_replay_buffer_batch_insert_wraps_ring():
    buffer = ReplayBuffer(capacity=5, state_dim=2, action_dim=1, joints=1)
    for offset in (0, 3):
        values = np.arange(offset, offset + 3, dtype=np.float32)
        buffer.push_batch(
            np.stack([values, values + 10], axis=1),
            values[:, None], values, np.stack([values + 1, values + 11], axis=1),
            values % 2 == 0, q=values[:, None], dq=values[:, None],
            dq_next=(values + 1)[:, None], J=values[:, None, None] * np.ones((1, 3, 1)),
            sigma=values, dx_nom=np.repeat(values[:, None], 3, axis=1),
            cost=values / 10,
        )
    assert len(buffer) == 5
    assert buffer.ptr == 1
    assert buffer.states[0, 0] == 5
    assert set(buffer.states[:, 0]) == {1, 2, 3, 4, 5}
