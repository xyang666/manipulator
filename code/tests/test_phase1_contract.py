import numpy as np
import torch

from agent.physics_policy import PhysicsInformedActor
from agent.vanilla_sac_agent import VanillaSACAgent
from experiment_config import EVALUATION, PHASE1_METHODS, PHASE1_SCENARIOS
from experiments.manifest import build_manifest


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
    assert len(manifest["training_jobs"]) == 5 * len(EVALUATION.train_seeds)
    assert all(job["seed"] in EVALUATION.train_seeds
               for job in manifest["evaluation_jobs"])
    assert all("--episodes" in job["command"] for job in manifest["evaluation_jobs"])
