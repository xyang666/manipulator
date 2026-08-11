"""Single source of truth for phase-one training and evaluation defaults."""

from dataclasses import asdict, dataclass, field
from typing import Any


PHASE1_METHODS = (
    "pd",
    "gradient_projection",
    "cbf_qp",
    "gradient_cbf",
    "adaptive_gradient_cbf",
    "sac_joint",
    "sac_residual",
    "ours_no_physics",
    "ours_physics",
    "ours_full",
)
PHASE1_SCENARIOS = (
    "free_space",
    "whole_body",
    "confined_space",
    "generalization",
)
TRAINING_PROTOCOL_VERSION = 5
TRAIN_SEEDS = (11, 23, 37, 53, 71)


@dataclass(frozen=True)
class AlgorithmDefaults:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.1
    minimum_alpha: float = 0.05
    batch_size: int = 512
    replay_size: int = 500_000
    start_steps: int = 10_000
    hidden_dims: tuple[int, int] = (256, 256)
    n_critics: int = 5
    lambda_dyn: float = 1.0
    physics_soft_limit_ratio: float = 0.80
    task_scale: float = 1.0
    nullspace_scale: float = 0.5
    cost_limit: float = 0.10
    lagrange_learning_rate: float = 0.0001
    lagrange_initial_value: float = 0.1
    lagrange_maximum: float = 10.0
    parallel_envs: int = 32
    gradient_steps: int = 8
    validation_interval_steps: int = 25_000
    checkpoint_interval_steps: int = 50_000
    uniform_scene_mix: float = 0.50
    scene_weight_ratio_max: float = 3.0


@dataclass(frozen=True)
class EnvironmentDefaults:
    dt: float = 0.02
    episode_len: int = 500
    trajectory_steps: int = 350
    tracking_full_speed_error: float = 0.03
    tracking_stop_error: float = 0.08
    success_tolerance: float = 0.05
    success_hold_steps: int = 10
    d_safe: float = 0.06
    w_track: float = 12.0
    w_obs: float = 5.0
    w_manip: float = 0.05
    w_energy: float = 0.001
    w_collision: float = 100.0
    collision_event_penalty: float = 500.0
    w_action: float = 0.05
    w_null: float = 0.05
    success_bonus: float = 100.0
    reward_scale: float = 10.0
    obs_scene_embed: int = 10
    obs_waypoint_steps: tuple[int, ...] = (10, 25, 50)


@dataclass(frozen=True)
class EvaluationDefaults:
    episodes_per_seed: int = 100
    confidence: float = 0.95
    success_tolerance_m: float = 0.05
    train_seeds: tuple[int, ...] = field(default_factory=lambda: TRAIN_SEEDS)


ALGORITHM = AlgorithmDefaults()
ENVIRONMENT = EnvironmentDefaults()
EVALUATION = EvaluationDefaults()


def phase1_defaults() -> dict[str, Any]:
    """Return a JSON-serializable snapshot embedded in every experiment result."""
    return {
        "algorithm": asdict(ALGORITHM),
        "environment": asdict(ENVIRONMENT),
        "evaluation": asdict(EVALUATION),
        "methods": list(PHASE1_METHODS),
        "scenarios": list(PHASE1_SCENARIOS),
    }
