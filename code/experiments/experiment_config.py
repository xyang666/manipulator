"""
experiment_config.py
--------------------
Single source of truth for phase-one training/evaluation configuration.
Used by manifest.py to generate the 50 training + 160 evaluation jobs.

All learning methods share the same hyperparameters below.
Differences between methods are encoded as flag lists in manifest.py.

E-Walker-inspired 7-DoF manipulator, spherical obstacle proxy.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── Algorithm ────────────────────────────────────────────────────────────

@dataclass
class _ALGORITHM:
    parallel_envs: int = 32
    gradient_steps: int = 4          # UTD = 4/32 = 0.125（v4 降低）
    validation_interval_steps: int = 25_000
    checkpoint_interval_steps: int = 50_000
    train_steps: int = 500_000
    episode_len: int = 500
    batch_size: int = 256
    start_steps: int = 10_000
    replay_size: int = 500_000
    learning_rate: float = 3e-4
    tau: float = 0.005
    alpha: float = 0.05               # v4 降低探索噪声
    lambda_dyn: float = 1.0
    task_scale: float = 1.0
    nullspace_scale: float = 0.5
    hidden_dims: list = field(default_factory=lambda: [256, 256])
    n_critics: int = 5
    cost_limit: float = 0.1
    # ── v4 稳定化 ──
    lagrange_learning_rate: float = 1e-4    # 旧值 0.01
    lagrange_maximum: float = 10.0          # 旧值 100
    lagrange_initial_value: float = 0.1
    uniform_scene_mix: float = 0.50         # 旧值 0.20
    scene_weight_ratio_max: float = 3.0     # 单场景最大权重倍数

    def to_cli_args(self) -> list[str]:
        return [
            "--n_envs", str(self.parallel_envs),
            "--grad_steps", str(self.gradient_steps),
            "--val_every_steps", str(self.validation_interval_steps),
            "--episode_len", str(self.episode_len),
            "--batch_size", str(self.batch_size),
            "--start_steps", str(self.start_steps),
            "--buffer_size", str(self.buffer_size),
            "--lr", str(self.learning_rate),
            "--tau", str(self.tau),
            "--alpha", str(self.alpha),
            "--lambda_dyn", str(self.lambda_dyn),
            "--task_scale", str(self.task_scale),
            "--nullspace_scale", str(self.nullspace_scale),
            "--hidden_dims", ",".join(str(d) for d in self.hidden_dims),
            "--n_critics", str(self.n_critics),
        ]


ALGORITHM = _ALGORITHM()


# ── Environment ──────────────────────────────────────────────────────────

@dataclass
class _ENVIRONMENT:
    dt: float = 0.02
    episode_len: int = 500
    trajectory_steps: int = 350
    tracking_full_speed_error: float = 0.03
    tracking_stop_error: float = 0.08
    success_tolerance: float = 0.05
    success_hold_steps: int = 10
    w_track: float = 12.0
    w_obs: float = 5.0
    w_collision: float = 100.0
    collision_event_penalty: float = 500.0   # v4: MuJoCo 碰撞额外 -500
    w_manip: float = 0.0
    w_energy: float = 0.0
    w_action: float = 0.0
    d_safe: float = 0.06
    success_bonus: float = 100.0


ENVIRONMENT = _ENVIRONMENT()

# ── Protocol version (stored in checkpoints to prevent cross-protocol resume) ──
TRAINING_PROTOCOL_VERSION = 4


# ── Evaluation ───────────────────────────────────────────────────────────

@dataclass
class _EVALUATION:
    train_seeds: tuple = (11, 23, 37, 53, 71)
    episodes_per_seed: int = 100


EVALUATION = _EVALUATION()


# ── Methods (ordered for paper table) ────────────────────────────────────

PHASE1_METHODS = [
    "pd",
    "gradient_projection",
    "cbf_qp",
    "sac_joint",
    "sac_residual",
    "ours_no_physics",
    "ours_physics",
    "ours_full",
]

PHASE1_SCENARIOS = [
    "free_space",
    "whole_body",
    "confined_space",
    "generalization",
]


# ── Reward / cost defaults ───────────────────────────────────────────────

def phase1_defaults() -> dict:
    return {
        # ---- reward weights ----
        "w_track": 12.0,
        "w_obs": 5.0,
        "w_collision": 100.0,
        "collision_termination_penalty": -500.0,  # v4 新增
        "w_manip": 0.0,
        "w_energy": 0.0,
        "w_action": 0.0,
        "success_bonus": 100.0,
        # ---- safety ----
        "d_safe": 0.06,
        "cost_scale": 1.0,
        "cost_limit": 0.1,
        "no_collision_term": False,
        # ---- reward shaping ----
        "reward_scale": 1.0,
        "reward_min": -10.0,
        # ---- paths ----
        "scene_root": str(Path("results/ewalker_scenes")),
        "checkpoint_root": str(Path("/root/autodl-tmp/manipulator/checkpoints/phase1")),
        "result_root": str(Path("results/phase1")),
    }
