"""Canonical episode metrics used by every paper baseline and ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from statistics import NormalDist
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    method: str
    scenario: str
    seed: int
    scene_id: int
    success: bool
    collision: bool
    completion_time_s: float
    tracking_rms_m: float
    tracking_peak_m: float
    min_clearance_m: float
    joint_velocity_smoothness: float
    torque_rate_nm_s: float
    energy: float
    nullspace_utilization: float
    reference_correction: float
    gate_trigger_rate: float
    torque_violation_count: int
    steps: int

    def to_dict(self) -> dict:
        return asdict(self)


class EpisodeRecorder:
    """Accumulate per-step signals and produce one canonical result row."""

    def __init__(self, method: str, scenario: str, seed: int, scene_id: int,
                 dt: float, torque_limits: np.ndarray | None = None):
        self.method = method
        self.scenario = scenario
        self.seed = seed
        self.scene_id = scene_id
        self.dt = dt
        self.torque_limits = torque_limits
        self.tracking_errors: list[float] = []
        self.clearances: list[float] = []
        self.dq: list[np.ndarray] = []
        self.torques: list[np.ndarray] = []
        self.nullspace_norms: list[float] = []
        self.reference_norms: list[float] = []
        self.gates: list[float] = []
        self.collided = False
        self.succeeded = False

    def add(self, *, info: dict, dq: np.ndarray, torque: np.ndarray | None = None,
            nullspace_norm: float = 0.0, reference_norm: float = 0.0,
            gate: float = 0.0) -> None:
        self.tracking_errors.append(float(info.get("tracking_error", 0.0)))
        self.clearances.append(float(info.get("d_obs", np.inf)))
        self.dq.append(np.asarray(dq, dtype=float).copy())
        if torque is not None:
            self.torques.append(np.asarray(torque, dtype=float).copy())
        self.nullspace_norms.append(float(nullspace_norm))
        self.reference_norms.append(float(reference_norm))
        self.gates.append(float(gate))
        self.collided = self.collided or bool(info.get("collision", False))
        self.succeeded = self.succeeded or bool(info.get("success", False))

    def finish(self) -> EpisodeMetrics:
        errors = np.asarray(self.tracking_errors, dtype=float)
        velocities = np.asarray(self.dq, dtype=float)
        torques = np.asarray(self.torques, dtype=float)
        velocity_delta = np.diff(velocities, axis=0) if len(velocities) > 1 else np.empty((0,))
        torque_delta = np.diff(torques, axis=0) if len(torques) > 1 else np.empty((0,))
        violations = 0
        if len(torques) and self.torque_limits is not None:
            violations = int(np.count_nonzero(np.abs(torques) > self.torque_limits))
        return EpisodeMetrics(
            method=self.method,
            scenario=self.scenario,
            seed=self.seed,
            scene_id=self.scene_id,
            success=self.succeeded and not self.collided,
            collision=self.collided,
            completion_time_s=len(errors) * self.dt,
            tracking_rms_m=float(np.sqrt(np.mean(errors ** 2))) if len(errors) else 0.0,
            tracking_peak_m=float(np.max(errors)) if len(errors) else 0.0,
            min_clearance_m=float(np.min(self.clearances)) if self.clearances else float("inf"),
            joint_velocity_smoothness=_mean_norm(velocity_delta) / self.dt,
            torque_rate_nm_s=_mean_norm(torque_delta) / self.dt,
            energy=float(np.sum(velocities ** 2) * self.dt) if len(velocities) else 0.0,
            nullspace_utilization=_mean(self.nullspace_norms),
            reference_correction=_mean(self.reference_norms),
            gate_trigger_rate=float(np.mean(np.asarray(self.gates) > 1e-6)) if self.gates else 0.0,
            torque_violation_count=violations,
            steps=len(errors),
        )


def summarize_episodes(rows: Iterable[EpisodeMetrics | dict], confidence: float = 0.95) -> dict:
    """Compute mean, sample standard deviation, and normal 95% CI per metric."""
    records = [r.to_dict() if isinstance(r, EpisodeMetrics) else dict(r) for r in rows]
    if not records:
        raise ValueError("cannot summarize an empty result set")
    summary = {"n": len(records), "method": records[0]["method"],
               "scenario": records[0]["scenario"], "metrics": {}}
    metric_names = [
        "success", "collision", "completion_time_s", "tracking_rms_m",
        "tracking_peak_m", "min_clearance_m", "joint_velocity_smoothness",
        "torque_rate_nm_s", "energy", "nullspace_utilization",
        "reference_correction", "gate_trigger_rate", "torque_violation_count",
    ]
    z_score = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    for name in metric_names:
        values = np.asarray([float(r[name]) for r in records], dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        half_width = z_score * std / sqrt(len(values)) if len(values) > 1 else 0.0
        summary["metrics"][name] = {
            "mean": mean, "std": std,
            "ci_low": mean - half_width, "ci_high": mean + half_width,
        }
    return summary


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _mean_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(values, axis=-1)))
