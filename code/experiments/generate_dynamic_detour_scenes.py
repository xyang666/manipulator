#!/usr/bin/env python3
"""Turn planner-certified static detours into predictive dynamic detours.

The added blocker starts away from its certified conflict pose, reaches that
pose at the nominal direct-path arrival time, then reflects and reopens.  Scene
acceptance uses only endpoint geometry plus the RRT certificate already stored
in the source scene; no controller is evaluated during generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env.kinematics import ManipulatorKinematics
from experiments.generate_rl_detour_pilot import _configuration_clearance
from robot_config import DEFAULT_URDF


def _static_bounds(obstacle: list[float]) -> list[list[float]]:
    center = np.asarray(obstacle[:3], dtype=float)
    return [center.tolist(), (center + 1e-6).tolist()]


def _conflict_fraction(scene: dict, kin, blockers: list[list[float]]) -> float:
    q0 = np.asarray(scene["start_q"], dtype=float)
    q1 = np.asarray(scene["goal_q"], dtype=float)
    fractions = np.linspace(0.15, 0.85, 71)
    clearances = [
        _configuration_clearance(kin, (1.0 - f) * q0 + f * q1, blockers)
        for f in fractions
    ]
    return float(fractions[int(np.argmin(clearances))])


def make_dynamic_detour(scene: dict, kin, rng: np.random.Generator,
                        swing_range: tuple[float, float], duration: float,
                        endpoint_clearance: float) -> dict | None:
    """Create one transient blocker motion without changing its RRT oracle."""
    if not scene.get("rrt_connect_feasible"):
        return None
    added = int(scene.get("added_blocker_count", 0))
    if added <= 0:
        return None
    obstacles = [list(obstacle[:4]) for obstacle in scene["obstacles"]]
    moving = list(range(len(obstacles) - added, len(obstacles)))
    blockers = [obstacles[index] for index in moving]
    fraction = _conflict_fraction(scene, kin, blockers)
    conflict_time = max(0.75, fraction * duration)

    q_start = np.asarray(scene["start_q"], dtype=float)
    q_goal = np.asarray(scene["goal_q"], dtype=float)
    for _ in range(32):
        candidate = [list(obstacle) for obstacle in obstacles]
        bounds = [_static_bounds(obstacle) for obstacle in obstacles]
        velocities = []
        for index in moving:
            target = np.asarray(obstacles[index][:3], dtype=float)
            direction = rng.normal(size=3)
            direction /= max(np.linalg.norm(direction), 1e-12)
            swing = float(rng.uniform(*swing_range))
            initial = target + swing * direction
            velocity = (target - initial) / conflict_time
            candidate[index] = initial.tolist() + [obstacles[index][3]]
            bounds[index] = [
                np.minimum(initial, target).tolist(),
                np.maximum(initial, target).tolist(),
            ]
            velocities.append(velocity)
        if min(_configuration_clearance(kin, q_start, candidate),
               _configuration_clearance(kin, q_goal, candidate)) < endpoint_clearance:
            continue
        result = dict(scene)
        dynamic_obstacles = []
        velocity_by_index = dict(zip(moving, velocities))
        for index, obstacle in enumerate(candidate):
            velocity = velocity_by_index.get(index, np.zeros(3))
            dynamic_obstacles.append(obstacle + velocity.tolist())
        result.update({
            "scene_id": f"dynamic-{scene['scene_id']}",
            "scenario": "rl_challenge_detour",
            "challenge_type": "predictive_dynamic_detour",
            "dynamic": True,
            "obstacles": dynamic_obstacles,
            "obstacle_bounds": bounds,
            "moving_obstacle_indices": moving,
            "nominal_conflict_fraction": fraction,
            "nominal_conflict_time_s": conflict_time,
            "dynamic_endpoint_clearance_m": min(
                _configuration_clearance(kin, q_start, candidate),
                _configuration_clearance(kin, q_goal, candidate)),
            "dynamic_generation_controller_conditioned": False,
        })
        return result
    return None


def generate(source: Path, output: Path, seed: int,
             swing_range: tuple[float, float], duration: float,
             endpoint_clearance: float, urdf: str = DEFAULT_URDF) -> dict:
    scenes = json.loads(source.read_text())
    kin = ManipulatorKinematics(urdf, 7)
    rng = np.random.default_rng(seed)
    converted = []
    for scene in scenes:
        result = make_dynamic_detour(
            scene, kin, rng, swing_range, duration, endpoint_clearance)
        if result is not None:
            converted.append(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(converted, indent=2) + "\n")
    manifest = {
        "protocol": "predictive_dynamic_detour_v1",
        "controller_conditioned_selection": False,
        "source": str(source), "seed": seed,
        "source_count": len(scenes), "output_count": len(converted),
        "swing_range_m": list(swing_range),
        "nominal_duration_s": duration,
        "min_endpoint_clearance_m": endpoint_clearance,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--swing-range", nargs=2, type=float,
                        default=(0.08, 0.14), metavar=("MIN", "MAX"))
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--endpoint-clearance", type=float, default=0.01)
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    args = parser.parse_args()
    if not 0 < args.swing_range[0] <= args.swing_range[1]:
        parser.error("--swing-range must satisfy 0 < MIN <= MAX")
    if args.duration <= 0 or args.endpoint_clearance < 0:
        parser.error("duration must be positive and clearance non-negative")
    manifest = generate(args.source, args.output, args.seed,
                        tuple(args.swing_range), args.duration,
                        args.endpoint_clearance, args.urdf)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
