#!/usr/bin/env python3
"""Generate a transparent dynamic challenge set for predictive RL control.

The challenge distribution is intentionally aimed at failure modes of purely
instantaneous reactive controllers.  It must therefore be reported separately
from the standard benchmark, never as an unbiased replacement for it.

Two families are derived from the planner-certified phase-one scenes:

* ``timed_crossing``: one obstacle reaches its certified blocking position at
  approximately the nominal arm-arrival time;
* ``closing_gate``: a left/right corridor pair closes near nominal mid-path
  arrival and later reopens because the environment reflects it at its bounds.

The source train/validation/test split is preserved.  No tested controller is
run during generation and no scene is accepted or rejected by method outcome.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env.kinematics import ManipulatorKinematics
from planner.rrt_star import capsule_sphere_distance
from robot_config import DEFAULT_URDF


SPLITS = ("train", "validation", "test")


def _unit_perpendicular(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    direction = goal - start
    norm = np.linalg.norm(direction)
    if norm <= 1e-9:
        raise ValueError("challenge scene start and goal must differ")
    direction /= norm
    sweep = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(sweep) <= 1e-9:
        sweep = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    return sweep / np.linalg.norm(sweep)


def _path_fraction(point: np.ndarray, start: np.ndarray,
                   goal: np.ndarray) -> float:
    delta = goal - start
    return float(np.clip(np.dot(point - start, delta) /
                         max(np.dot(delta, delta), 1e-12), 0.15, 0.85))


def _segment_clearance(obstacle: list[float], start: np.ndarray,
                       goal: np.ndarray) -> float:
    point = np.asarray(obstacle[:3], dtype=float)
    fraction = _path_fraction(point, start, goal)
    closest = start + fraction * (goal - start)
    return float(np.linalg.norm(point - closest) - obstacle[3])


def _static_bounds(obstacle: list[float]) -> list[list[float]]:
    center = np.asarray(obstacle[:3], dtype=float)
    return [center.tolist(), (center + 1e-6).tolist()]


def endpoint_clearance(scene: dict, kin) -> float:
    """Whole-arm endpoint clearance over every obstacle's complete sweep.

    Dynamic scenes must remain feasible while the arm waits at either task
    endpoint. Checking only the initial obstacle centers can retain a closing
    gate whose later sweep physically intersects the stationary start pose.
    """
    bounds = scene.get("obstacle_bounds", [])
    obstacles = []
    for index, obstacle in enumerate(scene["obstacles"]):
        centers = [obstacle[:3]]
        if index < len(bounds) and len(bounds[index]) == 2:
            centers.extend(bounds[index])
        obstacles.extend([list(center) + [obstacle[3]] for center in centers])
    minimum = float("inf")
    for key in ("start_q", "goal_q"):
        for p1, p2, radius in kin.get_link_capsules(
                np.asarray(scene[key], dtype=float)):
            for obstacle in obstacles:
                minimum = min(minimum, capsule_sphere_distance(
                    p1, p2, radius, np.asarray(obstacle[:3]), obstacle[3]))
    return float(minimum)


def make_timed_crossing(scene: dict, rng: np.random.Generator,
                        swing: float, duration: float) -> dict:
    """Move the path-nearest obstacle through its certified blocking pose."""
    result = dict(scene)
    start = np.asarray(scene["start"], dtype=float)
    goal = np.asarray(scene["goal"], dtype=float)
    sweep = _unit_perpendicular(start, goal)
    obstacles = [list(obstacle[:4]) for obstacle in scene["obstacles"]]
    blocker = min(range(len(obstacles)), key=lambda index: _segment_clearance(
        obstacles[index], start, goal))
    center = np.asarray(obstacles[blocker][:3], dtype=float)
    fraction = _path_fraction(center, start, goal)
    conflict_time = max(0.75, fraction * duration)
    side = float(rng.choice((-1.0, 1.0)))
    initial = center + side * swing * sweep
    velocity = -side * (swing / conflict_time) * sweep
    lo = center - swing * np.abs(sweep)
    hi = center + swing * np.abs(sweep)

    bounds = [_static_bounds(obstacle) for obstacle in obstacles]
    obstacles[blocker] = initial.tolist() + [obstacles[blocker][3]] + velocity.tolist()
    bounds[blocker] = [lo.tolist(), hi.tolist()]
    result.update({
        "scene_id": str(scene["scene_id"]).replace("whole_body", "rl-crossing"),
        "scenario": "rl_challenge_crossing",
        "challenge_type": "timed_crossing",
        "dynamic": True,
        "obstacles": obstacles,
        "obstacle_bounds": bounds,
        "source_scene_id": scene["scene_id"],
        "nominal_conflict_time_s": conflict_time,
        "moving_obstacle_indices": [blocker],
        "challenge_distribution": True,
    })
    return result


def make_closing_gate(scene: dict, swing: float, duration: float) -> dict:
    """Close the corridor pair nearest the nominal path midpoint."""
    result = dict(scene)
    start = np.asarray(scene["start"], dtype=float)
    goal = np.asarray(scene["goal"], dtype=float)
    midpoint = 0.5 * (start + goal)
    sweep = _unit_perpendicular(start, goal)
    obstacles = [list(obstacle[:4]) for obstacle in scene["obstacles"]]
    coordinates = np.array([
        np.dot(np.asarray(obstacle[:3]) - midpoint, sweep)
        for obstacle in obstacles
    ])
    negative = np.where(coordinates < 0.0)[0]
    positive = np.where(coordinates > 0.0)[0]
    if not len(negative) or not len(positive):
        raise ValueError(f"scene {scene['scene_id']} has no two-sided corridor")
    # Among each wall, select the obstacle closest to the path midpoint along
    # the travel direction.  This creates one localized, time-dependent gate.
    path = (goal - start) / np.linalg.norm(goal - start)
    along = np.abs(np.array([
        np.dot(np.asarray(obstacle[:3]) - midpoint, path)
        for obstacle in obstacles
    ]))
    left = int(negative[np.argmin(along[negative])])
    right = int(positive[np.argmin(along[positive])])
    close_time = 0.5 * duration
    speed = swing / max(close_time, 0.75)
    bounds = [_static_bounds(obstacle) for obstacle in obstacles]
    for index in (left, right):
        center = np.asarray(obstacles[index][:3], dtype=float)
        inward = -np.sign(coordinates[index]) * sweep
        velocity = speed * inward
        lo = center - swing * np.abs(sweep)
        hi = center + swing * np.abs(sweep)
        obstacles[index] = center.tolist() + [obstacles[index][3]] + velocity.tolist()
        bounds[index] = [lo.tolist(), hi.tolist()]

    result.update({
        "scene_id": str(scene["scene_id"]).replace("confined_space", "rl-gate"),
        "scenario": "rl_challenge_gate",
        "challenge_type": "closing_gate",
        "dynamic": True,
        "obstacles": obstacles,
        "obstacle_bounds": bounds,
        "source_scene_id": scene["scene_id"],
        "nominal_conflict_time_s": close_time,
        "moving_obstacle_indices": [left, right],
        "challenge_distribution": True,
    })
    return result


def generate(input_dir: Path, output_dir: Path, seed: int, swing: float,
             trajectory_steps: int, dt: float,
             min_endpoint_clearance: float = 0.0,
             urdf: str = DEFAULT_URDF) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = trajectory_steps * dt
    counts: dict[str, int] = {}
    rejected: dict[str, int] = {}
    kin = (ManipulatorKinematics(urdf, 7)
           if min_endpoint_clearance > 0.0 else None)
    for split in SPLITS:
        with (input_dir / "whole_body" / f"{split}.json").open() as stream:
            whole_body = json.load(stream)
        with (input_dir / "confined_space" / f"{split}.json").open() as stream:
            confined = json.load(stream)
        scenes = [make_timed_crossing(scene, rng, swing, duration)
                  for scene in whole_body]
        scenes.extend(make_closing_gate(scene, swing, duration)
                      for scene in confined)
        before = len(scenes)
        if kin is not None:
            scenes = [scene for scene in scenes
                      if endpoint_clearance(scene, kin)
                      >= min_endpoint_clearance]
        rejected[split] = before - len(scenes)
        destination = output_dir / f"{split}.json"
        destination.write_text(json.dumps(scenes, indent=2) + "\n")
        counts[split] = len(scenes)
    manifest = {
        "protocol": "rl_challenge_v1",
        "purpose": "predictive-RL challenge; report separately from standard benchmark",
        "seed": seed,
        "swing_m": swing,
        "trajectory_steps": trajectory_steps,
        "dt_s": dt,
        "counts": counts,
        "rejected_endpoint_infeasible": rejected,
        "min_endpoint_clearance_m": min_endpoint_clearance,
        "families": ["timed_crossing", "closing_gate"],
        "source": str(input_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=Path("results/ewalker_scenes"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/ewalker_scenes/rl_challenge_v1"))
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--swing", type=float, default=0.05)
    parser.add_argument("--trajectory-steps", type=int, default=350)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--min-endpoint-clearance", type=float, default=0.0,
                        help="reject transformed scenes whose whole arm is too close at start/goal")
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    args = parser.parse_args()
    if args.swing <= 0.0 or args.trajectory_steps <= 0 or args.dt <= 0.0:
        parser.error("swing, trajectory-steps and dt must be positive")
    if args.min_endpoint_clearance < 0.0:
        parser.error("--min-endpoint-clearance must be non-negative")
    counts = generate(args.input_dir, args.output_dir, args.seed, args.swing,
                      args.trajectory_steps, args.dt,
                      args.min_endpoint_clearance, args.urdf)
    print("generated", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
