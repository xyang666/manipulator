#!/usr/bin/env python3
"""Generate planner-certified non-local detour scenes for an RL pilot.

Acceptance uses geometry and a seeded RRT-Connect oracle only.  It never runs
PD, Gradient, CBF, or an RL policy.  The oracle path is stored for auditing but
must not be included in policy observations or controller references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from env.kinematics import ManipulatorKinematics
from planner.rrt_connect import RRTConnect
from planner.rrt_star import capsule_sphere_distance
from robot_config import DEFAULT_URDF


def _normalized_length(path: list[np.ndarray], joint_range: np.ndarray) -> float:
    return float(sum(np.linalg.norm((b - a) / joint_range)
                     for a, b in zip(path[:-1], path[1:])))


def _task_deviation(kin, path: list[np.ndarray], start: np.ndarray,
                    goal: np.ndarray) -> float:
    delta = goal - start
    denom = max(float(np.dot(delta, delta)), 1e-12)
    maximum = 0.0
    for a, b in zip(path[:-1], path[1:]):
        for fraction in np.linspace(0.0, 1.0, 8):
            q = (1.0 - fraction) * a + fraction * b
            position, _ = kin.forward_kinematics(q)
            progress = np.clip(np.dot(position - start, delta) / denom, 0.0, 1.0)
            closest = start + progress * delta
            maximum = max(maximum, float(np.linalg.norm(position - closest)))
    return maximum


def _configuration_clearance(kin, q: np.ndarray, obstacles: list[list[float]]) -> float:
    return min(
        capsule_sphere_distance(p1, p2, radius,
                                np.asarray(obstacle[:3]), obstacle[3])
        for p1, p2, radius in kin.get_link_capsules(q)
        for obstacle in obstacles
    )


def _candidate_blocker(kin, q: np.ndarray, capsule_index: int,
                       rng: np.random.Generator) -> list[float] | None:
    p1, p2, capsule_radius = kin.get_link_capsules(q)[capsule_index]
    axis = p2 - p1
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1e-9:
        return None
    axis /= axis_norm
    alpha = float(rng.uniform(0.25, 0.75))
    point = (1.0 - alpha) * p1 + alpha * p2
    # Draw one direction normal to the capsule.  Both signs and arbitrary
    # azimuths occur across seeds, preventing a fixed left/right solution.
    direction = rng.normal(size=3)
    direction -= np.dot(direction, axis) * axis
    norm = np.linalg.norm(direction)
    if norm <= 1e-9:
        return None
    radius = float(rng.uniform(0.018, 0.032))
    penetration = float(rng.uniform(0.002, 0.006))
    center = point + direction / norm * (capsule_radius + radius - penetration)
    if not (-0.85 <= center[0] <= 0.85 and -0.85 <= center[1] <= 0.85
            and 0.05 <= center[2] <= 1.30):
        return None
    return [*center.tolist(), radius]


def make_detour_scene(scene: dict, kin, seed: int, attempts: int,
                      minimum_ratio: float, maximum_ratio: float,
                      minimum_deviation: float, maximum_deviation: float,
                      clearance: float) -> dict | None:
    rng = np.random.default_rng(seed)
    q_start = np.asarray(scene["start_q"], dtype=float)
    q_goal = np.asarray(scene["goal_q"], dtype=float)
    start = np.asarray(scene["start"], dtype=float)
    goal = np.asarray(scene["goal"], dtype=float)
    joint_range = kin.q_max - kin.q_min
    direct_length = np.linalg.norm((q_goal - q_start) / joint_range)
    capsules_start = kin.get_link_capsules(q_start)
    capsules_goal = kin.get_link_capsules(q_goal)
    moving = [
        index for index in range(min(len(capsules_start), len(capsules_goal)) - 3)
        if np.linalg.norm(0.5 * (capsules_start[index][0] + capsules_start[index][1])
                          - 0.5 * (capsules_goal[index][0] + capsules_goal[index][1]))
        >= 0.04
    ]
    if not moving:
        return None

    for attempt in range(attempts):
        fraction = float(rng.uniform(0.3, 0.7))
        q_mid = (1.0 - fraction) * q_start + fraction * q_goal
        blocker = _candidate_blocker(kin, q_mid, int(rng.choice(moving)), rng)
        if blocker is None:
            continue
        obstacles = [list(item[:4]) for item in scene["obstacles"]] + [blocker]
        if min(_configuration_clearance(kin, q_start, obstacles),
               _configuration_clearance(kin, q_goal, obstacles)) < clearance:
            continue
        planner = RRTConnect(
            kin, kin.q_min, kin.q_max, obstacles, seed=seed + attempt,
            max_iterations=900, step_size=0.14, goal_bias=0.15,
            clearance=clearance, n_interpolation_steps=16,
        )
        if not planner._segment_collision(q_start, q_goal):
            continue
        path, elapsed, nodes = planner.plan(q_start, q_goal)
        if not path:
            continue
        ratio = _normalized_length(path, joint_range) / max(direct_length, 1e-9)
        deviation = _task_deviation(kin, path, start, goal)
        if not (minimum_ratio <= ratio <= maximum_ratio
                and minimum_deviation <= deviation <= maximum_deviation):
            continue
        result = dict(scene)
        digest = hashlib.sha256(json.dumps({
            "source": scene["scene_id"], "blocker": blocker, "seed": seed,
        }, sort_keys=True).encode()).hexdigest()[:12]
        result.update({
            "scene_id": f"rl-detour-{digest}",
            "scenario": "rl_challenge_detour",
            "challenge_type": "rrt_detour",
            "challenge_distribution": True,
            "source_scene_id": scene["scene_id"],
            "obstacles": obstacles,
            "direct_joint_path_collision": True,
            "rrt_connect_feasible": True,
            "rrt_detour_ratio": ratio,
            "rrt_max_task_deviation_m": deviation,
            "rrt_planning_time_s": elapsed,
            "rrt_nodes": nodes,
            "feasible_q_path": [q.tolist() for q in path],
            "oracle": "seeded_joint_space_rrt_connect",
        })
        return result
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("results/ewalker_scenes/whole_body/validation.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/ewalker_scenes/rl_detour_pilot.json"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--min-ratio", type=float, default=1.20)
    parser.add_argument("--max-ratio", type=float, default=3.0)
    parser.add_argument("--min-deviation", type=float, default=0.06)
    parser.add_argument("--max-deviation", type=float, default=0.20)
    parser.add_argument("--clearance", type=float, default=0.02)
    args = parser.parse_args()
    scenes = json.loads(args.input.read_text())
    kin = ManipulatorKinematics(DEFAULT_URDF, 7)
    output = []
    for index, scene in enumerate(scenes):
        candidate = make_detour_scene(
            scene, kin, args.seed + 10007 * index, args.attempts,
            args.min_ratio, args.max_ratio, args.min_deviation,
            args.max_deviation, args.clearance,
        )
        if candidate is not None:
            output.append(candidate)
            print(f"accepted {len(output)}/{args.count}: {candidate['scene_id']} "
                  f"ratio={candidate['rrt_detour_ratio']:.2f} "
                  f"deviation={candidate['rrt_max_task_deviation_m']:.3f}", flush=True)
        if len(output) >= args.count:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {len(output)} scenes to {args.output}")
    return 0 if len(output) == args.count else 2


if __name__ == "__main__":
    raise SystemExit(main())
