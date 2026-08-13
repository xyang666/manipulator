#!/usr/bin/env python3
"""
generate_dynamic_scenes.py
--------------------------
Convert certified static scenes (whole_body / confined_space) into dynamic
obstacle scenes. Obstacles get a velocity perpendicular to the straight-line
path, with bounce bounds around their certified position, so the certified
path remains traversable when timed correctly.

Output schema (backward compatible):
    obstacles: [[x, y, z, r, vx, vy, vz], ...]   # 7 elements = dynamic
    obstacle_bounds: [[lo_x, lo_y, lo_z, hi_x, hi_y, hi_z], ...]

Usage:
    code/.venv/bin/python -m experiments.generate_dynamic_scenes \
        --input-dir results/ewalker_scenes \
        --output-dir results/ewalker_scenes/dynamic \
        --speed 0.05 0.15 --swing 0.06 --n-moving 1
"""

import argparse
import json
import numpy as np
from pathlib import Path


def _add_motion(scene, rng, speed_range, swing, n_moving, mode="random"):
    """Add velocity + bounds to obstacles of a certified scene.

    mode="random":   up to n_moving random obstacles move along the sweep axis
    mode="squeeze":  confined-space corridor walls close/open alternately —
                     pairs at ±offset move toward/away from each other so the
                     repulsion forces on the arm cancel at the corridor mouth
    """
    obstacles = scene["obstacles"]
    n = len(obstacles)
    if n == 0:
        return scene

    start = np.asarray(scene["start"], dtype=float)
    goal = np.asarray(scene["goal"], dtype=float)
    path_dir = goal - start
    norm = np.linalg.norm(path_dir)
    if norm > 1e-6:
        path_dir /= norm
    # Perpendicular sweep axis: cross path with world-Z, fall back to X
    sweep = np.cross(path_dir, [0.0, 0.0, 1.0])
    if np.linalg.norm(sweep) < 1e-6:
        sweep = np.array([1.0, 0.0, 0.0])
    sweep /= np.linalg.norm(sweep)

    out_obstacles = [list(o) for o in obstacles]
    bounds = [None] * n

    def _set_velocity(i, velocity):
        o = out_obstacles[i]
        half = float(swing)
        center = np.asarray(o[:3], dtype=float)
        lo = center - half * np.abs(sweep)
        hi = center + half * np.abs(sweep)
        lo = np.maximum(lo, [-0.8, -0.8, 0.1])
        hi = np.minimum(hi, [0.8, 0.8, 1.25])
        out_obstacles[i] = list(o) + velocity.tolist()
        bounds[i] = [lo.tolist(), hi.tolist()]

    if mode == "squeeze" and len(obstacles) >= 4:
        # Corridor walls: obstacles appear in ±offset pairs along the sweep
        # axis (x for corridors). Pair them by |sweep-coordinate|, pick the
        # pair closest to the path midpoint, and drive them toward/away.
        sweep_coord = np.array([np.dot(np.asarray(o[:3]), sweep)
                                for o in obstacles])
        order = np.argsort(sweep_coord)
        # Two extreme pairs = left/right walls
        left_pair = order[:2]
        right_pair = order[-2:]
        pair = (list(left_pair) + list(right_pair))[:n_moving * 2 if n_moving > 1 else 2]
        mid = 0.5 * (sweep_coord[left_pair].mean() + sweep_coord[right_pair].mean())
        for i in left_pair:
            _set_velocity(int(i), sweep * float(rng.uniform(*speed_range)))
        for i in right_pair:
            _set_velocity(int(i), -sweep * float(rng.uniform(*speed_range)))
        if n_moving > 1 and len(obstacles) >= 6:
            mid_left = order[len(order)//2 - 1]
            mid_right = order[len(order)//2]
            _set_velocity(int(mid_left), -sweep * float(rng.uniform(*speed_range)))
            _set_velocity(int(mid_right), sweep * float(rng.uniform(*speed_range)))
    else:
        # Prefer the PD blocker (last obstacle) then spread evenly
        indices = list(range(n))
        if n > 1:
            indices = [n - 1] + indices[:-1]  # last = PD blocker first
        chosen = indices[:n_moving]
        for i in chosen:
            speed = float(rng.uniform(*speed_range))
            direction = sweep * float(rng.choice([-1.0, 1.0]))
            _set_velocity(i, direction * speed)

    scene = dict(scene)
    scene["obstacles"] = out_obstacles
    scene["obstacle_bounds"] = [
        (bounds[i] if bounds[i] is not None else
         [list(o[:3]), list(np.asarray(o[:3], dtype=float) + 0.001)])
        for i, o in enumerate(out_obstacles)
    ]
    scene["dynamic"] = True
    scene["dynamic_mode"] = mode
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=Path("results/ewalker_scenes"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/ewalker_scenes/dynamic"))
    parser.add_argument("--speed", type=float, nargs=2, default=[0.05, 0.15],
                        help="obstacle speed range (m/s)")
    parser.add_argument("--swing", type=float, default=0.06,
                        help="bounce half-width around certified position (m)")
    parser.add_argument("--n-moving", type=int, default=1,
                        help="number of obstacles to move per scene")
    parser.add_argument("--mode", type=str, default="random",
                        choices=["random", "squeeze"],
                        help="motion pattern: random sweep or corridor squeeze")
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for kind in ["whole_body", "confined_space"]:
        for split in ["train", "validation", "test"]:
            src = args.input_dir / kind / f"{split}.json"
            if not src.exists():
                print(f"[skip] {src}")
                continue
            with open(src) as f:
                scenes = json.load(f)
            dynamic = [_add_motion(dict(s), rng, args.speed, args.swing,
                                   args.n_moving, mode=args.mode)
                       for s in scenes]
            dst = args.output_dir / f"{kind}_{split}.json"
            with open(dst, "w") as f:
                json.dump(dynamic, f, indent=2)
            n_dyn = sum(1 for s in dynamic if s.get("dynamic"))
            print(f"{dst}: {len(dynamic)} scenes ({n_dyn} dynamic)")
    print("done")


if __name__ == "__main__":
    main()
