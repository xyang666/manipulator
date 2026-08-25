#!/usr/bin/env python3
"""Roll out the Adaptive Gradient-CBF baseline into a BC demonstration file.

Only training scenes may be passed to this script.  Validation and test scenes
remain untouched.  The stored action is the raw structured controller command;
the CBF remains an environment safety layer during both teacher rollout and RL.
By default, samples from unsuccessful episodes are discarded.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from env.manipulator_env import ManipulatorEnv
from experiments.runner import _gradient_projection_action
from robot_config import DEFAULT_URDF, DEFAULT_XML
from utils.validation import ValidationSet

_WORKER_ENV = None
_WORKER_VALIDATION = None
_WORKER_CONFIG = None


def collect_episode(env, validation, scene, scale, smoothing, steps):
    env.reset(seed=int(scene.get("seed", 0)))
    validation.apply_scene_to_env(env, scene)
    observation = env._get_obs()
    previous = np.zeros(env.act_dim, dtype=float)
    states, actions = [], []
    success = False
    for _ in range(steps):
        raw = _gradient_projection_action(env, scale=scale)
        action = smoothing * previous + (1.0 - smoothing) * raw
        states.append(observation.astype(np.float32))
        actions.append(action.astype(np.float32))
        observation, _, done, info = env.step(action)
        previous = action
        success = success or bool(info.get("success", False))
        if done:
            break
    return states, actions, success


def _init_worker(input_path, env_kwargs, scale, smoothing, steps):
    global _WORKER_ENV, _WORKER_VALIDATION, _WORKER_CONFIG
    # Avoid BLAS/OpenMP oversubscription when several simulator workers run.
    os.environ["OMP_NUM_THREADS"] = "1"
    _WORKER_VALIDATION = ValidationSet(input_path)
    _WORKER_ENV = ManipulatorEnv(**env_kwargs)
    _WORKER_CONFIG = (scale, smoothing, steps)


def _collect_worker(scene):
    scale, smoothing, steps = _WORKER_CONFIG
    states, actions, success = collect_episode(
        _WORKER_ENV, _WORKER_VALIDATION, scene, scale, smoothing, steps
    )
    return scene["scene_id"], states, actions, success


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Training-scene JSON (never validation/test)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--gradient-scale", type=float, default=0.3)
    parser.add_argument("--smoothing", type=float, default=0.8)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--obs-scene-embed", type=int, default=10)
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--xml", default=DEFAULT_XML)
    args = parser.parse_args()
    if (args.steps <= 0 or args.max_samples <= 0 or args.gradient_scale <= 0
            or args.workers <= 0):
        raise ValueError("steps, max-samples and gradient-scale must be positive")
    if not 0.0 <= args.smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")

    validation = ValidationSet(str(args.input))
    scenes = validation.scenes
    if not scenes:
        raise ValueError("training scene set is empty")
    env_kwargs = dict(
        urdf_path=args.urdf, xml_path=args.xml, episode_len=args.steps,
        trajectory_steps=args.steps,
        n_obstacles=max(len(scene["obstacles"]) for scene in scenes),
        use_trajectory_generator=False, use_cbf=True,
        cbf_self_d_safe=0.02, cbf_multi_self_constraints=True,
        gate_enabled=False, obs_scene_embed=args.obs_scene_embed,
        obs_waypoint_steps=[10, 25, 50],
    )
    all_states, all_actions, all_scene_ids = [], [], []
    successes = 0
    if args.workers == 1:
        env = ManipulatorEnv(**env_kwargs)
        results = (
            (scene["scene_id"], *collect_episode(
                env, validation, scene, args.gradient_scale, args.smoothing,
                args.steps))
            for scene in scenes
        )
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(
            args.workers, initializer=_init_worker,
            initargs=(str(args.input), env_kwargs, args.gradient_scale,
                      args.smoothing, args.steps),
        )
        results = pool.imap(_collect_worker, scenes)
    for scene_id, states, actions, success in results:
        successes += int(success)
        if success or args.keep_failures:
            remaining = args.max_samples - len(all_states)
            all_states.extend(states[:remaining])
            all_actions.extend(actions[:remaining])
            all_scene_ids.extend([scene_id] * min(len(states), remaining))
        if len(all_states) >= args.max_samples:
            break
    if pool is not None:
        pool.close()
        pool.join()
    if not all_states:
        raise RuntimeError("teacher produced no retained demonstrations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, states=np.asarray(all_states, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.float32),
        scene_ids=np.asarray(all_scene_ids),
        teacher=np.asarray("adaptive_gradient_cbf"),
    )
    print(f"teacher success={successes}/{len(scenes)}; wrote "
          f"{len(all_states)} samples from {len(set(all_scene_ids))} scenes "
          f"to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
