"""Phase-one position-only scenario setup shared by all controllers."""

from __future__ import annotations

import numpy as np


def apply_named_scenario(env, name: str) -> None:
    if name == "free_space":
        _free_space(env)
    elif name == "whole_body":
        _linear_scene(env, 0.18, _whole_body_obstacles())
    elif name == "confined_space":
        _linear_scene(env, 0.15, _corridor_obstacles())
    elif name == "generalization":
        _linear_scene(env, 0.18, _generalization_obstacles())
    else:
        raise ValueError(f"unknown phase-one scenario: {name}")
    _initialize_at_start(env)


def _initialize_at_start(env) -> None:
    target = env.x_d.copy()
    q_start = None
    seeds = [env.q.copy()]
    rng = np.random.default_rng(0)
    for _ in range(19):
        seeds.append(rng.uniform(env.kin.q_min, env.kin.q_max))
    for q_seed in seeds:
        q_start = env.kin.inverse_kinematics(target, q_init=q_seed, max_iter=250)
        if q_start is not None:
            break
    if q_start is None:
        raise RuntimeError(f"inverse kinematics failed for scenario start {target}")
    env.q = np.asarray(q_start, dtype=float)
    env.dq = np.zeros(env.n)
    env._reset_episode_progress()
    env._integral_err = np.zeros(3)
    env._ever_collided = False
    env.ee_trajectory.clear()
    env._cached_x_ee = None
    env._cached_w = None
    env._cached_capsule_dists = None
    if env.mj_data is not None:
        env.mj_data.qpos[:env.n] = env.q
        env.mj_data.qvel[:env.n] = 0.0
        import mujoco
        mujoco.mj_forward(env.mj_model, env.mj_data)


def _free_space(env) -> None:
    env.sdf.set_static_obstacles([], [])
    env._sync_obstacles_to_mujoco()
    period = 4.0
    omega = 2.0 * np.pi / period

    def position(t: float) -> np.ndarray:
        phase = omega * t
        return np.array([0.4, 0.15 * np.sin(phase), 0.4 + 0.1 * np.sin(2 * phase)])

    def velocity(t: float) -> np.ndarray:
        phase = omega * t
        return np.array([0.0, 0.15 * omega * np.cos(phase),
                         0.2 * omega * np.cos(2 * phase)])

    env.set_parametric_trajectory(position, velocity)


def _linear_scene(env, half_span: float, obstacles: list[tuple[np.ndarray, float]]) -> None:
    env.use_parametric_traj = False
    env.x_start = np.array([0.4, -half_span, 0.4])
    env.x_goal = np.array([0.4, half_span, 0.4])
    env.x_d = env.x_start.copy()
    env.dx_d[:3] = 0.0
    env.sdf.set_static_obstacles([o[0] for o in obstacles], [o[1] for o in obstacles])
    env._sync_obstacles_to_mujoco()


def _whole_body_obstacles() -> list[tuple[np.ndarray, float]]:
    return [
        (np.array([0.32, -0.08, 0.39]), 0.045),
        (np.array([0.47, 0.02, 0.43]), 0.04),
        (np.array([0.34, 0.11, 0.46]), 0.035),
    ]


def _corridor_obstacles() -> list[tuple[np.ndarray, float]]:
    obstacles = []
    # 0.24 m free width between sphere surfaces. The previous centres at
    # x=0.32/0.48 with r=0.04 left only 0.08 m and was oracle-infeasible.
    for y in np.linspace(-0.14, 0.14, 5):
        obstacles.append((np.array([0.25, y, 0.4]), 0.03))
        obstacles.append((np.array([0.55, y, 0.4]), 0.03))
    return obstacles


def _generalization_obstacles() -> list[tuple[np.ndarray, float]]:
    return [
        (np.array([0.31, -0.12, 0.38]), 0.03),
        (np.array([0.48, -0.02, 0.44]), 0.05),
        (np.array([0.33, 0.10, 0.47]), 0.04),
    ]
