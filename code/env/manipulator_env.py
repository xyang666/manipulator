"""
manipulator_env.py
------------------
MuJoCo-based 7-DOF manipulator environment with:
  - 7D action space: [Δẋ_RL (3), z (4)] — task relaxation + null-space coordinates
  - Dense reward combining tracking, obstacle avoidance, manipulability, energy
  - Signed distance field (simplified sphere model) for obstacle detection
  - Tracking-error-driven path progression (parameterized by s ∈ [0,1])

Observation space:
    s = [q (7), dq (7), x_ee (3), x_d (3), capsule_dists, self_dists, scene_embed, waypoints, path_progress, sigma]

Action space (paper, Route A — position-only):
    a = [Δẋ_RL ∈ R^3, z ∈ R^4]  dim=7
    Control law: q̇ = J⁺(ẋ_d + Kp(x_d - x) + σ·Δẋ_RL) + B(q)z
    Gate operator diag(σ): scaled by d_obs (σ→0 when safe, σ→1 when dangerous)
    Uses position-only Jacobian J_pos ∈ ℝ³ˣ⁷ → null-space dimension = 4.

Author: xie yang
Date:   2025-06
"""

import numpy as np
import collections
from typing import Optional

try:
    import mujoco
    import mujoco.viewer
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("[env] WARNING: mujoco not found. Running in kinematics-only mode.")

from env.kinematics import ManipulatorKinematics
from env.dynamics import ManipulatorDynamics
from agent.reward import RewardFunction
from utils.sdf import ObstacleSDF
from utils.collision import CollisionDetector
from utils.cbf import CBFController
from trajectory.generator import TrajectoryGenerator
from experiment_config import ENVIRONMENT


def dense_safety_cost(constraint_distance: float, d_safe: float) -> float:
    """Normalized safety-margin violation, capped for numerical stability."""
    if d_safe <= 0:
        raise ValueError("d_safe must be positive")
    return float(np.clip(max(0.0, d_safe - constraint_distance) / d_safe,
                         0.0, 2.0))

try:
    from control.mpc_controller import MPCController
    HAS_MPC = True
except ImportError:
    HAS_MPC = False
    print("[env] WARNING: MPC controller not available.")


# Default Panda-like joint limits
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])


class ManipulatorEnv:
    """
    Gym-compatible environment for 7-DOF manipulator obstacle avoidance.

    If MuJoCo + URDF are available, runs full physics simulation.
    Otherwise runs a kinematics-only simulation for algorithm validation.
    """
    def __init__(self,
                 urdf_path: Optional[str] = None,
                 xml_path: Optional[str] = None,
                 n_joints: int = 7,
                 dt: float = ENVIRONMENT.dt,
                 episode_len: int = ENVIRONMENT.episode_len,
                 trajectory_steps: int = ENVIRONMENT.trajectory_steps,
                 tracking_full_speed_error: float = ENVIRONMENT.tracking_full_speed_error,
                 tracking_stop_error: float = ENVIRONMENT.tracking_stop_error,
                 success_tolerance: float = ENVIRONMENT.success_tolerance,
                 success_hold_steps: int = ENVIRONMENT.success_hold_steps,
                 n_obstacles: int = 3,
                 obs_radius: float = 0.1,
                 controller: str = "rl",
                 mpc_horizon: int = 10,
                 use_trajectory_generator: bool = False,
                 manipulability_threshold: float = 0.01,
                 collision_term: bool = True,
                 path_deadzone: float = 0.20,
                 w_obs: float = ENVIRONMENT.w_obs,
                 w_collision: float = ENVIRONMENT.w_collision,
                 collision_event_penalty: float = ENVIRONMENT.collision_event_penalty,
                 w_track: float = ENVIRONMENT.w_track,
                 w_manip: float = ENVIRONMENT.w_manip,
                 w_energy: float = ENVIRONMENT.w_energy,
                 w_action: float = ENVIRONMENT.w_action,
                 w_null: float = ENVIRONMENT.w_null,
                 d_safe: float = ENVIRONMENT.d_safe,
                 use_cbf: bool = False,
                 cbf_alpha: float = 1.0,
                 cbf_self_d_safe: float = 0.02,
                 cbf_multi_self_constraints: bool = False,
                 success_bonus: float = ENVIRONMENT.success_bonus,
                 reward_min: Optional[float] = None,
                 reward_scale: float = ENVIRONMENT.reward_scale,
                 lr_lag: float = 0.01,
                 lag_target: float = 0.05,
                 gate_enabled: bool = True,
                 obs_waypoint_steps: list | None = None,
                 obs_scene_embed: int = ENVIRONMENT.obs_scene_embed,
                 frame_stack: int = 1):
        """
        Parameters
        ----------
        urdf_path   : URDF for kinematics/dynamics (Pinocchio)
        xml_path    : MuJoCo XML model path
        dt          : simulation timestep (s)
        episode_len : max steps per episode
        n_obstacles : number of spherical obstacles
        obs_radius  : obstacle radius (m)
        controller  : control mode ("rl", "mpc")
        mpc_horizon : MPC prediction horizon
        use_trajectory_generator : if True, use TrajectoryGenerator for reset
        manipulability_threshold : minimum manipulability for generated trajectories
        """
        self.n = n_joints
        self.dt = dt
        self.episode_len = episode_len
        if not 0 < trajectory_steps <= episode_len:
            raise ValueError("trajectory_steps must be in [1, episode_len]")
        if not 0 <= tracking_full_speed_error < tracking_stop_error:
            raise ValueError("tracking error thresholds must be ordered and non-negative")
        if success_tolerance <= 0 or success_hold_steps <= 0:
            raise ValueError("success tolerance and hold steps must be positive")
        self.trajectory_steps = trajectory_steps
        self.tracking_full_speed_error = tracking_full_speed_error
        self.tracking_stop_error = tracking_stop_error
        self.success_tolerance = success_tolerance
        self.success_hold_steps = success_hold_steps
        self.use_trajectory_generator = use_trajectory_generator
        self.collision_term = collision_term
        self.path_deadzone = path_deadzone
        self.frame_stack = frame_stack
        self.gate_enabled = gate_enabled

        # Observation dimensions
        self.obs_waypoint_steps = (
            list(ENVIRONMENT.obs_waypoint_steps)
            if obs_waypoint_steps is None else list(obs_waypoint_steps)
        )
        self.obs_scene_embed = obs_scene_embed
        self.obs_dim = n_joints * 2 + 3 + 3 + 3 + 1 + 1 + 3  # fallback, overwritten below
        self.act_dim = n_joints  # 7D: 3 (task relaxation) + 4 (nullspace, = n-3)

        self.kin = ManipulatorKinematics(urdf_path, n_joints)
        # Sync env DOF with actual model loaded by Pinocchio (may differ from n_joints)
        self.n = self.kin.n

        # MuJoCo setup (must come before obs_dim — MuJoCo geom count determines
        # capsule dimension when available; also needed for _robot_geom_ids)
        self.mj_model = None
        self.mj_data = None
        if HAS_MUJOCO and xml_path is not None:
            self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
            self.mj_data = mujoco.MjData(self.mj_model)
            print(f"[env] MuJoCo model loaded: {xml_path}")

        # Trajectory generator (placeholder — set below after full init)
        self.traj_gen = None

        # Collision detector
        self.collision_detector = CollisionDetector(self.mj_model, self.mj_data)

        # Identify robot geom IDs from MuJoCo model (for per-capsule distances).
        # When MuJoCo is available, use its actual geom count for obs dimension.
        self._robot_geom_ids = []
        if HAS_MUJOCO and self.mj_model is not None:
            for i in range(self.mj_model.ngeom):
                body_id = self.mj_model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(self.mj_model,
                                               mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name and ("panda" in body_name.lower()
                                  or "ewalker" in body_name.lower()
                                  or "link" in body_name.lower()):
                    self._robot_geom_ids.append(i)

        # Per-capsule obstacle distances for observations — use MuJoCo geom count
        # when available (exact) else fall back to Pinocchio capsule model.
        zero_q = np.zeros(self.n)
        if self._robot_geom_ids:
            self._capsule_dists_dim = len(self._robot_geom_ids)
        else:
            try:
                self._capsule_dists_dim = len(self.kin.get_link_capsules(zero_q))
            except Exception:
                self._capsule_dists_dim = 0

        # Self-collision distances (non-adjacent capsule pairs)
        try:
            self._self_dists_dim = self.kin.n_self_pairs
        except Exception:
            self._self_dists_dim = 0

        if self.obs_scene_embed > 0:
            # Scene-embed observation: no top-K (redundant with scene_embed)
            self.obs_dim = (self.n * 2 + 3 + 3    # q, dq, x_ee, x_d
                            + self._capsule_dists_dim * 4
                            + self._self_dists_dim
                            + self.obs_scene_embed * 5
                            + len(self.obs_waypoint_steps) * 3
                            + 1                   # path_progress s
                            + 1)                  # sigma gate
        else:
            self.obs_dim = (self.n * 2 + 3 + 3
                            + self._capsule_dists_dim * 4
                            + self._self_dists_dim
                            + 1
                            + 1)                  # sigma gate
        self.act_dim = self.n  # 7D: 3 (task) + 4 (nullspace, via nullspace basis)

        # Frame stacking: store single-frame dim, then multiply obs_dim
        self._single_obs_dim = self.obs_dim
        self.obs_dim = self._single_obs_dim * self.frame_stack
        self._obs_history = collections.deque(maxlen=self.frame_stack)

        if self.kin.model is not None:
            self._dq_max = self.kin.model.velocityLimit[:self.n].copy()
        else:
            self._dq_max = np.full(self.n, 0.5)
        self.dyn = ManipulatorDynamics(urdf_path, n_joints)

        # Trajectory generator (optional)
        self.traj_gen = None
        if use_trajectory_generator and urdf_path is not None:
            self.traj_gen = TrajectoryGenerator(
                urdf_path=urdf_path,
                n_joints=n_joints,
                manipulability_threshold=manipulability_threshold,
                obstacle_radius_range=(obs_radius * 0.5, obs_radius * 1.5)
            )
            print(f"[env] TrajectoryGenerator enabled with manip_threshold={manipulability_threshold}")
            self.traj_gen.collision_detector = self.collision_detector

        # Reward function with collision detection
        self.reward_fn = RewardFunction(
            dt=dt, w_obs=w_obs,
            w_collision=w_collision, w_track=w_track,
            w_manip=w_manip, w_energy=w_energy, w_action=w_action,
            d_safe=d_safe,
            collision_detector=self.collision_detector)
        self.d_safe = d_safe
        self.success_bonus = success_bonus
        self.reward_min = reward_min
        self.reward_scale = reward_scale
        if collision_event_penalty < 0.0:
            raise ValueError("collision_event_penalty must be non-negative")
        self.collision_event_penalty = float(collision_event_penalty)
        self.w_null = w_null
        self.sdf = ObstacleSDF(n_obstacles, obs_radius)

        # CBF safety filter (optional, post-hoc dq_cmd safety wrapper)
        self.cbf = None
        if use_cbf:
            self.cbf = CBFController(self.sdf, self.kin,
                                     d_safe=d_safe, alpha=cbf_alpha,
                                     self_d_safe=cbf_self_d_safe,
                                     multi_self_constraints=(
                                         cbf_multi_self_constraints))
            print(f"[env] CBF safety filter enabled (alpha={cbf_alpha}, "
                  f"d_safe={d_safe}, self_d_safe={cbf_self_d_safe}, "
                  f"multi_self={cbf_multi_self_constraints})")

        # Lagrangian multiplier for constraint-based gating
        # λ ≥ 0, updated via dual ascent: λ += lr_lag * (violation - target)
        self.lr_lag = lr_lag
        self.lag_target = lag_target
        self._lag_lambda = 0.0

        # Controllers
        self.controller = controller
        self.mpc = None
        if self.controller == "mpc" and HAS_MPC:
            self.mpc = MPCController(
                n_states=n_joints * 2,
                n_controls=n_joints,
                horizon=mpc_horizon,
                dt=dt
            )
            print(f"[env] MPC controller enabled (horizon={mpc_horizon})")

        # End-effector trajectory tracking
        self.ee_trajectory = []
        self.max_trajectory_len = 500

        # Path parameterization (tracking-error-driven)
        self.path_param = 0.0  # s ∈ [0, 1]
        self._trajectory_phase = 0.0
        self._success_hold_count = 0

        # Parametric trajectory support (for figure-8, etc.)
        self.use_parametric_traj = False
        self._parametric_pos_func = None   # callable(t) → position (3,)
        self._parametric_vel_func = None   # callable(t) → velocity (3,)


        self._reset_state()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self._reset_state()
        self.ee_trajectory.clear()
        self._reset_episode_progress()
        self._obs_history.clear()
        return self._get_obs()

    @staticmethod
    def _minimum_jerk_progress(phase: float) -> float:
        """C2-continuous progress with zero endpoint velocity/acceleration."""
        u = float(np.clip(phase, 0.0, 1.0))
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

    def _tracking_progress_gate(self, tracking_error: float) -> float:
        if tracking_error <= self.tracking_full_speed_error:
            return 1.0
        if tracking_error >= self.tracking_stop_error:
            return 0.0
        u = ((tracking_error - self.tracking_full_speed_error) /
             (self.tracking_stop_error - self.tracking_full_speed_error))
        smooth = u * u * (3.0 - 2.0 * u)
        return float(1.0 - smooth)

    def _reset_episode_progress(self) -> None:
        self.step_count = 0
        self.path_param = 0.0
        self._trajectory_phase = 0.0
        self._success_hold_count = 0
        self._last_advance = 1.0

    def _update_reference(self, x_ee: np.ndarray) -> float:
        """Advance the reference without catching up after tracking stalls."""
        tracking_error = float(np.linalg.norm(x_ee - self.x_d))
        prev_x_d = self.x_d.copy()
        if self.use_parametric_traj and self._parametric_pos_func is not None:
            t = self.step_count * self.dt
            self.x_d = self._parametric_pos_func(t)
            self.dx_d[:3] = self._parametric_vel_func(t)
            return tracking_error

        gate = self._tracking_progress_gate(tracking_error)
        self._last_advance = gate
        self._trajectory_phase = min(
            1.0, self._trajectory_phase + gate / self.trajectory_steps
        )
        self.path_param = self._minimum_jerk_progress(self._trajectory_phase)
        self.x_d = (1.0 - self.path_param) * self.x_start + self.path_param * self.x_goal
        self.dx_d[:3] = (self.x_d - prev_x_d) / self.dt
        return tracking_error

    def _termination_status(self, x_ee: np.ndarray, collision: bool):
        """Return success, done and an explicit termination reason."""
        if self.use_parametric_traj:
            success = self.step_count >= self.episode_len and not self._ever_collided
        else:
            final_error = float(np.linalg.norm(x_ee - self.x_goal))
            candidate = (self.path_param >= 0.99 and
                         final_error < self.success_tolerance and
                         not self._ever_collided)
            self._success_hold_count = self._success_hold_count + 1 if candidate else 0
            success = self._success_hold_count >= self.success_hold_steps

        timed_out = self.step_count >= self.episode_len
        done = success or timed_out or (self.collision_term and collision)
        if success:
            reason = "success"
        elif self.collision_term and collision:
            reason = "collision"
        elif timed_out:
            reason = "timeout"
        else:
            reason = None
        return success, done, reason

    def _apply_collision_event_penalty(self, reward, reward_info, collision):
        """Align the scalar reward with authoritative MuJoCo contact events."""
        if not collision or self.collision_event_penalty == 0.0:
            return reward
        reward_info["r_collision"] = (
            float(reward_info.get("r_collision", 0.0))
            - self.collision_event_penalty
        )
        return reward - self.collision_event_penalty

    def set_parametric_trajectory(self, pos_func, vel_func):
        """
        Switch to a time-parameterized trajectory (e.g. figure-8).

        When set, the target position x_d and velocity dx_d are computed from
        the provided functions of time (t = step_count * dt), overriding the
        default linear start→goal progression.

        Parameters
        ----------
        pos_func : callable(t: float) -> ndarray (3,)
            Desired EE position at time t.
        vel_func : callable(t: float) -> ndarray (3,)
            Desired EE velocity (analytical derivative) at time t.
        """
        self.use_parametric_traj = True
        self._parametric_pos_func = pos_func
        self._parametric_vel_func = vel_func
        t = self.step_count * self.dt
        self.x_d = pos_func(t)
        self.dx_d[:3] = vel_func(t)

    def step(self, action: np.ndarray):
        """
        Parameters
        ----------
        action : 7D action [Δẋ_RL (3), z (4)]
                 Δẋ_RL: position-space relaxation velocity (gated by d_obs)
                 z     : coordinates in the position-Jacobian null-space basis

        Returns
        -------
        obs, reward, done, info
        """
        action = np.asarray(action, dtype=float)
        if action.shape != (self.act_dim,):
            raise ValueError(
                f"expected action shape {(self.act_dim,)}, got {action.shape}"
            )
        if self.controller == "mpc" and self.mpc is not None:
            # MPC mode: directly optimize task-space tracking with obstacle avoidance
            dq_cmd = self.mpc.compute_control_task_space(
                self.q, self.dq, self.x_d, self.dx_d, self.kin,
                obs_centers=self.sdf.centers if self.sdf.n_obs > 0 else None,
                obs_radii=self.sdf.radii if self.sdf.n_obs > 0 else None
            )
            # Store dummy values for MPC mode (physics loss disabled via buffer guard)
            self._last_J = np.zeros((3, self.n), dtype=np.float32)
            self._last_sigma = np.float32(0.0)
            self._last_dx_nom = np.zeros(3, dtype=np.float32)
            self._lag_lambda = 0.0

        else:
            # Decompose 7D action into task relaxation + null-space coefficients
            delta_x_rl = action[:3]   # Δẋ_RL ∈ R^3: task-space relaxation (gated by σ)
            z          = action[3:]   # z ∈ R^{n-3} (nullspace coefficients, via SVD basis)

            # Compute nominal task-space velocity (PID tracking)
            dx_nom = self._compute_task_velocity()  # ẋ_d + Kp(x_d - x) + Ki*∫(x_d - x)dt

            # Lagrangian gate σ: learned multiplier for constraint-based gating.
            # λ ≥ 0 updated via dual ascent: λ += lr_lag * (violation - target)
            # sigma = clip(λ, 0, 1) gates RL vs tracking control.
            # sigma_override bypasses the gate (used for random exploration in start_steps)
            sigma_ov = getattr(self, 'sigma_override', None)
            if not self.gate_enabled:
                sigma = 0.0
            elif sigma_ov is not None:
                sigma = float(sigma_ov)
            else:
                # Compute current d_obs for constraint violation (before integration)
                x_ee_cur, _ = self.kin.forward_kinematics(self.q)
                d_obs_cur = self._mujoco_min_distance()
                # Immediate sigma response: no slow Lagrangian accumulation.
                # d_obs ≥ 2*d_safe → sigma=0  (pure tracking, safe)
                # d_obs ≤ 0       → sigma=1  (full RL emergency)
                # In between → smoothstep blend
                if d_obs_cur >= 2.0 * self.d_safe:
                    sigma = 0.0
                elif d_obs_cur <= 0.0:
                    sigma = 1.0
                else:
                    t = 1.0 - d_obs_cur / (2.0 * self.d_safe)
                    sigma = float(t * t * (3.0 - 2.0 * t))  # smoothstep
            delta_x_gated = sigma * delta_x_rl  # diag(σ) · Δẋ_RL

            # Reconstruct 7D nullspace velocity from 4D coefficients via SVD basis
            B = self.kin.null_space_basis_position(self.q)  # (7, 4), J_pos @ B ≈ 0
            dq0 = B @ z  # (7,) nullspace self-motion

            # Combine: q̇ = J_pos⁺(dx_nom + delta_x_gated) + B·z
            dq_cmd = self.kin.combine_velocities_with_relaxation_position(
                self.q, dx_nom, delta_x_gated, dq0
            )

            # Save intermediate values for differentiable physics loss (Plan B)
            self._last_J = self.kin.jacobian_position(self.q).copy()
            self._last_sigma = sigma
            self._last_dx_nom = dx_nom.copy()

            # CBF safety filter: modify dq_cmd to ensure d_obs stays above d_safe
            # This is a post-hoc wrapper — does NOT affect physics loss intermediates
            # (buffer stores raw RL action, CBF is part of the environment dynamics)
            self._cbf_active = False
            self._cbf_mod = 0.0
            if self.cbf is not None:
                dq_cmd, cbf_info = self.cbf.filter(dq_cmd, self.q)
                self._cbf_active = cbf_info["active"]
                self._cbf_mod = cbf_info["dq_norm"]

        # Save previous joint velocity for smoothness penalty
        prev_dq = self.dq.copy()

        # Apply joint velocity limits before integration
        dq_cmd = np.clip(dq_cmd, -self._dq_max, self._dq_max)

        # Integrate (kinematics-only mode)
        q_new = self.q + dq_cmd * self.dt
        dq_new = dq_cmd

        if self.mj_data is not None:
            self._mujoco_step(dq_cmd)
        else:
            self.q = q_new
            self.dq = dq_new

        # Tracking-error-gated minimum-jerk reference progression.
        x_ee, _ = self.kin.forward_kinematics(self.q)
        self._cached_x_ee = x_ee
        tracking_error = self._update_reference(x_ee)

        self.step_count += 1

        # Compute reward (use cached FK from progression step)
        d_obs = self._mujoco_min_distance()
        d_obs = float(np.clip(d_obs, -0.5, 0.5))  # cap inf for numerical stability
        w = self._manipulability()
        self._cached_w = w

        # Record end-effector position for trajectory visualization
        if len(self.ee_trajectory) >= self.max_trajectory_len:
            self.ee_trajectory.pop(0)
        self.ee_trajectory.append(x_ee.copy())

        # Per-capsule distances (MuJoCo geometry).
        capsule_dists, capsule_directions = (
            self._mujoco_per_capsule_obstacle_features()
        )
        self._cached_capsule_dists = capsule_dists
        self._cached_capsule_directions = capsule_directions

        reward, reward_info = self.reward_fn.compute(
            q=self.q, dq=self.dq, x_ee=x_ee,
            x_d=self.x_d, dx_d=self.dx_d,
            d_obs=d_obs, w=w,
            action=action, prev_dq=prev_dq,
            capsule_dists=capsule_dists,
        )
        # Structured policies can otherwise exploit large null-space motions
        # that preserve end-effector tracking while increasing full-arm risk.
        null_penalty = self.w_null * float(np.square(action[3:]).sum())
        reward -= null_penalty
        reward_info["r_action"] = (
            float(reward_info.get("r_action", 0.0)) - null_penalty
        )
        # Clip per-step reward to prevent Q-value divergence from collision spikes
        if self.reward_min is not None:
            reward = max(reward, self.reward_min)
        # Collision detection: MuJoCo robot-obstacle + non-adjacent self-collisions.
        # Adjacent link contacts and finger-finger initial contact are excluded.
        if self.collision_detector is not None:
            _, n_obs = self.collision_detector.detect_obstacle_collisions()
            _, n_self = self.collision_detector.detect_self_collisions()
            collision = (n_obs + n_self) > 0
        else:
            n_obs = 0
            n_self = 0
            collision = False
        reward = self._apply_collision_event_penalty(
            reward, reward_info, collision
        )

        # Track cumulative collision flag for the entire episode
        self._ever_collided = self._ever_collided or collision

        success, done, termination_reason = self._termination_status(x_ee, collision)

        # Sparse success bonus only after the goal has been held continuously.
        if success:
            reward += self.success_bonus

        # Scale reward for stable Q-learning (compresses Q-value range)
        reward = reward / self.reward_scale

        tracking_error = float(np.linalg.norm(x_ee - self.x_d))
        # Dense constraint violation for the safety critic.  MuJoCo collision
        # remains the authoritative termination signal, while capsule distances
        # provide useful learning signal before contact occurs.
        self_dists = self.kin.compute_self_distances(self.q)
        d_self = float(np.min(self_dists)) if len(self_dists) else float("inf")
        constraint_distance = min(float(d_obs), d_self)
        cost = dense_safety_cost(constraint_distance, self.d_safe)

        info = {"d_obs": d_obs, "d_self": d_self,
                "constraint_distance": constraint_distance,
                "w": w, "success": success,
                "collision": collision, "cost": cost,
                "obstacle_collision": n_obs > 0,
                "self_collision": n_self > 0,
                "path_param": self.path_param, "tracking_error": tracking_error,
                "termination_reason": termination_reason,
                "success_hold_count": self._success_hold_count,
                "cbf_active": self._cbf_active, "cbf_mod": self._cbf_mod,
                **reward_info}

        return self._get_obs(), reward, done, info

    def render(self, show_robot: bool = False):
        """Launch or sync the passive MuJoCo viewer and draw end-effector trajectory.

        Parameters
        ----------
        show_robot : if False, hide the opaque robot geoms and draw the same
            collision geoms semi-transparently. Their original MuJoCo type,
            size and orientation are preserved.
        """
        if self.mj_model is None:
            return
        if not hasattr(self, '_viewer'):
            self._viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        if self._viewer.is_running():
            # Hide only robot body geoms by setting alpha to 0. The collision
            # overlay below redraws them with the real MuJoCo geometry type.
            for i in range(self.mj_model.ngeom):
                body_id = self.mj_model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name and (
                        "panda" in body_name.lower()
                        or "ewalker" in body_name.lower()
                        or "link" in body_name.lower()):
                    self.mj_model.geom_rgba[i, 3] = 1.0 if show_robot else 0.0
            # Draw visualizations
            self._draw_visualizations()
            self._viewer.sync()

    def _draw_visualizations(self):
        """Draw visualizations: obstacles, fixed target point, EE trajectory, and link capsules."""
        scene = self._viewer.user_scn
        scene.ngeom = 0  # Clear previous geometries

        # 1. Draw the actual robot collision geometry from MuJoCo. Previously
        # every geom was forced to mjGEOM_SPHERE, which made capsule links look
        # like a chain of balls in the interactive viewer.
        for gid in self._robot_geom_ids:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                int(self.mj_model.geom_type[gid]),
                self.mj_model.geom_size[gid].copy(),
                self.mj_data.geom_xpos[gid].copy(),
                self.mj_data.geom_xmat[gid].copy(),
                np.array([0.0, 0.5, 1.0, 0.3])  # Blue, semi-transparent
            )
            scene.ngeom += 1

        # 2. Draw obstacles (semi-transparent red spheres)
        for i, obs_center in enumerate(self.sdf.centers):
            if scene.ngeom >= scene.maxgeom:
                break

            # Use individual radius for each obstacle
            size = np.array([self.sdf.radii[i], 0, 0])

            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size, obs_center, np.eye(3).flatten(),
                np.array([1.0, 0.0, 0.0, 0.3])  # Red, semi-transparent
            )
            scene.ngeom += 1

        # 3. Draw fixed target point (yellow sphere, larger)
        if scene.ngeom < scene.maxgeom:
            size = np.array([0.02, 0, 0])  # Larger sphere for fixed target

            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size, self.x_d, np.eye(3).flatten(),
                np.array([1.0, 1.0, 0.0, 1.0])  # Yellow
            )
            scene.ngeom += 1

        # 4. Draw end-effector trajectory (green points)
        if len(self.ee_trajectory) < 1:
            return

        for i in range(len(self.ee_trajectory)):
            if scene.ngeom >= scene.maxgeom:
                break

            p1 = self.ee_trajectory[i]
            size = np.array([0.004, 0., 0.])

            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size, p1, np.eye(3).flatten(),
                np.array([0.0, 1.0, 0.0, 1.0])  # Green for trajectory
            )
            scene.ngeom += 1

    @property
    def observation_space_dim(self):
        return self.obs_dim

    @property
    def action_space_dim(self):
        return self.act_dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self):
        """
        Reset environment state with trajectory and obstacles.

        If use_trajectory_generator=True, generates collision-free scenes using TrajectoryGenerator.
        Otherwise uses default fixed trajectory (legacy behavior).
        """
        self._reset_episode_progress()
        self._integral_err = np.zeros(3)
        self._ever_collided = False
        self.reward_fn._prev_dist_to_goal = None  # reset goal distance tracking
        self._lag_lambda = 0.0  # reset Lagrangian multiplier

        # Initialize physics loss storage fields (set during step())
        self._last_J = np.zeros((3, self.n), dtype=np.float32)
        self._last_sigma = np.float32(0.0)
        self._last_dx_nom = np.zeros(3, dtype=np.float32)
        self._cbf_active = False
        self._cbf_mod = 0.0

        # Cached values for _get_obs() to avoid recomputation
        self._cached_x_ee = None
        self._cached_w = None
        self._cached_capsule_dists = None
        self._cached_capsule_directions = None

        if self.use_trajectory_generator and self.traj_gen is not None:
            # Generate new scene using TrajectoryGenerator
            scene = self.traj_gen.generate_scene(
                scene_id=0,
                n_obstacles=self.sdf.n_obs,
                max_attempts=100,
            )

            if scene is not None:
                # Extract trajectory
                self.x_start = np.array(scene["start"])
                self.x_goal = np.array(scene["goal"])

                # Extract obstacles
                obstacles = scene["obstacles"]
                obstacle_centers = [np.array(obs[:3]) for obs in obstacles]
                obstacle_radii = [obs[3] for obs in obstacles]

                # Update SDF with variable radii
                self.sdf.set_static_obstacles(obstacle_centers, obstacle_radii)

                # print(f"[env] Generated scene: manip={scene['manipulability_mean']:.4f}, "
                #       f"dist={np.linalg.norm(self.x_goal - self.x_start):.3f}m")
            else:
                print("[env] WARNING: Scene generation failed, using default trajectory")
                self._reset_state_default()
                return
        else:
            # Use default fixed trajectory
            self._reset_state_default()
            return

        # Current target (starts at start position)
        self.x_d = self.x_start.copy()

        # Minimum-jerk point-to-point trajectories start at zero velocity.
        self.dx_d = np.zeros(3)

        # Use scene-verified IK config (avoids recomputing IK that may self-collide)
        if "start_q" in scene:
            self.q = np.array(scene["start_q"])
        else:
            q_init = self.kin.inverse_kinematics(self.x_start)
            if q_init is not None:
                self.q = q_init
            else:
                print("[env] WARNING: IK failed for start position, using home pose")
                self.q = np.zeros(self.n)

        self.dq = np.zeros(self.n)
        self._sync_obstacles_to_mujoco()

        # Reset MuJoCo state
        if self.mj_data is not None:
            self.mj_data.qpos[:self.n] = self.q
            self.mj_data.qvel[:self.n] = self.dq
            self.mj_data.qpos[self.n:] = 0.0
            self.mj_data.qvel[self.n:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)

    def _reset_state_default(self):
        """
        Default fixed trajectory (legacy behavior).
        场景1：人机协作-狭窄空间装配（论文 Section 4.1.3）
        """
        # Fixed trajectory
        self.x_start = np.array([0.8, 0.0, 0.5])
        self.x_goal = np.array([0.8, 0.0, 0.3])
        self.x_d = self.x_start.copy()
        self.dx_d = np.zeros(3)

        # IK for initial configuration
        # The experiment controls end-effector position only. Requiring a
        # legacy Panda orientation here can make a reachable E-Walker start
        # pose fail IK unnecessarily.
        q_init = self.kin.inverse_kinematics(self.x_start)
        if q_init is not None:
            self.q = q_init
        else:
            print("[env] WARNING: IK failed, using home pose")
            self.q = np.zeros(self.n)

        self.dq = np.zeros(self.n)

        # Generate obstacles near trajectory
        if self.sdf.n_obs > 0:
            obstacle_centers = self._generate_obstacles_near_trajectory()
            self.sdf.set_static_obstacles(obstacle_centers)
        else:
            self.sdf.set_static_obstacles([])

        self._sync_obstacles_to_mujoco()

        # Reset MuJoCo
        if self.mj_data is not None:
            self.mj_data.qpos[:self.n] = self.q
            self.mj_data.qvel[:self.n] = self.dq
            self.mj_data.qpos[self.n:] = 0.0
            self.mj_data.qvel[self.n:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)

    def _generate_obstacles_near_trajectory(self) -> list:
        """
        Generate obstacles randomly near the trajectory but not interfering with it.

        Returns
        -------
        list of np.ndarray
            List of obstacle center positions
        """
        obstacles = []
        # Use default radius for legacy obstacle generation
        default_radius = self.sdf.default_radius
        min_dist_to_trajectory = default_radius + 0.05  # Safety margin: radius + 5cm
        max_attempts = 100

        # Trajectory bounding box with margin
        traj_min = np.minimum(self.x_start, self.x_goal) - 0.15
        traj_max = np.maximum(self.x_start, self.x_goal) + 0.15

        for _ in range(self.sdf.n_obs):
            for attempt in range(max_attempts):
                # Random position in bounding box
                candidate = np.random.uniform(traj_min, traj_max)

                # Check distance to trajectory (line segment from start to goal)
                dist_to_traj = self._point_to_segment_distance(
                    candidate, self.x_start, self.x_goal
                )

                # Check distance to existing obstacles
                too_close = False
                for existing_obs in obstacles:
                    if np.linalg.norm(candidate - existing_obs) < 2 * default_radius:
                        too_close = True
                        break

                # Accept if far enough from trajectory and other obstacles
                if dist_to_traj >= min_dist_to_trajectory and not too_close:
                    obstacles.append(candidate)
                    break
            else:
                # Fallback: place obstacle far from trajectory
                offset = np.random.randn(3)
                offset = offset / np.linalg.norm(offset) * (min_dist_to_trajectory + 0.1)
                mid_point = (self.x_start + self.x_goal) / 2
                obstacles.append(mid_point + offset)

        return obstacles

    def _point_to_segment_distance(self, point: np.ndarray,
                                   seg_start: np.ndarray,
                                   seg_end: np.ndarray) -> float:
        """
        Calculate minimum distance from point to line segment.

        Parameters
        ----------
        point : np.ndarray
            Query point
        seg_start : np.ndarray
            Segment start point
        seg_end : np.ndarray
            Segment end point

        Returns
        -------
        float
            Minimum distance
        """
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)

        if seg_len_sq < 1e-8:
            return np.linalg.norm(point - seg_start)

        # Project point onto line, clamp to [0, 1]
        t = np.clip(np.dot(point - seg_start, seg_vec) / seg_len_sq, 0.0, 1.0)
        projection = seg_start + t * seg_vec

        return np.linalg.norm(point - projection)

    # ------------------------------------------------------------------
    # MuJoCo-based distance computation (replaces capsule SDF)
    # ------------------------------------------------------------------

    def _mujoco_obstacle_geom_ids(self):
        """Return list of obstacle geom IDs currently in the scene."""
        ids = []
        for i in range(self.sdf.n_obs):
            gid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"obs{i}")
            if gid < 0:
                break
            ids.append(gid)
        return ids

    def _mujoco_min_distance(self) -> float:
        """
        Minimum signed distance from any robot geom to any obstacle
        using MuJoCo's actual geometry positions and sizes.
        """
        if self.mj_data is None or not self._robot_geom_ids:
            return np.inf

        obs_ids = self._mujoco_obstacle_geom_ids()
        if not obs_ids:
            return np.inf

        min_dist = np.inf
        for rgid in self._robot_geom_ids:
            # Capsule centre and axis
            pos = self.mj_data.geom_xpos[rgid]
            r = self.mj_model.geom_size[rgid, 0]      # capsule radius
            h = self.mj_model.geom_size[rgid, 1]      # half-height
            mat = self.mj_data.geom_xmat[rgid].reshape(3, 3)
            z = mat[:, 2]                               # capsule direction

            p1 = pos - h * z
            p2 = pos + h * z

            for ogid in obs_ids:
                opos = self.mj_data.geom_xpos[ogid]
                orad = self.mj_model.geom_size[ogid, 0]

                # Project obstacle onto capsule axis
                t = np.dot(opos - pos, z)
                t = np.clip(t, -h, h)
                closest = pos + t * z
                d = np.linalg.norm(opos - closest) - r - orad
                if d < min_dist:
                    min_dist = d
        return float(min_dist)

    def _mujoco_per_capsule_obstacle_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Return nearest-obstacle distance and avoidance direction per capsule.

        Directions point from the nearest obstacle centre toward the closest
        point on the capsule axis. Unlike a distance scalar, this makes left-
        versus right-side blockers distinguishable to the policy.
        """
        if self.mj_data is None or not self._robot_geom_ids:
            empty = np.array([], dtype=np.float32)
            return empty, np.empty((0, 3), dtype=np.float32)

        obs_ids = self._mujoco_obstacle_geom_ids()
        if not obs_ids:
            n_caps = len(self._robot_geom_ids)
            return (np.full(n_caps, 0.5, dtype=np.float32),
                    np.zeros((n_caps, 3), dtype=np.float32))

        n_caps = len(self._robot_geom_ids)
        dists = np.full(n_caps, 0.5, dtype=np.float32)
        directions = np.zeros((n_caps, 3), dtype=np.float32)

        for j, rgid in enumerate(self._robot_geom_ids):
            pos = self.mj_data.geom_xpos[rgid]
            r = self.mj_model.geom_size[rgid, 0]
            h = self.mj_model.geom_size[rgid, 1]
            mat = self.mj_data.geom_xmat[rgid].reshape(3, 3)
            z = mat[:, 2]

            d_min = np.inf
            nearest_direction = np.zeros(3, dtype=np.float32)
            for ogid in obs_ids:
                opos = self.mj_data.geom_xpos[ogid]
                orad = self.mj_model.geom_size[ogid, 0]
                t = np.dot(opos - pos, z)
                t = np.clip(t, -h, h)
                closest = pos + t * z
                away = closest - opos
                centre_distance = np.linalg.norm(away)
                d = centre_distance - r - orad
                if d < d_min:
                    d_min = d
                    if centre_distance > 1e-8:
                        nearest_direction = (away / centre_distance).astype(
                            np.float32
                        )
            dists[j] = float(np.clip(d_min, -0.5, 0.5))
            directions[j] = nearest_direction

        return dists, directions

    def _mujoco_per_capsule_distances(self) -> np.ndarray:
        """Backward-compatible distance-only view used by reward code/tests."""
        return self._mujoco_per_capsule_obstacle_features()[0]

    def _sync_obstacles_to_mujoco(self):
        """Sync SDF obstacle centers and radius to MuJoCo mocap bodies and geoms."""
        if self.mj_data is None:
            return
        for i, center in enumerate(self.sdf.centers):
            # Sync position via mocap body
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
            if bid >= 0:
                mocap_id = self.mj_model.body_mocapid[bid]
                if mocap_id >= 0:
                    self.mj_data.mocap_pos[mocap_id] = center
            # Sync radius via geom size
            gid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"obs{i}")
            if gid >= 0:
                self.mj_model.geom_size[gid, 0] = self.sdf.radii[i]

        # Move unused model obstacles out of the workspace. The XML contains a
        # fixed maximum number, while phase-one scenarios use different counts.
        i = self.sdf.n_obs
        while True:
            bid = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}"
            )
            if bid < 0:
                break
            mocap_id = self.mj_model.body_mocapid[bid]
            if mocap_id >= 0:
                self.mj_data.mocap_pos[mocap_id] = np.array([10.0 + i, 10.0, 10.0])
            i += 1
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _compute_task_velocity(self) -> np.ndarray:
        """
        PID tracking in position space with SDF-based repulsive velocity.
        Returns 3D position-only velocity (Route A).

        v_cmd = tracking_PID + v_rep
        v_rep = k_rep * max(0, d_safe - d_obs) * (x_ee - obs) / ||x_ee - obs||
        """
        x_ee, _ = self.kin.forward_kinematics(self.q)
        pos_err = self.x_d - x_ee
        err_norm = np.linalg.norm(pos_err)

        # Adaptive proportional gain — stronger when far from target
        Kp_base = 4.0
        Kp = Kp_base * (1.0 + np.tanh(err_norm / 0.05))

        # Leaky integral with anti-windup clamp
        Ki = 0.5
        self._integral_err = getattr(self, '_integral_err', np.zeros(3))
        self._integral_err *= 0.98
        self._integral_err += pos_err * self.dt
        self._integral_err = np.clip(self._integral_err, -0.02, 0.02)

        dx_cmd = np.zeros(3)
        dx_cmd[:] = self.dx_d[:3] + Kp * pos_err + Ki * self._integral_err

        return dx_cmd

    def _mujoco_step(self, dq_cmd):
        # Direct kinematic control: set joint positions directly
        # This bypasses dynamics for precise tracking evaluation
        q_desired = self.q + dq_cmd * self.dt

        # Apply to MuJoCo
        self.mj_data.qpos[:self.n] = q_desired
        self.mj_data.qvel[:self.n] = dq_cmd

        # Keep any non-arm joints (e.g. legacy Panda fingers) fixed.
        self.mj_data.qpos[self.n:] = 0.0
        self.mj_data.qvel[self.n:] = 0.0

        mujoco.mj_forward(self.mj_model, self.mj_data)  # Update kinematics only
        self.q = self.mj_data.qpos[:self.n].copy()
        self.dq = self.mj_data.qvel[:self.n].copy()

    def _manipulability(self) -> float:
        """Yoshikawa manipulability: w = sqrt(det(J J^T))"""
        J = self.kin.jacobian(self.q)
        JJT = J @ J.T
        val = np.sqrt(max(np.linalg.det(JJT), 0))
        return float(val)

    def _get_obs(self) -> np.ndarray:
        if self._cached_x_ee is not None:
            x_ee = self._cached_x_ee
        else:
            x_ee, _ = self.kin.forward_kinematics(self.q)
        if self._cached_w is not None:
            w = self._cached_w
        else:
            w = self._manipulability()

        if self.obs_scene_embed > 0:
            # Future waypoints along the planned path (relative to end-effector)
            waypoints = []
            for s in self.obs_waypoint_steps:
                if self.use_parametric_traj and self._parametric_pos_func is not None:
                    t = (self.step_count + s) * self.dt
                    wp = self._parametric_pos_func(t)
                else:
                    future_param = min(1.0, self.path_param + s / self.episode_len)
                    wp = (1.0 - future_param) * self.x_start + future_param * self.x_goal
                waypoints.append(wp - x_ee)  # relative to end-effector

            # Sort by current surface distance so the representation is
            # permutation-stable. Each slot is [relative xyz, radius, mask].
            scene_embed = np.zeros(self.obs_scene_embed * 5, dtype=np.float32)
            n_embed = min(self.obs_scene_embed, self.sdf.n_obs)
            if n_embed:
                rel = self.sdf.centers - x_ee
                order = np.argsort(
                    np.linalg.norm(rel, axis=1) - self.sdf.radii
                )[:n_embed]
                for slot, obstacle_index in enumerate(order):
                    offset = slot * 5
                    scene_embed[offset:offset+3] = rel[obstacle_index]
                    scene_embed[offset+3] = self.sdf.radii[obstacle_index]
                    scene_embed[offset+4] = 1.0

            if (self._cached_capsule_dists is not None
                    and self._cached_capsule_directions is not None):
                capsule_dists = self._cached_capsule_dists
                capsule_directions = self._cached_capsule_directions
            else:
                capsule_dists, capsule_directions = (
                    self._mujoco_per_capsule_obstacle_features()
                )
            capsule_features = np.concatenate(
                [capsule_dists[:, None], capsule_directions], axis=1
            ).reshape(-1)

            # Per-capsule-pair self-collision distances
            # (n_self_pairs scalars — direct signal for link-to-link proximity)
            self_dists = self.kin.compute_self_distances(self.q)

            obs = np.concatenate([
                self.q, self.dq, x_ee, self.x_d,
                *waypoints,
                capsule_features,
                self_dists,
                scene_embed,
                [self.path_param],
                [self._last_sigma],
            ])
        else:
            # Legacy observation
            if (self._cached_capsule_dists is not None
                    and self._cached_capsule_directions is not None):
                capsule_dists = self._cached_capsule_dists
                capsule_directions = self._cached_capsule_directions
            else:
                capsule_dists, capsule_directions = (
                    self._mujoco_per_capsule_obstacle_features()
                )
            capsule_features = np.concatenate(
                [capsule_dists[:, None], capsule_directions], axis=1
            ).reshape(-1)
            self_dists = self.kin.compute_self_distances(self.q)

            obs = np.concatenate([
                self.q, self.dq, x_ee, self.x_d,
                capsule_features, self_dists, [self.path_param],
                [self._last_sigma],
            ])

        obs = obs.astype(np.float32)

        # Frame stacking: maintain sliding window of recent observations
        if self.frame_stack > 1:
            self._obs_history.append(obs)
            # Pad with copies of first frame until history is full
            while len(self._obs_history) < self.frame_stack:
                self._obs_history.append(self._obs_history[0])
            stacked = np.concatenate(list(self._obs_history))
            return stacked.astype(np.float32)
        return obs

    def _solve_ik_mujoco(self, x_target: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """
        Solve IK using MuJoCo's built-in solver.

        Parameters
        ----------
        x_target : desired end-effector position [3]
        max_iter : maximum iterations

        Returns
        -------
        q : joint configuration [n]
        """
        # Start from home pose
        q_init = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
        self.mj_data.qpos[:self.n] = q_init
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Get site ID for end-effector
        site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        # Target position
        target_pos = x_target.copy()

        # Iterative IK
        for _ in range(max_iter):
            # Current EE position
            ee_pos = self.mj_data.site_xpos[site_id].copy()
            error = target_pos - ee_pos

            if np.linalg.norm(error) < 0.01:  # 1cm tolerance
                break

            # Compute Jacobian
            jacp = np.zeros((3, self.mj_model.nv))
            jacr = np.zeros((3, self.mj_model.nv))
            mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, site_id)

            # Damped least squares
            J = jacp[:, :self.n]  # Only arm joints
            lam = 0.01
            dq = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(3)) @ error

            # Update
            self.mj_data.qpos[:self.n] += 0.5 * dq
            mujoco.mj_forward(self.mj_model, self.mj_data)

        return self.mj_data.qpos[:self.n].copy()

if __name__ == "__main__":
    env = ManipulatorEnv()
    obs = env.reset()
    print(f"obs shape: {obs.shape}  (expected ({env.obs_dim},))")
    action = np.zeros(env.act_dim)
    obs, r, done, info = env.step(action)
    print(f"step ok  reward={r:.4f}  d_obs={info['d_obs']:.3f}")
    print("manipulator_env.py unit test PASSED")
