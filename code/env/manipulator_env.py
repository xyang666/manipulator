"""
manipulator_env.py
------------------
MuJoCo-based 7-DOF manipulator environment with:
  - 10D action space: [Δẋ_RL (3), dq0 (7)] — task relaxation + null-space self-motion
  - Dense reward combining tracking, obstacle avoidance, manipulability, energy
  - Signed distance field (simplified sphere model) for obstacle detection
  - Tracking-error-driven path progression (parameterized by s ∈ [0,1])

Observation space (paper Eq. state):
    s = [q (7), dq (7), x_ee (3), x_d (3), dx_d (3), d_obs (1), w(q) (1)]  dim=25

Action space (paper, Route A — position-only):
    a = [Δẋ_RL ∈ R^3, dq0 ∈ R^7]  dim=10
    Control law: q̇ = J⁺(ẋ_d + Kp(x_d - x) + diag(σ)·Δẋ_RL) + N(q)dq0
    Gate operator diag(σ): scaled by d_obs (σ→0 when safe, σ→1 when dangerous)
    Uses position-only Jacobian J_pos ∈ ℝ³ˣ⁷ → null-space dimension = 4.
"""

import numpy as np
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
from trajectory.generator import TrajectoryGenerator

try:
    from control.mpc_controller import MPCController
    HAS_MPC = True
except ImportError:
    HAS_MPC = False
    print("[env] WARNING: MPC controller not available.")


# Default Panda-like joint limits
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
DQ_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])  # rad/s


class ManipulatorEnv:
    """
    Gym-compatible environment for 7-DOF manipulator obstacle avoidance.

    If MuJoCo + URDF are available, runs full physics simulation.
    Otherwise runs a kinematics-only simulation for algorithm validation.
    """

    # Joint velocity limits (rad/s) from MuJoCo actuator specifications
    DQ_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])

    def __init__(self,
                 urdf_path: Optional[str] = None,
                 xml_path: Optional[str] = None,
                 n_joints: int = 7,
                 dt: float = 0.02,
                 episode_len: int = 200,
                 n_obstacles: int = 3,
                 obs_radius: float = 0.1,
                 controller: str = "rl",
                 mpc_horizon: int = 10,
                 d_critical: float = 0.05,
                 alpha_relax: float = 0.1,
                 use_trajectory_generator: bool = False,
                 manipulability_threshold: float = 0.01,
                 collision_term: bool = True,
                 path_deadzone: float = 0.20,
                 w_obs: float = 1.0,
                 w_obs_safe: float = 0.1,
                 w_collision: float = 100.0,
                 w_track: float = 12.0,
                 w_goal: float = 1.0,
                 w_manip: float = 0.05,
                 w_action: float = 0.5,
                 w_apf: float = 0.0,
                 d_safe: float = 0.02,
                 success_bonus: float = 50.0,
                 sigma_d_safe: Optional[float] = None,
                 sigma_d_critical: Optional[float] = None,
                 sigma_smooth: float = 0.9,
                 obs_waypoint_steps: list | None = None,
                 obs_scene_embed: int = 0):
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
        self.use_trajectory_generator = use_trajectory_generator
        self.collision_term = collision_term
        self.path_deadzone = path_deadzone

        # Observation dimensions
        self.obs_waypoint_steps = obs_waypoint_steps or []
        self.obs_scene_embed = obs_scene_embed
        self.obs_dim = n_joints * 2 + 3 + 3 + 3 + 1 + 1 + 3  # placeholder, updated after self.n
        self.act_dim = n_joints  # 7D: 3 (task relaxation) + 4 (nullspace, = n-3)

        self.kin = ManipulatorKinematics(urdf_path, n_joints,
                                          q_min=Q_MIN, q_max=Q_MAX)
        # Sync env DOF with actual model loaded by Pinocchio (may differ from n_joints)
        self.n = self.kin.n

        # Per-capsule obstacle distances for observations (n_capsules scalars)
        if urdf_path is not None and self.obs_scene_embed > 0:
            zero_q = np.zeros(self.n)
            try:
                self._capsule_dists_dim = len(self.kin.get_link_capsules(zero_q))
            except Exception:
                self._capsule_dists_dim = 0
        else:
            self._capsule_dists_dim = 0

        if self.obs_scene_embed > 0:
            # Scene-embed observation: no top-K (redundant with scene_embed)
            self.obs_dim = (self.n * 2 + 3 + 3 + 3
                            + self._capsule_dists_dim
                            + self.obs_scene_embed * 4
                            + len(self.obs_waypoint_steps) * 3)
        else:
            self.obs_dim = self.n * 2 + 3 + 3 + 3 + 1 + 1 + 3  # legacy 28-dim
        self.act_dim = self.n  # 7D: 3 (task) + 4 (nullspace, via nullspace basis)

        # Truncate DQ_MAX to match actual DOF
        self._dq_max = DQ_MAX[:self.n]
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

        # MuJoCo setup
        self.mj_model = None
        self.mj_data = None
        if HAS_MUJOCO and xml_path is not None:
            self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
            self.mj_data = mujoco.MjData(self.mj_model)
            print(f"[env] MuJoCo model loaded: {xml_path}")

        # Collision detector (also share with trajectory generator for scene validation)
        self.collision_detector = CollisionDetector(self.mj_model, self.mj_data)
        if self.traj_gen is not None:
            self.traj_gen.collision_detector = self.collision_detector

        # Reward function with collision detection
        self.reward_fn = RewardFunction(
            dt=dt, w_obs=w_obs, w_obs_safe=w_obs_safe,
            w_collision=w_collision, w_track=w_track, w_goal=w_goal,
            w_manip=w_manip, w_action=w_action,
            d_safe=d_safe, d_critical=d_critical, alpha_relax=alpha_relax,
            collision_detector=self.collision_detector)
        self.d_safe = d_safe
        self.success_bonus = success_bonus
        self.w_apf = w_apf
        self.sdf = ObstacleSDF(n_obstacles, obs_radius)

        # Sigma gate parameters (default to reward d_safe/d_critical if not specified)
        self.sigma_d_safe = sigma_d_safe if sigma_d_safe is not None else d_safe
        self.sigma_d_critical = sigma_d_critical if sigma_d_critical is not None else d_critical
        self.sigma_smooth = sigma_smooth
        self._last_sigma = 0.0

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
        self.path_param = 0.0
        return self._get_obs()

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
        action : 10D action [Δẋ_RL (3), dq0 (7)]
                 Δẋ_RL: position-space relaxation velocity (gated by d_obs)
                 dq0  : null-space self-motion velocity

        Returns
        -------
        obs, reward, done, info
        """
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

        else:
            # Decompose 7D action into task relaxation + null-space coefficients
            delta_x_rl = action[:3]   # Δẋ_RL ∈ R^3 (position-space relaxation)
            z          = action[3:]   # z ∈ R^4 (nullspace coefficients, via SVD basis)

            # Compute nominal task-space velocity (PID tracking)
            dx_nom = self._compute_task_velocity()  # ẋ_d + Kp(x_d - x) + Ki*∫(x_d - x)dt

            # Gate operator σ: scales task relaxation based on obstacle distance
            # σ → 0 when safe (d_obs >= sigma_d_safe), σ → 1 when dangerous
            # sigma_override bypasses the gate (used for random exploration in start_steps)
            sigma_ov = getattr(self, 'sigma_override', None)
            if sigma_ov is not None:
                sigma = float(sigma_ov)
            else:
                x_ee_cur, _ = self.kin.forward_kinematics(self.q)
                d_obs_cur = self.sdf.min_distance(x_ee_cur, self.q, kinematics=self.kin)
                band = max(self.sigma_d_safe - self.sigma_d_critical, 1e-6)
                raw_sigma = float(np.clip((self.sigma_d_safe - d_obs_cur) / band, 0.0, 1.0))
                # Smoothstep: C1 continuity at 0 and 1 for smoother gate transitions
                sigma_smooth = raw_sigma * raw_sigma * (3.0 - 2.0 * raw_sigma)
                # Low-pass filter: prevent rapid sigma flickering from causing jitter
                sigma = self.sigma_smooth * self._last_sigma + (1.0 - self.sigma_smooth) * sigma_smooth
            self._last_sigma = sigma
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

        # Save previous joint velocity for smoothness penalty
        prev_dq = self.dq.copy()

        # Integrate (kinematics-only mode)
        q_new = self.q + dq_cmd * self.dt
        dq_new = dq_cmd

        if self.mj_data is not None:
            self._mujoco_step(dq_cmd)
        else:
            self.q = q_new
            self.dq = dq_new

        # Tracking-error-driven path progression with trapezoidal speed profile
        x_ee, _ = self.kin.forward_kinematics(self.q)
        tracking_error = np.linalg.norm(x_ee - self.x_d)

        # Nominal path parameter (trapezoidal: ease-in → constant → ease-out)
        total = self.episode_len
        a_end = int(total * 0.2)
        d_start = int(total * 0.8)

        if self.step_count < a_end:
            t = self.step_count / max(1, a_end)
            nominal_s = t * t * a_end / total  # quadratic ease-in
        elif self.step_count > d_start:
            rem = max(1, total - d_start)
            t = (self.step_count - d_start) / rem
            nominal_s = d_start / total + (2.0 * t - t * t) * rem / total  # quadratic ease-out
        else:
            nominal_s = self.step_count / total  # linear

        # Modulate by tracking error with dead zone and low-pass filter
        # Dead zone: errors < 2cm don't slow progression (prevents accumulating lag)
        err_deadzone = max(0.0, tracking_error - 0.02)
        # path_deadzone: configurable, larger values allow more deviation
        # before path progression stalls (default 0.20 = 22cm total)
        raw_advance = float(np.clip(1.0 - err_deadzone / self.path_deadzone, 0.0, 1.0))
        advance_rate = 0.5 * raw_advance + 0.5 * getattr(self, '_last_advance', raw_advance)
        self._last_advance = advance_rate
        self.path_param = min(1.0, self.path_param + (nominal_s - self.path_param) * advance_rate)

        # Update target position: parametric vs linear trajectory
        if self.use_parametric_traj and self._parametric_pos_func is not None:
            t = self.step_count * self.dt
            prev_x_d = self.x_d.copy()
            self.x_d = self._parametric_pos_func(t)
            self.dx_d[:3] = self._parametric_vel_func(t)
        else:
            # Original linear interpolation with tracking-error-driven progression
            prev_x_d = self.x_d.copy()
            self.x_d = (1.0 - self.path_param) * self.x_start + self.path_param * self.x_goal
            # Feed-forward velocity = actual target motion, decoupled from advance_rate
            self.dx_d[:3] = (self.x_d - prev_x_d) / self.dt

        self.step_count += 1

        # Compute reward
        x_ee, _ = self.kin.forward_kinematics(self.q)
        d_obs = self.sdf.min_distance(x_ee, self.q, kinematics=self.kin)
        d_obs = float(np.clip(d_obs, -0.5, 0.5))  # cap inf for numerical stability
        w = self._manipulability()

        # Record end-effector position for trajectory visualization
        if len(self.ee_trajectory) >= self.max_trajectory_len:
            self.ee_trajectory.pop(0)
        self.ee_trajectory.append(x_ee.copy())

        # APF reward: per-link repulsive force mapped through link Jacobians
        # (MPC-style Khatib potential field — Khatib 1986)
        r_apf = 0.0
        if self.w_apf > 0.0 and self.sdf.n_obs > 0:
            capsules = self.kin.get_link_capsules(self.q)
            link_names = [
                "panda_link0", "panda_link1", "panda_link2", "panda_link3",
                "panda_link4", "panda_link5", "panda_link5",
                "panda_link6", "panda_link7", "panda_hand",
                "panda_leftfinger", "panda_rightfinger",
            ]
            link_jacs = self.kin.link_jacobians_position(self.q, list(set(link_names)))

            dq_rep = np.zeros(self.n)
            apf_d_safe = max(self.d_safe * 5, 0.15)  # repulsive field range
            rep_gain = 0.5
            n_active = 0

            for ci, (p1, p2, cap_r) in enumerate(capsules):
                link_name = link_names[ci] if ci < len(link_names) else None
                if link_name not in link_jacs:
                    continue
                J_link = link_jacs[link_name]  # [3×n]

                for pt in [p1, (p1 + p2) / 2, p2]:
                    F = np.zeros(3)
                    for i in range(self.sdf.n_obs):
                        diff = pt - self.sdf.centers[i]
                        dist = np.linalg.norm(diff)
                        if dist < 1e-8:
                            continue
                        d_signed = dist - self.sdf.radii[i]
                        if 0 < d_signed < apf_d_safe:
                            magnitude = rep_gain * (1.0 / d_signed - 1.0 / apf_d_safe) / (d_signed * d_signed)
                            F += magnitude * diff / dist

                    F_norm = np.linalg.norm(F)
                    if F_norm < 1e-6:
                        continue
                    # Clip per-point force for stability
                    if F_norm > 2.0:
                        F = F / F_norm * 2.0
                    # Map to joint space via this link's Jacobian pseudo-inverse
                    Jpinv = self.kin.pseudo_inverse(J_link)
                    dq_rep += Jpinv @ F
                    n_active += 1

            if n_active > 0:
                dq_rep /= n_active  # average over active points
                rep_norm = np.linalg.norm(dq_rep)
                if rep_norm > 1e-6:
                    # Scalar projection of actual dq onto APF direction (m/s in joint space)
                    proj = float(np.dot(self.dq, dq_rep)) / rep_norm
                    r_apf = self.w_apf * max(0.0, proj)

        reward, reward_info = self.reward_fn.compute(
            q=self.q, dq=self.dq, x_ee=x_ee,
            x_d=self.x_d, dx_d=self.dx_d,
            d_obs=d_obs, w=w, x_goal=self.x_goal,
            action=action, prev_dq=prev_dq,
        )
        reward += r_apf
        reward_info["r_apf"] = r_apf
        # Collision detection: use MuJoCo collision detector from reward_info;
        # fall back to SDF distance when MuJoCo is unavailable
        if self.mj_model is not None:
            collision = (reward_info.get("n_obstacle_contacts", 0) > 0 or
                         reward_info.get("n_self_contacts", 0) > 0)
        else:
            collision = d_obs < 0.02

        # Track cumulative collision flag for the entire episode
        self._ever_collided = self._ever_collided or collision

        # Termination conditions
        if self.use_parametric_traj:
            path_complete = self.step_count >= self.episode_len
        else:
            path_complete = self.path_param >= 0.99
        done = self.step_count >= self.episode_len or path_complete
        if self.collision_term:
            done = done or collision

        # Sparse success bonus when reaching goal (only if no collision)
        if path_complete and not self._ever_collided:
            reward += self.success_bonus

        tracking_error = float(np.linalg.norm(x_ee - self.x_d))
        info = {"d_obs": d_obs, "w": w, "success": path_complete and not self._ever_collided, "collision": collision,
                "path_param": self.path_param, "tracking_error": tracking_error, **reward_info}

        return self._get_obs(), reward, done, info

    def render(self, show_robot: bool = True):
        """Launch or sync the passive MuJoCo viewer and draw end-effector trajectory.

        Parameters
        ----------
        show_robot : if False, hide the robot's visual geometry (only capsules shown)
        """
        if self.mj_model is None:
            return
        if not hasattr(self, '_viewer'):
            self._viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        if self._viewer.is_running():
            # Hide only robot body geoms by setting alpha to 0
            # Identify robot geoms by body name (Panda links start with "panda_")
            for i in range(self.mj_model.ngeom):
                body_id = self.mj_model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                # Hide geoms belonging to robot bodies (typically contain "panda" or "link")
                if body_name and ("panda" in body_name.lower() or "link" in body_name.lower()):
                    self.mj_model.geom_rgba[i, 3] = 1.0 if show_robot else 0.0
            # Draw visualizations
            self._draw_visualizations()
            self._viewer.sync()

    def _draw_visualizations(self):
        """Draw visualizations: obstacles, fixed target point, EE trajectory, and link capsules."""
        scene = self._viewer.user_scn
        scene.ngeom = 0  # Clear previous geometries

        # 1. Draw link capsules (semi-transparent blue)
        capsules = self.kin.get_link_capsules(self.q)
        for p1, p2, cap_radius in capsules:
        # for p1, p2, cap_radius in [capsules[-1]]:
            if scene.ngeom >= scene.maxgeom:
                break

            # Capsule center and orientation
            center = (p1 + p2) / 2
            length = np.linalg.norm(p2 - p1)

            if length > 1e-6:
                # Compute rotation matrix to align z-axis with capsule direction
                direction = (p2 - p1) / length
                z_axis = np.array([0, 0, 1])

                # Rotation axis: cross product
                rot_axis = np.cross(z_axis, direction)
                rot_axis_norm = np.linalg.norm(rot_axis)

                if rot_axis_norm > 1e-6:
                    rot_axis = rot_axis / rot_axis_norm
                    # Rotation angle
                    cos_angle = np.dot(z_axis, direction)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

                    # Rodrigues' rotation formula
                    K = np.array([
                        [0, -rot_axis[2], rot_axis[1]],
                        [rot_axis[2], 0, -rot_axis[0]],
                        [-rot_axis[1], rot_axis[0], 0]
                    ])
                    rot_mat = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                else:
                    # Already aligned or opposite
                    rot_mat = np.eye(3) if np.dot(z_axis, direction) > 0 else np.diag([1, 1, -1])
            else:
                rot_mat = np.eye(3)
                length = 0.001  # Avoid zero length

            # MuJoCo capsule size: [radius, half_length, 0]
            size = np.array([cap_radius, length / 2, 0])

            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                size, center, rot_mat.flatten(),
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
        self.step_count = 0
        self._integral_err = np.zeros(3)
        self._ever_collided = False
        self.reward_fn._prev_dist_to_goal = None  # reset goal distance tracking

        # Initialize physics loss storage fields (set during step())
        self._last_J = np.zeros((3, self.n), dtype=np.float32)
        self._last_sigma = np.float32(0.0)
        self._last_dx_nom = np.zeros(3, dtype=np.float32)

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

        # Desired velocity (towards goal)
        direction = self.x_goal - self.x_start
        distance = np.linalg.norm(direction)
        if distance > 1e-6:
            self.dx_d = (direction / distance) * 0.1  # 0.1 m/s
        else:
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
                self.q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])

        self.dq = np.zeros(self.n)
        self._sync_obstacles_to_mujoco()

        # Reset MuJoCo state
        if self.mj_data is not None:
            self.mj_data.qpos[:self.n] = self.q
            self.mj_data.qvel[:self.n] = self.dq
            self.mj_data.qpos[self.n:self.n + 2] = 0.0
            self.mj_data.qvel[self.n:self.n + 2] = 0.0
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
        self.dx_d = np.array([0.0, 0.0, -0.1])

        # IK for initial configuration
        q_init = self.kin.inverse_kinematics(
            np.concatenate([self.x_start, np.array([0, 0, 0, 1])])
        )
        if q_init is not None:
            self.q = q_init
        else:
            print("[env] WARNING: IK failed, using home pose")
            self.q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])

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
            self.mj_data.qpos[self.n:self.n + 2] = 0.0
            self.mj_data.qvel[self.n:self.n + 2] = 0.0
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

        # Keep fingers closed
        self.mj_data.qpos[self.n:self.n + 2] = 0.0
        self.mj_data.qvel[self.n:self.n + 2] = 0.0

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
        x_ee, _ = self.kin.forward_kinematics(self.q)
        w = self._manipulability()

        if self.obs_scene_embed > 0:
            # Future waypoints along the planned path
            waypoints = []
            for s in self.obs_waypoint_steps:
                if self.use_parametric_traj and self._parametric_pos_func is not None:
                    t = (self.step_count + s) * self.dt
                    wp = self._parametric_pos_func(t)
                else:
                    future_param = min(1.0, self.path_param + s / self.episode_len)
                    wp = (1.0 - future_param) * self.x_start + future_param * self.x_goal
                waypoints.append(wp)

            # Scene embedding: all obstacle positions (relative to EE) and radii
            # Provides full global layout — no top-K (redundant with full scene)
            scene_embed = np.zeros(self.obs_scene_embed * 4, dtype=np.float32)
            n_embed = min(self.obs_scene_embed, self.sdf.n_obs)
            for i in range(n_embed):
                rel_pos = self.sdf.centers[i] - x_ee
                scene_embed[i*4:i*4+3] = rel_pos
                scene_embed[i*4+3] = self.sdf.radii[i]

            # Per-capsule minimum distances to nearest obstacle
            # (n_capsules scalars — direct collision signal for each link)
            capsule_dists = self.sdf.per_capsule_distances(self.q, self.kin)

            # State: [q(7), dq(7), x_ee(3), x_d(3), dx_d(3),
            #         wp_1(3), ..., wp_N(3),
            #         capsule_dists(n_caps),
            #         scene_embed(N_obs * 4)]
            obs = np.concatenate([
                self.q, self.dq, x_ee, self.x_d, self.dx_d[:3],
                *waypoints,
                capsule_dists,
                scene_embed,
            ])
        else:
            # Legacy observation
            d_obs = self.sdf.min_distance(x_ee, self.q, kinematics=self.kin)
            d_obs = float(np.clip(d_obs, -0.5, 0.5))

            # Direction to nearest obstacle center (from EE)
            obs_dir = np.zeros(3, dtype=np.float32)
            if self.sdf.n_obs > 0:
                dists = np.linalg.norm(self.sdf.centers - x_ee, axis=1)
                nearest = self.sdf.centers[np.argmin(dists)]
                delta = nearest - x_ee
                norm = np.linalg.norm(delta)
                if norm > 1e-6:
                    obs_dir = delta / norm

            # State: [q(7), dq(7), x_ee(3), x_d(3), dx_d(3), d_obs(1), w(1), obs_dir(3)] = 28
            obs = np.concatenate([
                self.q, self.dq, x_ee, self.x_d, self.dx_d[:3],
                [d_obs], [w], obs_dir
            ])

        return obs.astype(np.float32)

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
