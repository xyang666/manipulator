"""
manipulator_env.py
------------------
MuJoCo-based 7-DOF manipulator environment with:
  - 13D action space: [Δẋ_RL (6), dq0 (7)] — task relaxation + null-space self-motion
  - Dense reward combining tracking, obstacle avoidance, manipulability, energy
  - Signed distance field (simplified sphere model) for obstacle detection
  - Time-decoupled path tracking mode (parameterized by s ∈ [0,1])

Observation space (paper Eq. state):
    Standard mode: s = [q (7), dq (7), x_ee (3), x_d (3), dx_d (3), d_obs (1), w(q) (1)]  dim=25
    Time-decoupled: same + s (1)  dim=26

Action space (paper):
    a = [Δẋ_RL ∈ R^6, dq0 ∈ R^7]  dim=13
    Control law: q̇ = J†(ẋ_d + Kp(x_d - x) + diag(σ)·Δẋ_RL) + N(q)dq0
    Gate operator diag(σ): scaled by d_obs (σ→0 when safe, σ→1 when dangerous)
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

    def __init__(self,
                 urdf_path: Optional[str] = None,
                 xml_path: Optional[str] = None,
                 n_joints: int = 7,
                 dt: float = 0.02,
                 episode_len: int = 200,
                 n_obstacles: int = 3,
                 obs_radius: float = 0.1,
                 use_mpc: bool = False,
                 mpc_horizon: int = 10,
                 time_decoupled: bool = False,
                 path_progress_threshold: float = 0.02,
                 d_critical: float = 0.05,
                 alpha_relax: float = 0.1):
        """
        Parameters
        ----------
        urdf_path   : URDF for kinematics/dynamics (Pinocchio)
        xml_path    : MuJoCo XML model path
        dt          : simulation timestep (s)
        episode_len : max steps per episode
        n_obstacles : number of spherical obstacles
        obs_radius  : obstacle radius (m)
        use_mpc     : whether to use MPC controller
        mpc_horizon : MPC prediction horizon
        time_decoupled : if True, use parameterized path (s ∈ [0,1]) instead of time-based trajectory
        path_progress_threshold : distance threshold to advance path parameter s
        """
        self.n = n_joints
        self.dt = dt
        self.episode_len = episode_len
        self.time_decoupled = time_decoupled
        self.path_progress_threshold = path_progress_threshold

        # Observation: [q(7), dq(7), x_ee(3), x_d(3), dx_d(3), d_obs(1), w(1)] = 25
        self.obs_dim = n_joints * 2 + 3 + 3 + 3 + 1 + 1  # 25
        if time_decoupled:
            self.obs_dim += 1  # add s to observation
        self.act_dim = 6 + n_joints  # 13D: 6 for task relaxation + 7 for null-space

        self.kin = ManipulatorKinematics(urdf_path, n_joints)
        self.dyn = ManipulatorDynamics(urdf_path, n_joints)

        # MuJoCo setup
        self.mj_model = None
        self.mj_data = None
        if HAS_MUJOCO and xml_path is not None:
            self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
            self.mj_data = mujoco.MjData(self.mj_model)
            print(f"[env] MuJoCo model loaded: {xml_path}")

        # Collision detector
        self.collision_detector = CollisionDetector(self.mj_model, self.mj_data)

        # Reward function with collision detection
        self.reward_fn = RewardFunction(dt=dt, collision_detector=self.collision_detector,
                                        d_critical=d_critical, alpha_relax=alpha_relax)
        self.sdf = ObstacleSDF(n_obstacles, obs_radius)

        # MPC controller (optional)
        self.use_mpc = use_mpc
        self.mpc = None
        if use_mpc and HAS_MPC:
            self.mpc = MPCController(
                n_states=n_joints * 2,
                n_controls=n_joints,
                horizon=mpc_horizon,
                dt=dt
            )
            print(f"[env] MPC controller enabled with horizon={mpc_horizon}")
        elif use_mpc and not HAS_MPC:
            print("[env] WARNING: MPC requested but not available")

        # End-effector trajectory tracking
        self.ee_trajectory = []
        self.max_trajectory_len = 500

        # Path parameterization (time-decoupled mode)
        self.path_param = 0.0  # s ∈ [0, 1]
        self.path_waypoints = []  # List of 3D positions defining the path

        self._reset_state()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self._reset_state()
        self.ee_trajectory.clear()

        # Initialize path parameter
        if self.time_decoupled:
            self.path_param = 0.0
            self._generate_path()

        return self._get_obs()

    def step(self, action: np.ndarray):
        """
        Parameters
        ----------
        action : 13D action [Δẋ_RL (6), dq0 (7)]
                 Δẋ_RL: task-space relaxation velocity (gated by d_obs)
                 dq0  : null-space self-motion velocity

        Returns
        -------
        obs, reward, done, info
        """
        if self.use_mpc and self.mpc is not None:
            # MPC mode: directly optimize task-space tracking
            dq_cmd = self.mpc.compute_control_task_space(
                self.q, self.dq, self.x_d, self.dx_d, self.kin
            )
        else:
            # Decompose 13D action into task relaxation + null-space components
            delta_x_rl = action[:6]   # Δẋ_RL ∈ R^6 (task-space relaxation)
            dq0        = action[6:]   # dq0 ∈ R^7 (null-space self-motion)

            # Compute nominal task-space velocity (PD tracking)
            dx_nom = self._compute_task_velocity()  # ẋ_d + Kp(x_d - x)

            # Gate operator σ: scales task relaxation based on obstacle distance
            # σ → 0 when safe (d_obs >= d_safe), σ → 1 when dangerous
            x_ee_cur, _ = self.kin.forward_kinematics(self.q)
            d_obs_cur = self.sdf.min_distance(x_ee_cur, self.q, kinematics=self.kin)
            d_safe = 0.10
            d_critical = 0.05
            sigma = float(np.clip((d_safe - d_obs_cur) / (d_safe - d_critical + 1e-6), 0.0, 1.0))
            delta_x_gated = sigma * delta_x_rl  # diag(σ) · Δẋ_RL

            # Combine: q̇ = J†(dx_nom + delta_x_gated) + N(q)dq0
            dq_cmd = self.kin.combine_velocities_with_relaxation(
                self.q, dx_nom, delta_x_gated, dq0
            )

        # Integrate (kinematics-only mode)
        q_new = self.q + dq_cmd * self.dt
        dq_new = dq_cmd

        if self.mj_data is not None:
            self._mujoco_step(dq_cmd)
        else:
            self.q = q_new
            self.dq = dq_new

        # Update target based on mode
        if self.time_decoupled:
            # Update path parameter based on tracking error
            x_ee, _ = self.kin.forward_kinematics(self.q)
            target_pos = self._get_path_position(self.path_param)
            tracking_error = np.linalg.norm(x_ee - target_pos)

            # Advance path parameter if close enough to current target
            if tracking_error < self.path_progress_threshold:
                self.path_param = min(1.0, self.path_param + 0.01)

            # Update target to current path position
            self.x_d = self._get_path_position(self.path_param)
            self.dx_d = np.zeros(3)  # No velocity reference in time-decoupled mode
        else:
            # Time-based mode: target is static or time-indexed
            pass

        self.step_count += 1

        # Compute reward
        x_ee, _ = self.kin.forward_kinematics(self.q)
        d_obs = self.sdf.min_distance(x_ee, self.q, kinematics=self.kin)
        w = self._manipulability()

        # Record end-effector position for trajectory visualization
        if len(self.ee_trajectory) >= self.max_trajectory_len:
            self.ee_trajectory.pop(0)
        self.ee_trajectory.append(x_ee.copy())

        reward, reward_info = self.reward_fn.compute(
            q=self.q, dq=self.dq, x_ee=x_ee,
            x_d=self.x_d, dx_d=self.dx_d,
            d_obs=d_obs, w=w
        )
        success = np.linalg.norm(x_ee - self.x_d) < 0.02
        collision = d_obs < 0.02

        # Termination conditions
        if self.time_decoupled:
            # Success if reached end of path
            path_complete = self.path_param >= 0.99
            done = self.step_count >= self.episode_len or collision or path_complete
            info = {"d_obs": d_obs, "w": w, "success": path_complete, "collision": collision,
                    "path_param": self.path_param, **reward_info}
        else:
            done = self.step_count >= self.episode_len or collision or success
            info = {"d_obs": d_obs, "w": w, "success": success, "collision": collision, **reward_info}

        return self._get_obs(), reward, done, info

    def render(self):
        """Launch or sync the passive MuJoCo viewer and draw end-effector trajectory."""
        if self.mj_model is None:
            return
        if not hasattr(self, '_viewer'):
            self._viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        if self._viewer.is_running():
            # Draw visualizations
            self._draw_visualizations()
            self._viewer.sync()

    def _draw_visualizations(self):
        """Draw visualizations: obstacles, fixed target point, and EE trajectory."""
        scene = self._viewer.user_scn
        scene.ngeom = 0  # Clear previous geometries

        # 1. Draw obstacles (semi-transparent red spheres)
        for obs_center in self.sdf.centers:
            if scene.ngeom >= scene.maxgeom:
                break

            size = np.array([self.sdf.radius, 0, 0])

            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size, obs_center, np.eye(3).flatten(),
                np.array([1.0, 0.0, 0.0, 0.3])  # Red, semi-transparent
            )
            scene.ngeom += 1

        # 2. Draw fixed target point (yellow sphere, larger)
        if scene.ngeom < scene.maxgeom:
            size = np.array([0.02, 0, 0])  # Larger sphere for fixed target

            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size, self.x_d, np.eye(3).flatten(),
                np.array([1.0, 1.0, 0.0, 1.0])  # Yellow
            )
            scene.ngeom += 1

        # 3. Draw end-effector trajectory (red points)
        if len(self.ee_trajectory) < 2:
            return

        for i in range(len(self.ee_trajectory) - 1):
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
        self.dx_d = np.zeros(3)
        self.step_count = 0

        # Randomize target position in reachable workspace
        self.x_d = np.array([
            np.random.uniform(0.4, 0.7),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0.3, 0.7),
        ])

        # Initial joint configuration (home pose)
        self.q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
        self.dq = np.zeros(self.n)

        # Randomize obstacle positions between start EE and target,
        # keeping a minimum clearance from both endpoints
        q_home = self.q.copy()
        x_home, _ = self.kin.forward_kinematics(q_home)
        clearance = self.sdf.radius + 0.08  # keep obstacles away from endpoints

        centers = []
        max_tries = 200
        n_placed = 0
        for _ in range(max_tries):
            if n_placed >= self.sdf.n_obs:
                break
            # Sample in the bounding box of [home, target]
            lo = np.minimum(x_home, self.x_d) - 0.05
            hi = np.maximum(x_home, self.x_d) + 0.05
            c = np.random.uniform(lo, hi)
            # Enforce workspace z-floor
            c[2] = max(c[2], 0.15)
            # Reject if too close to home EE or target
            if (np.linalg.norm(c - x_home) < clearance or
                    np.linalg.norm(c - self.x_d) < clearance):
                continue
            # Reject if too close to already-placed obstacles
            if any(np.linalg.norm(c - p) < 2 * self.sdf.radius + 0.02 for p in centers):
                continue
            centers.append(c)
            n_placed += 1

        # Pad with fallback positions if not enough were sampled
        fallbacks = [np.array([0.4, 0.1, 0.4]),
                     np.array([0.5, 0.0, 0.5]),
                     np.array([0.5, 0.2, 0.6])]
        for fb in fallbacks:
            if len(centers) >= self.sdf.n_obs:
                break
            centers.append(fb)

        self.sdf.set_static_obstacles(centers[:self.sdf.n_obs])
        self._sync_obstacles_to_mujoco()

        # Reset MuJoCo state and clamp fingers closed
        if self.mj_data is not None:
            self.mj_data.qpos[:self.n] = self.q
            self.mj_data.qvel[:self.n] = self.dq
            self.mj_data.qpos[self.n:self.n + 2] = 0.0
            self.mj_data.qvel[self.n:self.n + 2] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)  # Update kinematics

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
                self.mj_model.geom_size[gid, 0] = self.sdf.radius

    def _get_target_trajectory_point(self, theta: float):
        """
        Get a point on the target circular trajectory.

        Parameters
        ----------
        theta : angle in radians

        Returns
        -------
        point : [3] position on the circle
        """
        # Build circle in the plane perpendicular to target_axis
        if self.target_axis == 0:  # YZ plane (perpendicular to X)
            offset = np.array([
                0,
                self.target_radius * np.cos(theta),
                self.target_radius * np.sin(theta)
            ])
        elif self.target_axis == 1:  # XZ plane (perpendicular to Y)
            offset = np.array([
                self.target_radius * np.cos(theta),
                0,
                self.target_radius * np.sin(theta)
            ])
        else:  # XY plane (perpendicular to Z)
            offset = np.array([
                self.target_radius * np.cos(theta),
                self.target_radius * np.sin(theta),
                0
            ])
        return self.target_center + offset

    def _get_target_velocity(self, theta: float):
        """
        Get velocity on the target circular trajectory.

        Parameters
        ----------
        theta : angle in radians

        Returns
        -------
        velocity : [6] velocity (linear + angular)
        """
        dx_d = np.zeros(6)

        # Velocity is tangent to circle: derivative of position w.r.t. theta
        if self.target_axis == 0:  # YZ plane
            dx_d[1] = -self.target_radius * self.target_omega * np.sin(theta)
            dx_d[2] =  self.target_radius * self.target_omega * np.cos(theta)
        elif self.target_axis == 1:  # XZ plane
            dx_d[0] = -self.target_radius * self.target_omega * np.sin(theta)
            dx_d[2] =  self.target_radius * self.target_omega * np.cos(theta)
        else:  # XY plane
            dx_d[0] = -self.target_radius * self.target_omega * np.sin(theta)
            dx_d[1] =  self.target_radius * self.target_omega * np.cos(theta)

        return dx_d

    def _target_pose(self, t: float):
        """
        Circular end-effector trajectory at time t.

        Returns
        -------
        x_d : [3] desired position
        dx_d : [6] desired velocity
        """
        theta = self.target_omega * t
        x_d = self._get_target_trajectory_point(theta)
        dx_d = self._get_target_velocity(theta)
        return x_d, dx_d

    def _advance_target(self):
        t = self.step_count * self.dt
        self.x_d, self.dx_d = self._target_pose(t)

    def _compute_task_velocity(self) -> np.ndarray:
        """PD tracking in task space: ẋ_cmd = ẋ_d + Kp*(x_d - x_ee)"""
        x_ee, _ = self.kin.forward_kinematics(self.q)
        Kp = 30.0  # Balanced gain for stable tracking
        dx_cmd = np.zeros(6)
        dx_cmd[:3] = self.dx_d[:3] + Kp * (self.x_d - x_ee)
        return dx_cmd

    def _mujoco_step(self, dq_cmd):
        # Computed torque control with PD feedback
        # Integrate velocity command to get desired position
        q_desired = self.q + dq_cmd * self.dt

        # PD control: ddq = Kp*(q_desired - q) + Kd*(dq_cmd - dq)
        Kp = 50.0  # Position gain
        Kd = 10.0  # Velocity gain
        ddq_desired = Kp * (q_desired - self.q) + Kd * (dq_cmd - self.dq)

        # Compute required torque using inverse dynamics
        tau = self.dyn.compute_torque(self.q, self.dq, ddq_desired)

        self.mj_data.ctrl[:self.n] = tau

        # Keep fingers closed
        self.mj_data.qpos[self.n:self.n + 2] = 0.0
        self.mj_data.qvel[self.n:self.n + 2] = 0.0

        mujoco.mj_step(self.mj_model, self.mj_data)
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
        d_obs = self.sdf.min_distance(x_ee, self.q, kinematics=self.kin)
        w = self._manipulability()

        # State: [q(7), dq(7), x_ee(3), x_d(3), dx_d(3), d_obs(1), w(1)] = 25
        obs = np.concatenate([
            self.q, self.dq, x_ee, self.x_d, self.dx_d[:3],
            [d_obs], [w]
        ])

        # Add path parameter if time-decoupled
        if self.time_decoupled:
            obs = np.concatenate([obs, [self.path_param]])

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

    def _generate_path(self):
        """Generate a parameterized path from start to goal."""
        x_start, _ = self.kin.forward_kinematics(self.q)

        # Generate waypoints (e.g., straight line or curved path)
        n_waypoints = 20
        self.path_waypoints = []

        for i in range(n_waypoints + 1):
            alpha = i / n_waypoints
            # Linear interpolation (can be replaced with spline/bezier)
            waypoint = (1 - alpha) * x_start + alpha * self.x_d
            self.path_waypoints.append(waypoint)

    def _get_path_position(self, s: float) -> np.ndarray:
        """
        Get position on path at parameter s ∈ [0, 1].

        Parameters
        ----------
        s : path parameter (0 = start, 1 = end)

        Returns
        -------
        position : [3] position on path
        """
        if len(self.path_waypoints) == 0:
            return self.x_d

        # Map s to waypoint index
        idx_float = s * (len(self.path_waypoints) - 1)
        idx = int(np.floor(idx_float))
        alpha = idx_float - idx

        # Clamp to valid range
        idx = max(0, min(idx, len(self.path_waypoints) - 2))

        # Linear interpolation between waypoints
        p0 = self.path_waypoints[idx]
        p1 = self.path_waypoints[idx + 1]

        return (1 - alpha) * p0 + alpha * p1

if __name__ == "__main__":
    env = ManipulatorEnv()
    obs = env.reset()
    print(f"obs shape: {obs.shape}  (expected ({env.obs_dim},))")
    action = np.zeros(env.n)
    obs, r, done, info = env.step(action)
    print(f"step ok  reward={r:.4f}  d_obs={info['d_obs']:.3f}")
    print("manipulator_env.py unit test PASSED")
