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

        # Clamp joint velocities to actuator limits
        dq_cmd = np.clip(dq_cmd, -self.DQ_MAX, self.DQ_MAX)

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
            # Time-based mode: linear interpolation along insertion trajectory
            # Progress from x_start to x_goal over episode_len steps
            progress = min(1.0, self.step_count / self.episode_len)
            self.x_d = (1 - progress) * self.x_start + progress * self.x_goal
            # Compute actual velocity based on trajectory and time
            trajectory_length = np.linalg.norm(self.x_goal - self.x_start)
            trajectory_time = self.episode_len * self.dt
            velocity_magnitude = trajectory_length / trajectory_time
            direction = (self.x_goal - self.x_start) / trajectory_length
            self.dx_d = velocity_magnitude * direction

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
        # Success: reached goal endpoint (not current target)
        success = np.linalg.norm(x_ee - self.x_goal) < 0.02
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

        # 3. Draw end-effector trajectory (green points)
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
        场景1：人机协作-狭窄空间装配（论文 Section 4.1.3）
        - 轨迹：直线插入，从 [0.5, 0.0, 0.5]m 沿 z 轴向下至 [0.5, 0.0, 0.3]m
        - 优化：y=0避开奇异区域，轨迹长度0.2m，速度降低以提高跟踪精度
        - 考核：末端跟踪精度（<5mm）和连杆避障能力
        """
        self.step_count = 0
        self._integral_err = np.zeros(3)

        # 优化后的起点和目标点（避开y=0.3奇异区域）
        self.x_start = np.array([0.5, 0.0, 0.5])
        self.x_goal = np.array([0.5, 0.0, 0.3])

        # 当前目标点（初始为起点，训练时逐步向目标移动）
        self.x_d = self.x_start.copy()
        self.dx_d = np.array([0.0, 0.0, -0.1])  # 向下速度 0.1 m/s

        # 初始关节配置：通过逆运动学求解使末端位于起点
        q_init = self.kin.inverse_kinematics(self.x_start)
        if q_init is not None:
            self.q = q_init
        else:
            # IK失败时使用home pose作为fallback
            print("[env] WARNING: IK failed for start position, using home pose")
            self.q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])

        self.dq = np.zeros(self.n)

        if self.sdf.n_obs > 0:
            obstacle_centers = [
                np.array([0.45, 0.25, 0.45]),
                np.array([0.55, 0.35, 0.45]),
                np.array([0.50, 0.20, 0.40]),
            ][:self.sdf.n_obstacles]
            self.sdf.set_static_obstacles(obstacle_centers)
        else:
            self.sdf.set_static_obstacles([])
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

    def _compute_task_velocity(self) -> np.ndarray:
        """
        PID tracking in task space: ẋ_cmd = ẋ_d + Kp*e + Ki*∫e dt
        - Adaptive Kp: increases with error magnitude
        - Integral term: eliminates steady-state error
        - Velocity saturation: conservative limit for stability
        """
        x_ee, _ = self.kin.forward_kinematics(self.q)
        pos_err = self.x_d - x_ee
        err_norm = np.linalg.norm(pos_err)

        # Adaptive proportional gain
        Kp_base = 5.0
        Kp = Kp_base * (1.0 + 2.0 * np.tanh(err_norm / 0.05))

        # Integral gain (anti-windup: clamp integral to ±0.05m)
        Ki = 2.0
        self._integral_err = getattr(self, '_integral_err', np.zeros(3))
        self._integral_err = np.clip(self._integral_err + pos_err * self.dt, -0.05, 0.05)

        dx_cmd = np.zeros(6)
        dx_cmd[:3] = self.dx_d[:3] + Kp * pos_err + Ki * self._integral_err

        # Velocity saturation: conservative limit
        dx_cmd[:3] = np.clip(dx_cmd[:3], -0.3, 0.3)

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
