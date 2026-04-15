"""
manipulator_env.py
------------------
MuJoCo-based 7-DOF manipulator environment with:
  - Null-space decomposed action space (RL only controls self-motion)
  - Dense reward combining tracking, obstacle avoidance, manipulability, energy
  - Signed distance field (simplified sphere model) for obstacle detection

Observation space:
    s = [q (7), dq (7), x_d (3), dx_d (3), d_obs (1), w(q) (1)]  dim=22

Action space:
    a = dq0 ∈ R^7  (null-space self-motion velocities)
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
                 obs_radius: float = 0.1):
        """
        Parameters
        ----------
        urdf_path   : URDF for kinematics/dynamics (Pinocchio)
        xml_path    : MuJoCo XML model path
        dt          : simulation timestep (s)
        episode_len : max steps per episode
        n_obstacles : number of spherical obstacles
        obs_radius  : obstacle radius (m)
        """
        self.n = n_joints
        self.dt = dt
        self.episode_len = episode_len
        self.obs_dim = n_joints * 2 + 3 + 3 + 1 + 1  # 22
        self.act_dim = n_joints

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
        self.reward_fn = RewardFunction(dt=dt, collision_detector=self.collision_detector)
        self.sdf = ObstacleSDF(n_obstacles, obs_radius)

        # End-effector trajectory tracking
        self.ee_trajectory = []
        self.max_trajectory_len = 500

        self._reset_state()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self._reset_state()
        self.ee_trajectory.clear()
        return self._get_obs()

    def step(self, action: np.ndarray):
        """
        Parameters
        ----------
        action : null-space velocity command dq0 ∈ R^7

        Returns
        -------
        obs, reward, done, info
        """
        dq0 = np.clip(action, -DQ_MAX, DQ_MAX)

        # Combine task-space tracking + null-space RL action
        dx_desired = self._compute_task_velocity()
        dq_cmd = self.kin.combine_velocities(self.q, dx_desired, dq0)

        # Integrate
        q_new = np.clip(self.q + dq_cmd * self.dt, Q_MIN, Q_MAX)
        dq_new = dq_cmd

        if self.mj_data is not None:
            self._mujoco_step(dq_cmd)
        else:
            self.q = q_new
            self.dq = dq_new

        # Update target trajectory
        self._advance_target()
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

        done = self.step_count >= self.episode_len or d_obs < 0.02
        info = {"d_obs": d_obs, "w": w, **reward_info}

        return self._get_obs(), reward, done, info

    def render(self):
        """Launch or sync the passive MuJoCo viewer and draw end-effector trajectory."""
        if self.mj_model is None:
            return
        if not hasattr(self, '_viewer'):
            self._viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        if self._viewer.is_running():
            # Draw end-effector trajectory
            self._draw_ee_trajectory()
            self._viewer.sync()

    def _draw_ee_trajectory(self):
        """Draw end-effector trajectory as connected line segments."""
        if len(self.ee_trajectory) < 2:
            return

        # Use MuJoCo's scene visualization to draw lines
        scene = self._viewer.user_scn

        # Clear previous trajectory lines
        scene.ngeom = 0

        # Draw trajectory as line segments
        for i in range(len(self.ee_trajectory) - 1):
            if scene.ngeom >= scene.maxgeom:
                break

            p1 = self.ee_trajectory[i]
            p2 = self.ee_trajectory[i + 1]

            # Add line segment
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                np.array([0.8, 0.2, 0.2, 1.0])  # Red color with alpha
            )

            # Set line endpoints
            scene.geoms[scene.ngeom].pos[:] = p1
            scene.geoms[scene.ngeom].mat[:3] = p2 - p1
            scene.geoms[scene.ngeom].size[0] = 0.002  # Line thickness

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
        self.q = np.zeros(self.n)
        self.dq = np.zeros(self.n)
        self.step_count = 0

        # Random target trajectory: circular motion in Cartesian space
        self.target_center = np.array([0.5, 0.0, 0.5])
        self.target_radius = 0.15
        self.target_phase = 0.0
        self.target_omega = 0.5  # rad/s

        self.x_d, self.dx_d = self._target_pose(0.0)

        # Random obstacle positions
        self.sdf.randomize_obstacles(center=self.target_center, margin=0.3)
        self._sync_obstacles_to_mujoco()

        # Reset MuJoCo state and clamp fingers closed
        if self.mj_data is not None:
            self.mj_data.qpos[:self.n] = self.q
            self.mj_data.qvel[:self.n] = self.dq
            self.mj_data.qpos[self.n:self.n + 2] = 0.0
            self.mj_data.qvel[self.n:self.n + 2] = 0.0

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

    def _target_pose(self, t: float):
        """Circular end-effector trajectory."""
        x_d = self.target_center + np.array([
            self.target_radius * np.cos(self.target_omega * t), 0,
            self.target_radius * np.sin(self.target_omega * t)
        ])
        dx_d = np.zeros(6)
        dx_d[0] = -self.target_radius * self.target_omega * np.sin(self.target_omega * t)
        dx_d[2] =  self.target_radius * self.target_omega * np.cos(self.target_omega * t)
        return x_d, dx_d

    def _advance_target(self):
        t = self.step_count * self.dt
        self.x_d, self.dx_d = self._target_pose(t)

    def _compute_task_velocity(self) -> np.ndarray:
        """PD tracking in task space: ẋ_cmd = ẋ_d + Kp*(x_d - x_ee)"""
        x_ee, _ = self.kin.forward_kinematics(self.q)
        Kp = 5.0
        dx_cmd = np.zeros(6)
        dx_cmd[:3] = self.dx_d[:3] + Kp * (self.x_d - x_ee)
        return dx_cmd

    def _mujoco_step(self, dq_cmd):
        self.mj_data.qvel[:self.n] = dq_cmd
        # Keep fingers closed: zero out finger qpos and qvel (indices n and n+1)
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
        return np.concatenate([
            self.q, self.dq, self.x_d, self.dx_d[:3],
            [d_obs], [w]
        ]).astype(np.float32)


if __name__ == "__main__":
    env = ManipulatorEnv()
    obs = env.reset()
    print(f"obs shape: {obs.shape}  (expected ({env.obs_dim},))")
    action = np.zeros(env.n)
    obs, r, done, info = env.step(action)
    print(f"step ok  reward={r:.4f}  d_obs={info['d_obs']:.3f}")
    print("manipulator_env.py unit test PASSED")
