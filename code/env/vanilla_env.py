"""
vanilla_env.py
--------------
VanillaEnv: a drop-in replacement for ManipulatorEnv that applies
actions directly as joint velocities (dq), bypassing the task-space
control law used by the physics-informed agent.

Action: 7D joint velocities [dq_1, ..., dq_7] in rad/s, clipped to
        actuator limits (DQ_MAX ≈ 2.175 rad/s each).

This is the environment used by the VanillaSACAgent baseline.
Reference: Haarnoja et al., "Soft Actor-Critic", 2018
"""

import numpy as np

from env.manipulator_env import ManipulatorEnv


class VanillaEnv(ManipulatorEnv):
    """
    Subclass of ManipulatorEnv that overrides step() to apply actions
    directly as joint velocity commands.

    No task-space decomposition, no sigma gating, no nullspace projection.
    """

    def step(self, action: np.ndarray):
        """
        Apply action directly as joint velocity dq.

        Parameters
        ----------
        action : 7D joint velocities [dq_1, ..., dq_7] in rad/s

        Returns
        -------
        obs, reward, done, info
        """
        # ---- Phase 1: Direct dq control (bypasses task-space control law) ----
        dq_cmd = np.asarray(action, dtype=float).copy()
        dq_cmd = np.clip(dq_cmd, -self._dq_max, self._dq_max)

        # Dummy metadata for parallel env worker compatibility
        self._last_J = np.zeros((3, self.n), dtype=np.float32)
        self._last_sigma = np.float32(0.0)
        self._last_dx_nom = np.zeros(3, dtype=np.float32)

        # ---- Phase 2: Integration (identical to parent) ----
        q_new = self.q + dq_cmd * self.dt
        dq_new = dq_cmd

        if self.mj_data is not None:
            self._mujoco_step(dq_cmd)
        else:
            self.q = q_new
            self.dq = dq_new

        # ---- Phase 3: Post-integration bookkeeping ----
        # Tracking-error-driven path progression with trapezoidal speed profile
        x_ee, _ = self.kin.forward_kinematics(self.q)
        tracking_error = np.linalg.norm(x_ee - self.x_d)

        # Nominal path parameter (trapezoidal: ease-in -> constant -> ease-out)
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
        err_deadzone = max(0.0, tracking_error - 0.02)
        raw_advance = float(np.clip(1.0 - err_deadzone / self.path_deadzone, 0.0, 1.0))
        advance_rate = 0.5 * raw_advance + 0.5 * getattr(self, '_last_advance', raw_advance)
        self._last_advance = advance_rate
        self.path_param = min(1.0, self.path_param + (nominal_s - self.path_param) * advance_rate)

        # Update target position
        if self.use_parametric_traj and self._parametric_pos_func is not None:
            t = self.step_count * self.dt
            prev_x_d = self.x_d.copy()
            self.x_d = self._parametric_pos_func(t)
            self.dx_d[:3] = self._parametric_vel_func(t)
        else:
            prev_x_d = self.x_d.copy()
            self.x_d = (1.0 - self.path_param) * self.x_start + self.path_param * self.x_goal
            self.dx_d[:3] = (self.x_d - prev_x_d) / self.dt

        self.step_count += 1

        # Compute reward
        x_ee, _ = self.kin.forward_kinematics(self.q)
        d_obs = self.sdf.min_distance(x_ee, self.q, kinematics=self.kin)
        d_obs = float(np.clip(d_obs, -0.5, 0.5))
        w = self._manipulability()

        if len(self.ee_trajectory) >= self.max_trajectory_len:
            self.ee_trajectory.pop(0)
        self.ee_trajectory.append(x_ee.copy())

        reward, reward_info = self.reward_fn.compute(
            q=self.q, dq=self.dq, x_ee=x_ee,
            x_d=self.x_d, dx_d=self.dx_d,
            d_obs=d_obs, w=w, x_goal=self.x_goal,
        )

        # Collision detection
        if self.mj_model is not None:
            collision = (reward_info.get("n_obstacle_contacts", 0) > 0 or
                         reward_info.get("n_self_contacts", 0) > 0)
        else:
            collision = d_obs < 0.02

        self._ever_collided = self._ever_collided or collision

        # Termination conditions
        if self.use_parametric_traj:
            path_complete = self.step_count >= self.episode_len
        else:
            path_complete = self.path_param >= 0.99
        done = self.step_count >= self.episode_len or path_complete
        if self.collision_term:
            done = done or collision

        # Sparse success bonus
        if path_complete:
            reward += self.success_bonus

        tracking_error = float(np.linalg.norm(x_ee - self.x_d))
        info = {
            "d_obs": d_obs, "w": w,
            "success": path_complete and not self._ever_collided,
            "collision": collision,
            "path_param": self.path_param,
            "tracking_error": tracking_error,
            **reward_info,
        }

        return self._get_obs(), reward, done, info


# ------------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(os.path.dirname(_HERE))
    _venv_data = os.path.join(_HERE, "..", ".venv/lib/python3.12/site-packages/cmeel.prefix"
                              "/share/example-robot-data/robots/panda_description")
    _urdf = os.path.join(_venv_data, "urdf/panda.urdf")
    _xml = os.path.join(_ROOT, "models/panda_scene.xml")

    print("=== vanilla_env.py unit tests ===")

    env = VanillaEnv(
        urdf_path=_urdf, xml_path=_xml, n_joints=7,
        n_obstacles=1, episode_len=100,
    )

    # Initial state
    obs = env.reset()
    print(f"obs shape: {list(obs.shape)}  (expected [28])")

    # Step with zero action (joints should stay still)
    zero_action = np.zeros(env.n)
    obs2, reward, done, info = env.step(zero_action)
    q_diff = np.linalg.norm(env.q - env.q)
    print(f"zero action: q_diff={q_diff:.6f}  (expected ~0)")

    # Step with positive dq (joints should move)
    pos_action = np.ones(env.n) * 0.1  # 0.1 rad/s
    q_before = env.q.copy()
    obs3, reward, done, info = env.step(pos_action)
    q_moved = np.linalg.norm(env.q - q_before)
    print(f"pos action:  q_moved={q_moved:.6f}  (expected ~0.002 = 0.1*0.02)")

    # Check info fields
    expected_keys = ["d_obs", "w", "success", "collision", "path_param", "tracking_error"]
    for k in expected_keys:
        assert k in info, f"Missing key '{k}' in info"
    print(f"info keys: OK  success={info['success']} collision={info['collision']}")

    print("vanilla_env.py unit test PASSED")
