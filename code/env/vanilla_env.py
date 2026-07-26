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

Author: xie yang
Date:   2025-06

"""

import numpy as np

from env.manipulator_env import ManipulatorEnv, dense_safety_cost


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
        prev_dq = self.dq.copy()
        q_new = self.q + dq_cmd * self.dt
        dq_new = dq_cmd

        if self.mj_data is not None:
            self._mujoco_step(dq_cmd)
        else:
            self.q = q_new
            self.dq = dq_new

        # ---- Phase 3: Post-integration bookkeeping ----
        x_ee, _ = self.kin.forward_kinematics(self.q)
        self._cached_x_ee = x_ee
        tracking_error = self._update_reference(x_ee)

        self.step_count += 1

        # Compute reward using the same observation, collision, and scaling
        # contract as ManipulatorEnv.
        d_obs = self._mujoco_min_distance()
        d_obs = float(np.clip(d_obs, -0.5, 0.5))
        w = self._manipulability()
        self._cached_w = w

        if len(self.ee_trajectory) >= self.max_trajectory_len:
            self.ee_trajectory.pop(0)
        self.ee_trajectory.append(x_ee.copy())

        capsule_dists = self._mujoco_per_capsule_distances()
        self._cached_capsule_dists = capsule_dists

        reward, reward_info = self.reward_fn.compute(
            q=self.q, dq=self.dq, x_ee=x_ee,
            x_d=self.x_d, dx_d=self.dx_d,
            d_obs=d_obs, w=w, prev_dq=prev_dq,
            capsule_dists=capsule_dists,
        )
        if self.reward_min is not None:
            reward = max(reward, self.reward_min)

        # Collision detection
        if self.collision_detector is not None:
            _, n_obs = self.collision_detector.detect_obstacle_collisions()
            _, n_self = self.collision_detector.detect_self_collisions()
            collision = (n_obs + n_self) > 0
        else:
            collision = False

        self._ever_collided = self._ever_collided or collision

        success, done, termination_reason = self._termination_status(x_ee, collision)

        if success:
            reward += self.success_bonus

        reward = reward / self.reward_scale

        tracking_error = float(np.linalg.norm(x_ee - self.x_d))
        self_dists = self.kin.compute_self_distances(self.q)
        d_self = float(np.min(self_dists)) if len(self_dists) else float("inf")
        constraint_distance = min(float(d_obs), d_self)
        cost = dense_safety_cost(constraint_distance, self.d_safe)
        info = {
            "d_obs": d_obs, "d_self": d_self,
            "constraint_distance": constraint_distance,
            "w": w,
            "success": success,
            "collision": collision,
            "cost": cost,
            "path_param": self.path_param,
            "tracking_error": tracking_error,
            "termination_reason": termination_reason,
            "success_hold_count": self._success_hold_count,
            "cbf_active": False,
            "cbf_mod": 0.0,
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
        n_obstacles=1, episode_len=100, trajectory_steps=70,
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
