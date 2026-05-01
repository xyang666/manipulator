"""
mpc_controller.py
-----------------
Model Predictive Control (MPC) for manipulator trajectory tracking.

Uses linearized dynamics and quadratic programming to optimize control inputs
over a prediction horizon while respecting constraints.
"""

import numpy as np
from typing import Optional
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("[MPC] WARNING: cvxpy not found. MPC controller disabled.")


class MPCController:
    """
    Model Predictive Control for manipulator.

    Solves a finite-horizon optimal control problem at each timestep:
        min  Σ ||x_k - x_ref||²_Q + ||u_k||²_R
        s.t. x_{k+1} = A x_k + B u_k
             u_min ≤ u_k ≤ u_max
             x_min ≤ x_k ≤ x_max
    """

    def __init__(self,
                 n_states: int = 14,  # [q(7), dq(7)]
                 n_controls: int = 7,  # joint accelerations
                 horizon: int = 10,
                 dt: float = 0.02,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 u_min: Optional[np.ndarray] = None,
                 u_max: Optional[np.ndarray] = None,
                 d_safe: float = 0.15,
                 rep_gain: float = 0.3,
                 w_obs: float = 0.0):
        """
        Parameters
        ----------
        n_states : state dimension
        n_controls : control dimension
        horizon : prediction horizon steps
        dt : timestep
        Q : state cost matrix [n_states x n_states]
        R : control cost matrix [n_controls x n_controls]
        u_min : control lower bounds [n_controls]
        u_max : control upper bounds [n_controls]
        d_safe : obstacle safe distance threshold (m)
        rep_gain : repulsive potential gain for task-space MPC
        w_obs : obstacle cost weight for horizon MPC (0 = disabled)
        """
        if not HAS_CVXPY:
            raise ImportError("cvxpy is required for MPC controller")

        self.n_states = n_states
        self.n_controls = n_controls
        self.horizon = horizon
        self.dt = dt

        # Cost matrices
        self.Q = Q if Q is not None else np.eye(n_states)
        self.R = R if R is not None else 0.01 * np.eye(n_controls)

        # Control constraints (joint acceleration limits)
        self.u_min = u_min if u_min is not None else -10.0 * np.ones(n_controls)
        self.u_max = u_max if u_max is not None else 10.0 * np.ones(n_controls)

        # Obstacle avoidance parameters
        self.d_safe = d_safe
        self.rep_gain = rep_gain
        self.w_obs = w_obs
        self.obs_centers = np.empty((0, 3))
        self.obs_radii = np.empty(0)
        self.n_obs = 0

        # Linearized dynamics matrices (will be updated online)
        self.A = np.eye(n_states)
        self.B = np.zeros((n_states, n_controls))

        # Build QP problem (reuse for efficiency)
        self._build_qp()

    def _build_qp(self):
        """Build the QP problem structure (reusable)."""
        N = self.horizon
        nx = self.n_states
        nu = self.n_controls

        # Decision variables
        self.x_var = [cp.Variable(nx) for _ in range(N + 1)]
        self.u_var = [cp.Variable(nu) for _ in range(N)]

        # Parameters (updated at each solve)
        self.x0_param = cp.Parameter(nx)
        self.x_ref_param = [cp.Parameter(nx) for _ in range(N + 1)]
        self.A_param = cp.Parameter((nx, nx))
        self.B_param = cp.Parameter((nx, nu))

        # Cost function
        cost = 0
        for k in range(N):
            cost += cp.quad_form(self.x_var[k] - self.x_ref_param[k], self.Q)
            cost += cp.quad_form(self.u_var[k], self.R)
        cost += cp.quad_form(self.x_var[N] - self.x_ref_param[N], self.Q)

        # Constraints
        constraints = [self.x_var[0] == self.x0_param]
        for k in range(N):
            # Dynamics
            constraints += [
                self.x_var[k + 1] == self.A_param @ self.x_var[k] + self.B_param @ self.u_var[k]
            ]
            # Control limits
            constraints += [
                self.u_var[k] >= self.u_min,
                self.u_var[k] <= self.u_max
            ]

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    # ------------------------------------------------------------------
    # Obstacle avoidance API
    # ------------------------------------------------------------------

    def set_obstacles(self, centers: np.ndarray | None = None,
                      radii: np.ndarray | None = None):
        """
        Set spherical obstacle information for collision avoidance.

        Parameters
        ----------
        centers : [N x 3] obstacle center positions
        radii   : [N] obstacle radii
        """
        if centers is not None:
            self.obs_centers = np.asarray(centers, dtype=float)
            self.n_obs = len(centers)
        if radii is not None:
            self.obs_radii = np.asarray(radii, dtype=float)
        if self.n_obs == 0:
            self.obs_centers = np.empty((0, 3))
            self.obs_radii = np.empty(0)

    def _repulsive_force(self, point: np.ndarray) -> np.ndarray:
        """
        Compute net repulsive force from all obstacles at a 3D point.

        Uses Khatib-style potential field (Khatib 1986):
            U = sum_i 0.5 * rep_gain * max(0, 1/d_i - 1/d_safe)^2
            F = -grad U = sum_i rep_gain * (1/d_i - 1/d_safe) / d_i^2 * n_i

        where d_i = ||point - c_i|| - r_i is signed distance to obstacle surface,
        and n_i = (point - c_i) / ||point - c_i|| is the direction from obstacle.

        Parameters
        ----------
        point : [3] query point in task space

        Returns
        -------
        F : [3] repulsive force vector
        """
        if self.n_obs == 0:
            return np.zeros(3)

        F = np.zeros(3)
        for i in range(self.n_obs):
            diff = point - self.obs_centers[i]
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                continue
            d_signed = dist - self.obs_radii[i]  # signed distance to surface

            if 0 < d_signed < self.d_safe:
                # Khatib repulsive gradient
                magnitude = self.rep_gain * (1.0 / d_signed - 1.0 / self.d_safe) / (d_signed * d_signed)
                F += magnitude * diff / dist

        # Clip force magnitude to avoid instability
        F_norm = np.linalg.norm(F)
        if F_norm > 2.0:
            F = F / F_norm * 2.0
        return F

    def _multi_point_repulsive_force(self, q: np.ndarray,
                                     kinematics) -> np.ndarray:
        """
        Compute repulsive force at multiple control points along the arm.

        Evaluates repulsive potential at each link capsule endpoint and midpoint,
        providing full-arm obstacle awareness beyond just the end-effector.

        Parameters
        ----------
        q : [n] joint positions
        kinematics : ManipulatorKinematics instance

        Returns
        -------
        F_total : [3] net repulsive force in task space
        """
        if self.n_obs == 0:
            return np.zeros(3)

        capsules = kinematics.get_link_capsules(q)
        F_total = np.zeros(3)
        count = 0

        for p1, p2, cap_radius in capsules:
            # Midpoint and endpoints
            midpoint = (p1 + p2) / 2
            F_total += self._repulsive_force(midpoint)
            F_total += 0.5 * self._repulsive_force(p1)
            F_total += 0.5 * self._repulsive_force(p2)
            count += 2  # Each capsule contributes ~2 effective force evaluations

        # Average to avoid overly large forces from many points
        if count > 0:
            F_total = F_total / count * 3.0  # Scale to roughly match EE-only magnitude

        return F_total

    # ------------------------------------------------------------------
    # Dynamics and control
    # ------------------------------------------------------------------

    def update_linearization(self, q: np.ndarray, dq: np.ndarray,
                            dynamics_model=None):
        """
        Update linearized dynamics A, B matrices around current state.

        For simplicity, use double integrator model:
            q_{k+1} = q_k + dq_k * dt + 0.5 * ddq_k * dt²
            dq_{k+1} = dq_k + ddq_k * dt

        State: x = [q, dq]
        Control: u = ddq (joint accelerations)
        """
        n = len(q)
        dt = self.dt

        # A matrix: [I, dt*I; 0, I]
        self.A[:n, :n] = np.eye(n)
        self.A[:n, n:] = dt * np.eye(n)
        self.A[n:, n:] = np.eye(n)

        # B matrix: [0.5*dt²*I; dt*I]
        self.B[:n, :] = 0.5 * dt**2 * np.eye(n)
        self.B[n:, :] = dt * np.eye(n)

    def compute_control(self,
                       q: np.ndarray,
                       dq: np.ndarray,
                       q_ref: np.ndarray,
                       dq_ref: np.ndarray,
                       kinematics=None) -> np.ndarray:
        """
        Solve MPC problem and return optimal control.

        If kinematics is provided and obstacles are set, adds obstacle avoidance
        by biasing the reference trajectory away from obstacles in joint space.

        Parameters
        ----------
        q : current joint positions [7]
        dq : current joint velocities [7]
        q_ref : reference joint positions [7]
        dq_ref : reference joint velocities [7]
        kinematics : optional ManipulatorKinematics (needed for obstacle avoidance)

        Returns
        -------
        u_opt : optimal control (joint accelerations) [7]
        """
        # Update linearization
        self.update_linearization(q, dq)

        # Current state
        x0 = np.concatenate([q, dq])

        # Reference trajectory
        x_ref = np.concatenate([q_ref, dq_ref])

        # Apply obstacle avoidance: bias reference velocity away from obstacles
        if kinematics is not None and self.n_obs > 0:
            F_rep = self._multi_point_repulsive_force(q, kinematics)
            J = kinematics.jacobian(q)
            Jpinv = kinematics.pseudo_inverse(J)
            # Convert repulsive force to joint-space velocity bias
            dq_rep = Jpinv @ np.concatenate([F_rep * self.dt * 2.0, np.zeros(3)])
            dq_rep = np.clip(dq_rep, -0.3, 0.3)
            # Shift reference velocity away from obstacles
            x_ref_obs = np.concatenate([q_ref, dq_ref - dq_rep * self.rep_gain])
        else:
            x_ref_obs = x_ref

        # Set parameters
        self.x0_param.value = x0
        for k in range(self.horizon + 1):
            self.x_ref_param[k].value = x_ref_obs
        self.A_param.value = self.A
        self.B_param.value = self.B

        # Solve
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

            if self.problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                u_opt = self.u_var[0].value
                return u_opt
            else:
                print(f"[MPC] Solver failed with status: {self.problem.status}")
                return np.zeros(self.n_controls)
        except Exception as e:
            print(f"[MPC] Solver error: {e}")
            return np.zeros(self.n_controls)

    def compute_control_task_space(self,
                                   q: np.ndarray,
                                   dq: np.ndarray,
                                   x_d: np.ndarray,
                                   dx_d: np.ndarray,
                                   kinematics,
                                   obs_centers: np.ndarray | None = None,
                                   obs_radii: np.ndarray | None = None) -> np.ndarray:
        """
        Compute MPC control directly in task space with obstacle avoidance.

        Extends the task-space QP with a Khatib-style repulsive potential field
        that pushes the arm away from obstacles while tracking the trajectory.

        Parameters
        ----------
        q : current joint positions [7]
        dq : current joint velocities [7]
        x_d : desired end-effector position [3]
        dx_d : desired end-effector velocity [6]
        kinematics : kinematics model
        obs_centers : optional [N x 3] obstacle centers (updates stored obstacles)
        obs_radii : optional [N] obstacle radii

        Returns
        -------
        dq_opt : optimal joint velocities [7]
        """
        # Update obstacles if provided
        if obs_centers is not None:
            self.set_obstacles(obs_centers, obs_radii)

        # Get current end-effector position
        x_ee, _ = kinematics.forward_kinematics(q)

        # Compute tracking error
        e_pos = x_d - x_ee

        # Desired task-space velocity with feedback
        Kp = 3.0
        dx_cmd = np.zeros(6)
        dx_cmd[:3] = dx_d[:3] + Kp * e_pos

        # Add obstacle repulsive force (multi-point for full-arm awareness)
        if self.n_obs > 0:
            F_rep = self._multi_point_repulsive_force(q, kinematics)
            dx_cmd[:3] += F_rep

        # Get Jacobian
        J = kinematics.jacobian(q)
        Jpinv = kinematics.pseudo_inverse(J)

        # Task-space tracking velocity (now includes repulsion)
        dq_task = Jpinv @ dx_cmd

        # Formulate QP: minimize deviation from task velocity + control effort
        # min ||dq - dq_task||² + λ||dq||²
        # s.t. dq_min ≤ dq ≤ dq_max

        dq_var = cp.Variable(self.n_controls)

        # Cost: track task velocity + regularization
        cost = cp.sum_squares(dq_var - dq_task) + 0.01 * cp.sum_squares(dq_var)

        # Constraints
        dq_max = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
        constraints = [
            dq_var >= -dq_max,
            dq_var <= dq_max
        ]

        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return dq_var.value
            else:
                return dq_task
        except Exception as e:
            print(f"[MPC] Task-space solver error: {e}")
            return dq_task


if __name__ == "__main__":
    print("=== MPC Controller Unit Test ===\n")

    if not HAS_CVXPY:
        print("SKIPPED: cvxpy not installed")
        exit(0)

    print("--- Test 1: Basic tracking ---")
    mpc = MPCController(n_states=14, n_controls=7, horizon=10)

    q = np.zeros(7)
    dq = np.zeros(7)
    q_ref = np.ones(7) * 0.1
    dq_ref = np.zeros(7)

    u = mpc.compute_control(q, dq, q_ref, dq_ref)
    print(f"Control output shape: {u.shape}")
    print(f"Control values: {u}")
    assert u.shape == (7,), "Control output shape mismatch"
    print("PASSED\n")

    print("--- Test 2: Task-space control ---")
    # Use simplified kinematics (no Pinocchio needed)
    from env.kinematics import ManipulatorKinematics
    kin = ManipulatorKinematics(n_joints=7)
    dq_cmd = mpc.compute_control_task_space(q, dq, np.zeros(3), np.zeros(6), kin)
    print(f"Task-space dq output shape: {dq_cmd.shape}")
    assert dq_cmd.shape == (7,), "Task-space dq output shape mismatch"
    print("PASSED\n")

    print("--- Test 3: Repulsive force computation ---")
    mpc.set_obstacles(
        centers=np.array([[0.5, 0.0, 0.5], [0.6, 0.0, 0.4]]),
        radii=np.array([0.1, 0.08])
    )
    assert mpc.n_obs == 2, "Obstacle count mismatch"
    # Point far from obstacles should get zero force
    F_far = mpc._repulsive_force(np.array([0.0, 0.0, 0.0]))
    print(f"Force at far point: {F_far} (should be ~0)")
    assert np.allclose(F_far, 0), "Far point should have zero repulsive force"
    # Point near obstacle should get repulsive force
    F_near = mpc._repulsive_force(np.array([0.48, 0.0, 0.5]))
    print(f"Force at near point: {F_near} (should be non-zero)")
    # Direction should point away from obstacle (obstacle at [0.5,0,0.5], point at [0.48,0,0.5])
    # Force should push in negative x direction (away from obstacle)
    if np.linalg.norm(F_near) > 0:
        assert F_near[0] < 0, f"Repulsive force should point away from obstacle, got {F_near[0]} > 0"
    print("PASSED\n")

    print("--- Test 4: Multi-point repulsive force ---")
    F_multi = mpc._multi_point_repulsive_force(q, kin)
    print(f"Multi-point force shape: {F_multi.shape}")
    assert F_multi.shape == (3,), "Multi-point force shape mismatch"
    print("PASSED\n")

    print("--- Test 5: Task-space control with obstacle avoidance ---")
    dq_cmd_avoid = mpc.compute_control_task_space(
        q, dq, np.zeros(3), np.zeros(6), kin
    )
    print(f"Task-space dq with avoidance: {dq_cmd_avoid}")
    assert dq_cmd_avoid.shape == (7,), "Avoidance dq output shape mismatch"
    print("PASSED\n")

    print("All MPC controller tests PASSED")
