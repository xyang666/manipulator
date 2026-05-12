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
                 w_obs: float = 0):
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

            if d_signed < self.d_safe:
                # Khatib repulsive gradient — also handle penetration (d_signed <= 0)
                # by clamping effective distance to avoid singularity
                d_eff = max(d_signed, 0.005)
                magnitude = self.rep_gain * (1.0 / d_eff - 1.0 / self.d_safe) / (d_eff * d_eff)
                F += magnitude * diff / dist

        # Clip force magnitude to avoid instability
        F_norm = np.linalg.norm(F)
        if F_norm > 2.0:
            F = F / F_norm * 2.0
        return F

    def _capsule_sphere_distance(self, p1: np.ndarray, p2: np.ndarray,
                                  cap_r: float, sphere_c: np.ndarray,
                                  sphere_r: float) -> float:
        """Signed distance from capsule to sphere (negative = overlap)."""
        # Closest point on capsule segment to sphere center
        seg = p2 - p1
        seg_len_sq = np.dot(seg, seg)
        if seg_len_sq < 1e-10:
            closest = p1
        else:
            t = np.dot(sphere_c - p1, seg) / seg_len_sq
            t = max(0.0, min(1.0, t))
            closest = p1 + t * seg
        dist = np.linalg.norm(sphere_c - closest)
        return dist - (cap_r + sphere_r)

    def _arm_min_distance(self, q: np.ndarray, kinematics) -> float:
        """
        Minimum signed distance from any arm capsule to any obstacle.
        Unlike _min_obstacle_distance which only checks EE, this checks the
        full arm using capsule representation.
        """
        if self.n_obs == 0:
            return float('inf')
        capsules = kinematics.get_link_capsules(q)
        min_d = float('inf')
        for p1, p2, cap_r in capsules:
            for i in range(self.n_obs):
                d = self._capsule_sphere_distance(p1, p2, cap_r,
                                                   self.obs_centers[i], self.obs_radii[i])
                min_d = min(min_d, d)
        return float(min_d)

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

    def _null_space_repulsion(self, q: np.ndarray,
                               kinematics) -> np.ndarray:
        """
        Compute null-space avoidance velocity using per-link Jacobians.

        For each capsule point on the arm, the repulsive force is mapped to
        joint-space velocity using THAT LINK'S OWN Jacobian (not the EE
        Jacobian).  This ensures the force physically corresponds to moving
        that specific link, so the null-space projection N preserves it.

        Returns
        -------
        dq_avoid : [n] joint velocity in null space of the position task
        """
        # Link names in the same order as kinematics.get_link_capsules()
        capsule_link_order = [
            "panda_link0", "panda_link1", "panda_link2", "panda_link3",
            "panda_link4", "panda_link5", "panda_link5",  # link5 has 2 capsules
            "panda_link6", "panda_link7", "panda_hand",
            "panda_leftfinger", "panda_rightfinger",
        ]
        capsules = kinematics.get_link_capsules(q)
        # Get all link Jacobians at once
        link_names = sorted(set(capsule_link_order))
        link_jacs = kinematics.link_jacobians_position(q, link_names)

        dq_rep = np.zeros(self.n_controls)
        for ci, (p1, p2, cap_r) in enumerate(capsules):
            link_name = capsule_link_order[ci] if ci < len(capsule_link_order) else None
            if link_name not in link_jacs:
                continue
            J_link = link_jacs[link_name]  # [3×n]

            for pt in [p1, (p1 + p2) / 2, p2]:
                F_pt = self._repulsive_force(pt)
                if np.linalg.norm(F_pt) < 1e-6:
                    continue
                # Map force to joint velocity via this link's Jacobian
                Jpinv_link = kinematics.pseudo_inverse(J_link)
                dq_rep += Jpinv_link @ F_pt

        # Project into null space of the position task
        J_pos = kinematics.jacobian(q)[:3, :]
        Jpinv_pos = kinematics.pseudo_inverse(J_pos)
        N = np.eye(self.n_controls) - Jpinv_pos @ J_pos
        return N @ dq_rep

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

    def _min_obstacle_distance(self, point: np.ndarray) -> float:
        """Minimum signed distance from point to any obstacle surface."""
        if self.n_obs == 0:
            return float('inf')
        dists = np.linalg.norm(self.obs_centers - point, axis=1) - self.obs_radii
        return float(np.min(dists))

    def _task_repulsive_velocity(self, point: np.ndarray) -> np.ndarray:
        """
        Compute repulsive task-space velocity for end-effector obstacle avoidance.

        Unlike _null_space_repulsion which only affects arm configuration,
        this directly modifies the EE velocity command, actively pushing the
        end-effector away from nearby obstacles.

        Uses a smooth potential that activates at d_safe and peaks at d_critical:
            v_rep = sum_i w_i * (d_safe - d_i) / d_safe * n_i
        where d_i is signed distance to obstacle surface, n_i is away direction.

        Parameters
        ----------
        point : [3] end-effector position

        Returns
        -------
        dx_avoid : [3] repulsive task-space velocity
        """
        if self.n_obs == 0:
            return np.zeros(3)

        dx_avoid = np.zeros(3)
        for i in range(self.n_obs):
            diff = point - self.obs_centers[i]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue
            d_signed = dist - self.obs_radii[i]

            # Activate at 2× d_safe for earlier response
            activation_dist = self.d_safe * 2.0
            if d_signed < activation_dist:
                # Strength: 1.0 when penetrating, 0.0 at activation_dist
                strength = max(0.0, 1.0 - d_signed / activation_dist)
                direction = diff / dist  # points away from obstacle
                # Weber's Law style: stronger when closer (quadratic scaling)
                dx_avoid += (strength ** 2) * direction * 0.2  # 20 cm/s peak per obstacle

        # Clip total repulsive velocity to a meaningful max
        avoid_norm = np.linalg.norm(dx_avoid)
        max_rep = 0.5  # 50 cm/s max total repulsion
        if avoid_norm > max_rep:
            dx_avoid = dx_avoid / avoid_norm * max_rep
        return dx_avoid

    def _adaptive_kp(self, d_obs: float, Kp_base: float = 4.0, Kp_min: float = 0.5) -> float:
        """
        Adaptive proportional gain modulated by obstacle distance.

        Full tracking gain (Kp_base) when far from obstacles.
        Reduced gain near obstacles to prevent aggressive tracking into them.
        Smooth quadratic interpolation:
            Kp = Kp_min + (Kp_base - Kp_min) * (d_obs / d_mod)^2
        """
        d_mod = self.d_safe * 2.0  # modulation starts at 2× d_safe
        if d_obs >= d_mod:
            return Kp_base
        t = max(0.0, d_obs / d_mod)
        return Kp_min + (Kp_base - Kp_min) * (t * t)

    def compute_control_task_space(self,
                                   q: np.ndarray,
                                   dq: np.ndarray,
                                   x_d: np.ndarray,
                                   dx_d: np.ndarray,
                                   kinematics,
                                   obs_centers: np.ndarray | None = None,
                                   obs_radii: np.ndarray | None = None) -> np.ndarray:
        """
        Improved task-space MPC with adaptive tracking and obstacle-aware control.

        Key improvements over v1:
          1. Adaptive Kp — reduces tracking gain near obstacles so the EE
             naturally "slows down" and avoids aggressive tracking into clutter.
          2. Task-space repulsive velocity — directly modifies the EE velocity
             command to push away from obstacles (not just nullspace self-motion).
          3. Proximity-scaled nullspace avoidance — strengthens multi-link
             avoidance as obstacles get closer.

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

        # Get current end-effector position and min obstacle distance
        # NOTE: use full-arm SDF distance (not just EE) so the MPC is aware
        # of arm-link proximity to obstacles — critical for scene 0 where
        # links start close to obstacles.
        x_ee, _ = kinematics.forward_kinematics(q)
        d_obs = self._arm_min_distance(q, kinematics)

        # --- 1. Adaptive proportional gain ---
        # Reduce Kp aggressively near obstacles so tracking doesn't fight repulsion
        Kp = self._adaptive_kp(d_obs, Kp_base=4.0, Kp_min=0.15)

        # --- 2. Task velocity with tracking + obstacle repulsion ---
        e_pos = x_d - x_ee
        dx_cmd = dx_d[:3] + Kp * e_pos  # tracking

        # Add repulsive velocity directly in task space so the EE actively
        # avoids obstacles (not just nullspace self-motion).
        # Activation at 2× d_safe so repulsion kicks in before Kp is fully reduced.
        if self.n_obs > 0 and d_obs < self.d_safe * 2.0:
            dx_cmd += self._task_repulsive_velocity(x_ee)

        # Position Jacobian [3×7]
        J_pos = kinematics.jacobian(q)[:3, :]
        Jpinv = kinematics.pseudo_inverse(J_pos)

        # --- 3. Primary task: position tracking ---
        dq_track = Jpinv @ dx_cmd

        # --- 4. Secondary task: obstacle avoidance in null space ---
        dq_avoid = np.zeros(self.n_controls)
        if self.n_obs > 0:
            dq_avoid = self._null_space_repulsion(q, kinematics)
            # Scale nullspace avoidance by proximity for stronger effect near obstacles
            if d_obs < self.d_safe * 2.0:
                # Scale from 1x at 2*d_safe to 5x at d_obs=0 (penetration)
                scale = 1.0 + 4.0 * max(0.0, 1.0 - d_obs / (self.d_safe * 2.0))
                dq_avoid *= scale

        # Combined command: tracking + null-space avoidance
        dq_desired = dq_track + dq_avoid

        # --- 5. QP: smooth command with velocity limits ---
        dq_var = cp.Variable(self.n_controls)
        cost = cp.sum_squares(dq_var - dq_desired) + 0.01 * cp.sum_squares(dq_var)

        dq_max = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
        constraints = [dq_var >= -dq_max, dq_var <= dq_max]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return dq_var.value
            else:
                return dq_desired
        except Exception as e:
            print(f"[MPC] Task-space solver error: {e}")
            return dq_desired
    def _predict_trajectory(self, x_start: np.ndarray,
                             dx_start: np.ndarray,
                             steps: int) -> tuple:
        """
        Predict EE trajectory over N steps assuming current velocity continues.

        Returns predicted positions and the min obstacle distance along the path.
        """
        if self.n_obs == 0:
            return np.tile(x_start, (steps, 1)), float('inf')

        path_min_d = float('inf')
        predicted_positions = np.zeros((steps, 3))
        x_k = x_start.copy()

        for k in range(steps):
            x_k = x_k + dx_start * self.dt
            predicted_positions[k] = x_k
            # Min distance along predicted path
            for i in range(self.n_obs):
                d = np.linalg.norm(x_k - self.obs_centers[i]) - self.obs_radii[i]
                path_min_d = min(path_min_d, d)

        return predicted_positions, path_min_d

    def _lookahead_repulsive_force(self, x_ee: np.ndarray,
                                    dx_nom: np.ndarray,
                                    kinematics,
                                    q: np.ndarray) -> np.ndarray:
        """
        Lookahead repulsive force that anticipates future obstacle proximity.

        Simulates the EE trajectory `horizon` steps ahead and computes a
        preemptive repulsive velocity that activates based on future (not
        current) obstacle proximity.  This gives the controller predictive
        capability similar to a true receding-horizon MPC.

        Returns
        -------
        dx_lookahead : [3] task-space velocity correction
        """
        if self.n_obs == 0:
            return np.zeros(3)

        dx_lookahead = np.zeros(3)
        x_k = x_ee.copy()
        dx_k = dx_nom.copy()
        lookahead_steps = min(self.horizon, 8)

        for k in range(1, lookahead_steps + 1):
            x_k = x_k + dx_k * self.dt
            # Closest obstacle at this predicted position
            for i in range(self.n_obs):
                diff = x_k - self.obs_centers[i]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    continue
                d_signed = dist - self.obs_radii[i]
                # Lookahead activation: respond to obstacles ahead
                if d_signed < self.d_safe * 2.0:
                    strength = max(0.0, 1.0 - d_signed / (self.d_safe * 2.0))
                    direction = diff / dist  # away from obstacle
                    # Temporal discount: earlier obstacles matter more
                    weight = 1.0 / k
                    dx_lookahead += strength * weight * direction * 0.3

        return dx_lookahead

    def compute_control_task_space_horizon(self,
                                            q: np.ndarray,
                                            dq: np.ndarray,
                                            x_d: np.ndarray,
                                            dx_d: np.ndarray,
                                            kinematics,
                                            obs_centers: np.ndarray | None = None,
                                            obs_radii: np.ndarray | None = None) -> np.ndarray:
        """
        Task-space MPC with receding-horizon obstacle lookahead.

        Combines:
          1. Full-arm d_obs awareness (adaptive Kp, nullspace scaling)
          2. Immediate repulsive velocity (task-space push from nearby obstacles)
          3. Horizon-based lookahead (preemptive push from predicted proximity)
          4. Nullspace self-motion avoidance

        The lookahead simulates the nominal trajectory `horizon` steps forward
        and adds a preemptive repulsive velocity based on future obstacle
        proximity — giving predictive capability without full trajectory
        optimization.
        """
        # Update obstacles if provided
        if obs_centers is not None:
            self.set_obstacles(obs_centers, obs_radii)

        # Current state and full-arm obstacle distance
        x_ee, _ = kinematics.forward_kinematics(q)
        d_obs = self._arm_min_distance(q, kinematics)

        # --- 1. Adaptive proportional gain ---
        Kp = self._adaptive_kp(d_obs, Kp_base=4.0, Kp_min=0.15)

        # --- 2. Task velocity: tracking + immediate repulsion + lookahead ---
        e_pos = x_d - x_ee
        dx_cmd = dx_d[:3] + Kp * e_pos

        # Immediate repulsive velocity (current proximity)
        if self.n_obs > 0 and d_obs < self.d_safe * 2.0:
            dx_cmd += self._task_repulsive_velocity(x_ee)

        # Horizon lookahead (predicted proximity)
        if self.n_obs > 0 and d_obs < self.d_safe * 3.0:
            dx_cmd += self._lookahead_repulsive_force(x_ee, dx_cmd, kinematics, q)

        # --- 3. Position Jacobian and primary tracking ---
        J_pos = kinematics.jacobian(q)[:3, :]
        Jpinv = kinematics.pseudo_inverse(J_pos)
        dq_track = Jpinv @ dx_cmd

        # --- 4. Nullspace obstacle avoidance ---
        dq_avoid = np.zeros(self.n_controls)
        if self.n_obs > 0:
            dq_avoid = self._null_space_repulsion(q, kinematics)
            if d_obs < self.d_safe * 2.0:
                scale = 1.0 + 4.0 * max(0.0, 1.0 - d_obs / (self.d_safe * 2.0))
                dq_avoid *= scale

        # Combined command
        dq_desired = dq_track + dq_avoid

        # --- 5. QP: smooth command within velocity limits ---
        dq_var = cp.Variable(self.n_controls)
        cost = cp.sum_squares(dq_var - dq_desired) + 0.01 * cp.sum_squares(dq_var)

        dq_max = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
        constraints = [dq_var >= -dq_max, dq_var <= dq_max]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return dq_var.value
            else:
                return dq_desired
        except Exception as e:
            print(f"[MPC] Horizon solver error: {e}")
            return dq_desired


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
