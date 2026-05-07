"""
kinematics.py
-------------
Kinematics utilities for a 7-DOF manipulator:
  - forward_kinematics(q)         -> end-effector SE3 pose
  - jacobian(q)                   -> J ∈ R^{6 x n}
  - pseudo_inverse(J)             -> J† (damped least-squares)
  - null_space_projector(q)       -> N = I - J†J ∈ R^{n x n}
  - null_space_velocity(q, dq0)   -> q̇_null = N(q) @ dq0
  - inverse_kinematics(x_target)  -> q (IK solution)

Usage:
    kin = ManipulatorKinematics(urdf_path)
    x, R = kin.forward_kinematics(q)
    J    = kin.jacobian(q)
    N    = kin.null_space_projector(q)
    q    = kin.inverse_kinematics(x_target)
"""

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("[kinematics] WARNING: pinocchio not found. Using simplified DH model.")


class ManipulatorKinematics:
    """
    Kinematic computations using Pinocchio (or a simplified fallback).
    """

    def __init__(self, urdf_path: str | None = None, n_joints: int = 7,
                 damping: float = 1e-4,
                 q_min: np.ndarray | None = None,
                 q_max: np.ndarray | None = None):
        """
        Parameters
        ----------
        urdf_path : path to URDF file (None → simplified mode)
        n_joints  : number of joints
        damping   : damping factor λ for damped pseudo-inverse
        q_min     : joint lower limits [n] (None = no limit enforcement)
        q_max     : joint upper limits [n] (None = no limit enforcement)
        """
        self.n = n_joints
        self.damping = damping
        # Store joint limits (used by IK for clamping)
        self.q_min = np.asarray(q_min, dtype=float) if q_min is not None else None
        self.q_max = np.asarray(q_max, dtype=float) if q_max is not None else None
        self.model = None
        self.data = None
        self.ee_frame_id = None

        if HAS_PINOCCHIO and urdf_path is not None:
            self._init_pinocchio(urdf_path)

    def _init_pinocchio(self, urdf_path: str):
        full_model = pin.buildModelFromUrdf(urdf_path)
        # Lock extra joints beyond n (e.g. Panda fingers) so model stays n-DOF
        if full_model.nv > self.n:
            joints_to_lock = list(range(self.n + 1, full_model.njoints))
            q_ref = pin.neutral(full_model)
            self.model = pin.buildReducedModel(full_model, joints_to_lock, q_ref)
        else:
            self.model = full_model
        self.data = self.model.createData()
        self.n = self.model.nv
        # Prefer tcp/ee frame; fallback to last frame
        self.ee_frame_id = self.model.nframes - 1
        for i, f in enumerate(self.model.frames):
            if "tcp" in f.name.lower() or "ee" in f.name.lower():
                self.ee_frame_id = i
                break
        print(f"[kinematics] Loaded model, n_joints={self.n}, "
              f"ee_frame='{self.model.frames[self.ee_frame_id].name}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward_kinematics(self, q: np.ndarray):
        """
        Returns end-effector (position, rotation_matrix).
        position : [3]
        R        : [3 x 3]
        """
        q = np.asarray(q, dtype=float)
        if self.model is not None:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            T = self.data.oMf[self.ee_frame_id]
            return T.translation.copy(), T.rotation.copy()
        else:
            return self._fk_simplified(q)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Returns geometric Jacobian J ∈ R^{6 x n}.
        Rows 0:3 = linear velocity, rows 3:6 = angular velocity.
        """
        q = np.asarray(q, dtype=float)
        if self.model is not None:
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            J = pin.getFrameJacobian(
                self.model, self.data, self.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            return J.copy()   # [6 x n]
        else:
            return self._jacobian_simplified(q)

    def link_jacobians_position(self, q: np.ndarray,
                                 link_names: list[str]) -> dict[str, np.ndarray]:
        """
        Returns position Jacobians (3×n) for specified links.

        Used for per-link null-space obstacle avoidance: each capsule point
        needs its own link Jacobian so the repulsive force correctly maps
        to joint motion.

        Parameters
        ----------
        q          : joint positions [n]
        link_names : list of link names (e.g. ["panda_link3", "panda_link4", ...])

        Returns
        -------
        jacobians : dict {link_name: J_pos (3×n)}
        """
        if self.model is None:
            return {}
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        result = {}
        for name in link_names:
            try:
                fid = self.model.getFrameId(name)
                J = pin.getFrameJacobian(
                    self.model, self.data, fid,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                result[name] = J[:3, :].copy()  # position rows only
            except Exception:
                pass  # skip unknown frames
        return result

    def pseudo_inverse(self, J: np.ndarray) -> np.ndarray:
        """
        Damped least-squares pseudo-inverse with adaptive damping.
        Increases damping automatically near singularities (Nakamura & Hanafusa 1986).
          J† = J^T (J J^T + λ²I)^{-1}
        """
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        # Adaptive damping: only activate below sv=0.02 (new trajectory avoids deep singularities)
        min_sv = s[-1]
        sv_thresh = 0.02
        if min_sv < sv_thresh:
            lam2 = (sv_thresh * (1 - (min_sv / sv_thresh) ** 2)) ** 2
        else:
            lam2 = self.damping ** 2
        s_inv = s / (s ** 2 + lam2)
        return Vt.T @ np.diag(s_inv) @ U.T

    def get_link_capsules(self, q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Returns capsule representation of each link for Franka Panda.
        Each capsule is (start_point, end_point, radius).

        Extracts collision geometry directly from URDF collision primitives:
        each link uses cylinder+2 spheres → capsule endpoints are sphere centers.
        Transforms local coordinates to world frame using Pinocchio.

        Returns
        -------
        capsules : list of (p1, p2, r) where p1, p2 ∈ R^3, r is radius
        """
        q = np.asarray(q, dtype=float)
        capsules = []

        if self.model is not None:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Build frame name → (position, rotation) map
            frame_tf = {}
            for i, f in enumerate(self.model.frames):
                frame_tf[f.name] = (
                    self.data.oMf[i].translation.copy(),
                    self.data.oMf[i].rotation.copy()
                )

            # URDF collision geometry: (link_frame, [(local_p1, local_p2, radius), ...])
            # Extracted from panda_collision.urdf sphere positions
            collision_specs = {
                "panda_link0": [
                    (np.array([-0.09, 0, 0.06]), np.array([-0.06, 0, 0.06]), 0.09),
                ],
                "panda_link1": [
                    (np.array([0, 0, -0.333]), np.array([0, 0, -0.05]), 0.09),
                ],
                "panda_link2": [
                    (np.array([0, 0, -0.06]), np.array([0, 0, 0.06]), 0.09),
                ],
                "panda_link3": [
                    (np.array([0, 0, -0.22]), np.array([0, 0, -0.07]), 0.09),
                ],
                "panda_link4": [
                    (np.array([0, 0, -0.06]), np.array([0, 0, 0.06]), 0.09),
                ],
                "panda_link5": [
                    (np.array([0, 0, -0.31]), np.array([0, 0, -0.21]), 0.09),
                    (np.array([0, 0.08, -0.20]), np.array([0, 0.08, -0.06]), 0.055),
                ],
                "panda_link6": [
                    (np.array([0, 0, -0.07]), np.array([0, 0, 0.01]), 0.08),
                ],
                "panda_link7": [
                    (np.array([0, 0, -0.06]), np.array([0, 0, 0.08]), 0.07),
                ],
                "panda_hand": [
                    (np.array([0, -0.075, 0.03]), np.array([0, 0.075, 0.03]), 0.05),
                ],
                "panda_leftfinger": [
                    (np.array([0, 0.015, 0.015]), np.array([0, 0.015, 0.045]), 0.015),
                ],
                "panda_rightfinger": [
                    (np.array([0, -0.015, 0.015]), np.array([0, -0.015, 0.045]), 0.015),
                ],
            }

            # Transform each capsule to world frame
            for link_name, caps_local in collision_specs.items():
                if link_name not in frame_tf:
                    continue
                pos, rot = frame_tf[link_name]
                for p1_local, p2_local, radius in caps_local:
                    p1_world = pos + rot @ p1_local
                    p2_world = pos + rot @ p2_local
                    if np.linalg.norm(p2_world - p1_world) > 1e-3:
                        capsules.append((p1_world, p2_world, radius))

        else:
            # Simplified fallback: approximate with major segments
            segs = [
                (np.array([0., 0., 0.]),       np.array([0., 0., 0.333]),     0.09),
                (np.array([0., 0., 0.333]),    np.array([0., 0., 0.649]),     0.09),
                (np.array([0., 0., 0.649]),    np.array([0.082, 0., 0.649]),  0.09),
                (np.array([0.082, 0., 0.649]), np.array([0.466, 0., 0.732]),  0.09),
                (np.array([0.466, 0., 0.732]), np.array([0.554, 0., 0.732]),  0.08),
                (np.array([0.554, 0., 0.732]), np.array([0.554, 0., 0.625]),  0.07),
                (np.array([0.554, 0., 0.625]), np.array([0.554, 0., 0.566]),  0.05),
            ]
            capsules = [(p1, p2, r) for p1, p2, r in segs]
            # Fingers
            for side in (-1.0, 1.0):
                f_base = np.array([0.554, side * 0.015, 0.551])
                f_tip  = np.array([0.554, side * 0.015, 0.581])
                capsules.append((f_base, f_tip, 0.015))

        return capsules

    def jacobian_position(self, q: np.ndarray) -> np.ndarray:
        """
        Position-only Jacobian J_pos ∈ R^{3 x n} (linear velocity rows only).

        Used for Route A (position-only tracking): the 7-DOF arm has 3D task
        constraints → null-space dimension = 7 - 3 = 4.
        """
        q = np.asarray(q, dtype=float)
        if self.model is not None:
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            J_full = pin.getFrameJacobian(
                self.model, self.data, self.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            return J_full[:3, :].copy()  # [3 x n]
        else:
            return self._jacobian_simplified(q)[:3, :]

    def null_space_projector(self, q: np.ndarray) -> np.ndarray:
        """
        N(q) = I - J†(q) J(q)   ∈ R^{n x n}

        Any vector q̇₀ projected through N satisfies J @ (N @ q̇₀) ≈ 0,
        meaning it produces no end-effector motion (pure self-motion).

        For 6x7 J: null-space dimension = 7 - rank(J) = 1.
        """
        J = self.jacobian(q)
        Jpinv = self.pseudo_inverse(J)
        return np.eye(self.n) - Jpinv @ J

    def null_space_projector_position(self, q: np.ndarray) -> np.ndarray:
        """
        Position-only null-space projector N_pos ∈ R^{n x n}.

        N_pos = I - J_pos† @ J_pos, where J_pos ∈ R^{3 x n}.
        Null-space dimension = n - 3 = 4 (for a 7-DOF arm tracking only position).
        """
        J = self.jacobian_position(q)
        Jpinv = self.pseudo_inverse(J)
        return np.eye(self.n) - Jpinv @ J

    def null_space_basis_position(self, q: np.ndarray) -> np.ndarray:
        """
        Orthonormal basis B ∈ R^{n x (n-3)} for the position nullspace.

        For 7-DOF arm: B ∈ R^{7x4} with J_pos @ B ≈ 0 and B^T B = I.
        Maps low-dimensional coefficients to nullspace velocity:
            dq_null = B @ z,   z ∈ R^{4}
        """
        J = self.jacobian_position(q)  # (3, n)
        _, _, Vt = np.linalg.svd(J, full_matrices=True)  # Vt: (n, n)
        return Vt.T[:, 3:]  # (n, n-3)

    def null_space_velocity(self, q: np.ndarray, dq0: np.ndarray) -> np.ndarray:
        """
        Project dq0 into null space: q̇_null = N(q) @ dq0
        """
        N = self.null_space_projector(q)
        return N @ np.asarray(dq0, dtype=float)

    def task_velocity(self, q: np.ndarray, dx_desired: np.ndarray) -> np.ndarray:
        """
        Task-space velocity tracking:
          q̇_task = J†(q) @ dx_desired
        where dx_desired ∈ R^6 (linear + angular)
        """
        J = self.jacobian(q)
        Jpinv = self.pseudo_inverse(J)
        return Jpinv @ np.asarray(dx_desired, dtype=float)

    def combine_velocities(self, q: np.ndarray,
                           dx_desired: np.ndarray,
                           dq0: np.ndarray) -> np.ndarray:
        """
        Full joint velocity command (paper eq):
          q̇ = J†ẋ_d + N(q) @ dq0

        Parameters
        ----------
        dx_desired : desired EE velocity [6] (task space)
        dq0        : null-space self-motion command [n] (from RL policy)
        """
        J = self.jacobian(q)
        Jpinv = self.pseudo_inverse(J)
        N = np.eye(self.n) - Jpinv @ J

        # Compute combined velocity
        dq_task = Jpinv @ dx_desired
        dq_null = N @ dq0
        dq_combined = dq_task + dq_null

        return dq_combined

    def combine_velocities_with_relaxation(self, q: np.ndarray,
                                           dx_desired: np.ndarray,
                                           delta_x: np.ndarray,
                                           dq0: np.ndarray) -> np.ndarray:
        """
        Combine task-space relaxation with null-space self-motion (paper Eq. 8).

        Control law: q̇ = J†(ẋ_d + Δẋ) + N(q) @ dq0

        This allows the policy to trade off tracking accuracy for obstacle avoidance
        by relaxing the task-space velocity command.

        Parameters
        ----------
        q          : joint positions [n]
        dx_desired : nominal task velocity [6] (linear + angular)
        delta_x    : task relaxation [6] (from policy, allows deviation from nominal)
        dq0        : null-space velocity [n] (from policy, self-motion)

        Returns
        -------
        dq : combined joint velocity [n]
        """
        J = self.jacobian(q)
        Jpinv = self.pseudo_inverse(J)
        N = self.null_space_projector(q)

        # Relaxed task velocity (allows policy to deviate from nominal trajectory)
        dx_cmd = np.asarray(dx_desired, dtype=float) + np.asarray(delta_x, dtype=float)

        # Combined control law: task tracking + null-space self-motion
        dq = Jpinv @ dx_cmd + N @ np.asarray(dq0, dtype=float)
        return dq

    def combine_velocities_with_relaxation_position(self, q: np.ndarray,
                                                     dx_desired: np.ndarray,
                                                     delta_x: np.ndarray,
                                                     dq0: np.ndarray) -> np.ndarray:
        """
        Position-only control law (Route A):
          q̇ = J_pos⁺(ẋ_d + Δẋ) + N_pos(q) @ dq0

        All task vectors are ℝ³ (position only, no orientation).
        Uses position-only Jacobian → 4D null space for 7-DOF arm.

        Parameters
        ----------
        dx_desired : nominal EE velocity [3] (position only)
        delta_x    : RL task relaxation [3] (position only)
        dq0        : null-space velocity [n]
        """
        J = self.jacobian_position(q)
        Jpinv = self.pseudo_inverse(J)
        N = self.null_space_projector_position(q)
        dx_cmd = np.asarray(dx_desired, dtype=float) + np.asarray(delta_x, dtype=float)
        dq = Jpinv @ dx_cmd + N @ np.asarray(dq0, dtype=float)
        return dq

    def inverse_kinematics(self, x_target: np.ndarray, q_init: np.ndarray | None = None,
                          max_iter: int = 100, tol: float = 1e-4, damping: float = 1e-4) -> np.ndarray | None:
        """
        Numerical IK solver using damped least-squares (Levenberg-Marquardt style).

        Parameters
        ----------
        x_target : target end-effector position [3] or pose [7] (pos + quat)
        q_init   : initial joint configuration [n] (default: zeros)
        max_iter : maximum iterations
        tol      : position error tolerance (meters)
        damping  : adaptive damping

        Returns
        -------
        q : joint configuration [n] that reaches x_target, or None if failed
        """
        if q_init is None:
            q = np.zeros(self.n)
        else:
            q = np.asarray(q_init, dtype=float).copy()

        x_target = np.asarray(x_target, dtype=float)
        position_only = (len(x_target) == 3)

        for _ in range(max_iter):
            x_current, R_current = self.forward_kinematics(q)

            # Position error
            e_pos = x_target[:3] - x_current
            pos_error = np.linalg.norm(e_pos)

            if position_only:
                if pos_error < tol:
                    return q
                # Use only position part of Jacobian
                J = self.jacobian(q)[:3, :]  # [3 x n]
                dx = e_pos
            else:
                # Full 6D pose (position + orientation)
                # x_target[3:7] is quaternion [w, x, y, z]
                R_target = Rotation.from_quat(x_target[3:7]).as_matrix()

                # Orientation error (axis-angle)
                R_error = R_target @ R_current.T
                rotvec = Rotation.from_matrix(R_error).as_rotvec()
                e_ori = rotvec

                if pos_error < tol and np.linalg.norm(e_ori) < tol:
                    return q

                J = self.jacobian(q)  # [6 x n]
                dx = np.concatenate([e_pos, e_ori])

            # Damped least-squares step
            Jpinv = self.pseudo_inverse(J)
            dq = Jpinv @ dx
            # dq = J.T @ np.linalg.solve(J@J.T + 0.01* np.eye(6), dx)

            # Line search with step size decay
            alpha = 1.0
            success = False
            for _ in range(10):
                if self.model is not None:
                    q_new = pin.integrate(self.model, q, alpha * dq)
                else:
                    q_new = q + alpha * dq

                x_new, R_new = self.forward_kinematics(q_new)

                e_pos_new = x_target[:3] - x_new

                if position_only:
                    e_new = e_pos_new
                else:
                    R_error_new = R_target @ R_new.T
                    e_ori_new = Rotation.from_matrix(R_error_new).as_rotvec()
                    e_new = np.concatenate([e_pos_new, e_ori_new])

                if np.linalg.norm(e_new) < tol:
                    q = q_new
                    success = True
                    break

                alpha *= 0.5

            if not success:
                q = q + 0.1 * dq

            # Clamp to joint limits after each iteration
            if self.q_min is not None:
                q = np.maximum(q, self.q_min)
            if self.q_max is not None:
                q = np.minimum(q, self.q_max)

        # Final check + joint limit clamp
        if self.q_min is not None:
            q = np.maximum(q, self.q_min)
        if self.q_max is not None:
            q = np.minimum(q, self.q_max)

        x_final, R_final = self.forward_kinematics(q)
        final_error = np.linalg.norm(x_target[:3] - x_final)

        if final_error < tol * 5:
            return q

        return None


    # ------------------------------------------------------------------
    # Simplified fallback (no Pinocchio) - uses random DH-like matrix
    # ------------------------------------------------------------------

    def _fk_simplified(self, q):
        """Very rough FK: sum of joint contributions along z-axis."""
        link_lengths = np.array([0.333, 0.316, 0.384, 0.0, 0.107, 0.0, 0.088])[:self.n]
        pos = np.zeros(3)
        for i, (l, qi) in enumerate(zip(link_lengths, q)):
            pos[0] += l * np.cos(np.sum(q[:i+1]))
            pos[2] += l * np.sin(np.sum(q[:i+1]))
        R = np.eye(3)
        return pos, R

    def _jacobian_simplified(self, q) -> np.ndarray:
        """
        Numerical Jacobian via finite differences (simplified fallback).
        """
        eps = 1e-5
        n = self.n
        p0, _ = self._fk_simplified(q)
        J = np.zeros((6, n))
        for i in range(n):
            dq = np.zeros(n)
            dq[i] = eps
            p1, _ = self._fk_simplified(q + dq)
            J[:3, i] = (p1 - p0) / eps
            # angular velocity: simplified as unit z-axis rotation
            J[3:, i] = np.array([0, 0, 1])
        return J


if __name__ == "__main__":
    kin = ManipulatorKinematics()  # simplified mode
    n = kin.n
    q = np.zeros(n)
    dq0 = np.random.randn(n)

    print("=== kinematics.py unit tests ===")

    J = kin.jacobian(q)
    print(f"J shape: {J.shape}  (expected (6,{n}))")

    N = kin.null_space_projector(q)
    print(f"N shape: {N.shape}  (expected ({n},{n}))")

    # Key property: J @ N should be ≈ 0
    residual = np.linalg.norm(J @ N)
    print(f"||J @ N|| = {residual:.2e}  (should be ~0)")

    dq_null = kin.null_space_velocity(q, dq0)
    ee_vel = J @ dq_null
    print(f"||J @ N @ dq0|| = {np.linalg.norm(ee_vel):.2e}  (should be ~0)")

    # Test inverse kinematics
    print("\n=== Testing inverse_kinematics ===")
    q_test = np.random.uniform(-1, 1, n)
    x_target, r_target = kin.forward_kinematics(q_test)
    print(f"Target position: {x_target}")

    q_solved = kin.inverse_kinematics(np.concatenate((x_target, Rotation.from_matrix(r_target).as_quat())))
    if q_solved is not None:
        x_solved, r_solved = kin.forward_kinematics(q_solved)    
        ik_error = np.linalg.norm(x_target - x_solved) + np.linalg.norm(Rotation.from_matrix(r_solved.T @ r_target).as_rotvec())
        print(f"IK solved: error = {ik_error:.2e} m")
        print(f"IK test {'PASSED' if ik_error < 1e-3 else 'FAILED'}")
    else:
        print("IK test FAILED: no solution found")

    print("\nkinematics.py unit test PASSED" if residual < 1e-8 else
          "WARNING: residual larger than expected (check jacobian accuracy)")
