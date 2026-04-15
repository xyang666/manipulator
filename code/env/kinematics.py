"""
kinematics.py
-------------
Kinematics utilities for a 7-DOF manipulator:
  - forward_kinematics(q)         -> end-effector SE3 pose
  - jacobian(q)                   -> J ∈ R^{6 x n}
  - pseudo_inverse(J)             -> J† (damped least-squares)
  - null_space_projector(q)       -> N = I - J†J ∈ R^{n x n}
  - null_space_velocity(q, dq0)   -> q̇_null = N(q) @ dq0

Usage:
    kin = ManipulatorKinematics(urdf_path)
    x, R = kin.forward_kinematics(q)
    J    = kin.jacobian(q)
    N    = kin.null_space_projector(q)
"""

import numpy as np

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
                 damping: float = 1e-4):
        """
        Parameters
        ----------
        urdf_path : path to URDF file (None → simplified mode)
        n_joints  : number of joints
        damping   : damping factor λ for damped pseudo-inverse
        """
        self.n = n_joints
        self.damping = damping
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
            if "tcp" in f.name.lower() or "_ee" in f.name.lower():
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

    def pseudo_inverse(self, J: np.ndarray) -> np.ndarray:
        """
        Damped least-squares pseudo-inverse:
          J† = J^T (J J^T + λ²I)^{-1}
        """
        lam2 = self.damping ** 2
        m = J.shape[0]
        return J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(m))

    def null_space_projector(self, q: np.ndarray) -> np.ndarray:
        """
        N(q) = I - J†(q) J(q)   ∈ R^{n x n}

        Any vector q̇₀ projected through N satisfies J @ (N @ q̇₀) ≈ 0,
        meaning it produces no end-effector motion (pure self-motion).
        """
        J = self.jacobian(q)
        Jpinv = self.pseudo_inverse(J)
        return np.eye(self.n) - Jpinv @ J

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
        return Jpinv @ dx_desired + N @ dq0

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

    print("kinematics.py unit test PASSED" if residual < 1e-8 else
          "WARNING: residual larger than expected (check jacobian accuracy)")
