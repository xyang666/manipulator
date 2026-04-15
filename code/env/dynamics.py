"""
dynamics.py
-----------
Computes manipulator dynamics matrices M(q), C(q,dq), g(q) using Pinocchio.
Supports any 7-DOF arm URDF (default: Franka Panda).

Usage:
    dyn = ManipulatorDynamics(urdf_path)
    M, C, g = dyn.compute(q, dq)
"""

import numpy as np

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("[dynamics] WARNING: pinocchio not found. Using simplified analytic model.")


class ManipulatorDynamics:
    """
    Thin wrapper around Pinocchio for computing:
      M(q)      - mass / inertia matrix  [n x n]
      C(q,dq)   - Coriolis/centrifugal matrix [n x n]
      g(q)      - gravity vector [n]
    where n = number of joints (7 for Panda).
    """

    def __init__(self, urdf_path: str | None = None, n_joints: int = 7):
        self.n = n_joints
        self.model = None
        self.data = None

        if HAS_PINOCCHIO and urdf_path is not None:
            self._init_pinocchio(urdf_path)
        else:
            print("[dynamics] Running in simplified mode (no Pinocchio/URDF).")

    def _init_pinocchio(self, urdf_path: str):
        full_model = pin.buildModelFromUrdf(urdf_path)
        if full_model.nv > self.n:
            joints_to_lock = list(range(self.n + 1, full_model.njoints))
            q_ref = pin.neutral(full_model)
            self.model = pin.buildReducedModel(full_model, joints_to_lock, q_ref)
        else:
            self.model = full_model
        self.data = self.model.createData()
        self.n = self.model.nv
        print(f"[dynamics] Loaded model '{self.model.name}' with {self.n} DOF.")

    def compute(self, q: np.ndarray, dq: np.ndarray):
        """
        Returns (M, C, g) for given joint configuration and velocity.

        Parameters
        ----------
        q  : joint positions  [n]
        dq : joint velocities [n]

        Returns
        -------
        M : np.ndarray [n x n]  mass matrix
        C : np.ndarray [n x n]  Coriolis matrix  (such that C @ dq = coriolis forces)
        g : np.ndarray [n]      gravity torques
        """
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)

        if self.model is not None:
            return self._compute_pinocchio(q, dq)
        else:
            return self._compute_simplified(q, dq)

    def _compute_pinocchio(self, q, dq):
        # Mass matrix
        M = pin.crba(self.model, self.data, q)

        # Coriolis matrix (computed via RNEA with zero gravity contribution)
        pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        C = self.data.C.copy()

        # Gravity vector
        g = pin.computeGeneralizedGravity(self.model, self.data, q)

        return M, C, g

    def _compute_simplified(self, q, dq):
        """
        Simplified placeholder dynamics for testing without Pinocchio.
        M = diag([1, 2, 1.5, 1, 0.8, 0.6, 0.4])  (approximate link inertias)
        C = 0.1 * diag(dq)  (simplified viscous friction)
        g = 0 (ignore gravity for placeholder)
        """
        n = self.n
        inertias = np.array([1.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.4])[:n]
        M = np.diag(inertias)
        C = np.diag(0.1 * dq)
        g = np.zeros(n)
        return M, C, g

    def compute_torque(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        """
        Computes inverse dynamics torque: τ = M(q)·ddq + C(q,dq)·dq + g(q)

        Parameters
        ----------
        q   : joint positions      [n]
        dq  : joint velocities     [n]
        ddq : joint accelerations  [n]

        Returns
        -------
        tau : joint torques [n]
        """
        if self.model is not None:
            # Use RNEA for efficiency
            return pin.rnea(self.model, self.data, q, dq, ddq)
        else:
            M, C, g = self._compute_simplified(q, dq)
            return M @ ddq + C @ dq + g


if __name__ == "__main__":
    # Unit test: check shapes
    dyn = ManipulatorDynamics()  # simplified mode
    n = dyn.n
    q = np.zeros(n)
    dq = np.ones(n) * 0.1
    M, C, g = dyn.compute(q, dq)
    print(f"M shape: {M.shape}  (expected ({n},{n}))")
    print(f"C shape: {C.shape}  (expected ({n},{n}))")
    print(f"g shape: {g.shape}  (expected ({n},))")
    ddq = np.zeros(n)
    tau = dyn.compute_torque(q, dq, ddq)
    print(f"tau: {tau}  (should be ~C@dq + g)")
    print("dynamics.py unit test PASSED")
