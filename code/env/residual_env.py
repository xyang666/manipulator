"""Residual SAC environment with an unstructured 7D joint-velocity residual."""

import numpy as np

from env.manipulator_env import ManipulatorEnv


class ResidualEnv(ManipulatorEnv):
    """Add a direct joint residual to the nominal task-space controller.

    The policy emits ``delta_dq`` in joint coordinates. Internally it is
    decomposed into row-space and null-space coordinates before calling the
    shared environment. This preserves the exact residual while keeping all
    simulation, collision, and termination behavior common across methods.
    """

    def step(self, action: np.ndarray):
        residual = np.asarray(action, dtype=float)
        if residual.shape != (self.n,):
            raise ValueError(f"expected joint residual shape {(self.n,)}, got {residual.shape}")
        jacobian = self.kin.jacobian_position(self.q)
        basis = self.kin.null_space_basis_position(self.q)
        structured = np.concatenate([jacobian @ residual, basis.T @ residual])
        previous_override = getattr(self, "sigma_override", None)
        self.sigma_override = 1.0
        try:
            return super().step(structured)
        finally:
            self.sigma_override = previous_override
