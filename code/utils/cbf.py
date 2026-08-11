"""
cbf.py
------
Control Barrier Function safety filter for manipulator obstacle avoidance.

Applies a CBF-QP filter to joint velocity commands before execution:
    Barrier: h(q) = d_obs(q) - d_safe
    CBF:     dh/dt ≥ -α·h  →  ∇h·dq ≥ -α·h
    QP:      min ||dq - dq_cmd||²  s.t.  ∇h·dq ≥ -α·h

The QP has a closed-form solution (single constraint), making it O(n).
Uses numerical central-difference gradient for the barrier function.

Author: xie yang
Date:   2025-06

"""

import numpy as np


class CBFController:
    """
    Control Barrier Function safety filter.

    Filters joint velocity commands to ensure the minimum distance to
    obstacles never decreases below d_safe, using a CBF-QP formulation.

    Parameters
    ----------
    sdf        : ObstacleSDF instance (provides min_distance(q, kinematics))
    kinematics : ManipulatorKinematics instance (provides FK + capsules)
    d_safe     : safety distance threshold (m)
    alpha      : CBF convergence gain (larger = more aggressive)
    eps        : finite-difference step size for numerical gradient
    """

    def __init__(self, sdf, kinematics, d_safe=0.06, alpha=1.0, eps=1e-5,
                 self_d_safe=0.02):
        self.sdf = sdf
        self.kin = kinematics
        self.d_safe = d_safe
        self.alpha = alpha
        self.eps = eps
        self.self_d_safe = self_d_safe
        self.n = kinematics.n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def barrier(self, q: np.ndarray) -> float:
        """
        Compute barrier function h(q) = d_obs(q) - d_safe.

        Returns inf when no obstacles are present (CBF constraint
        trivially satisfied).
        """
        x_ee, _ = self.kin.forward_kinematics(q)
        d_obs = self.sdf.min_distance(x_ee, q, kinematics=self.kin)
        if not np.isfinite(d_obs):
            return np.inf
        return float(d_obs) - self.d_safe

    def compute_gradient(self, q: np.ndarray) -> tuple:
        """
        Numerical gradient of barrier function via central differences.

        Parameters
        ----------
        q : joint positions (n,)

        Returns
        -------
        grad : (n,) array, ∂h/∂q
        h    : scalar, h(q) = d_obs(q) - d_safe
        """
        h0 = self.barrier(q)
        if not np.isfinite(h0):
            return np.zeros(self.n), h0

        grad = np.zeros(self.n)
        eps = self.eps

        for i in range(self.n):
            q_plus = q.copy()
            q_plus[i] += eps
            h_plus = self.barrier(q_plus)

            q_minus = q.copy()
            q_minus[i] -= eps
            h_minus = self.barrier(q_minus)

            grad[i] = (h_plus - h_minus) / (2.0 * eps)

        # Clamp gradient magnitude to prevent numerical blowup
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 100.0:
            grad = grad / grad_norm * 100.0

        return grad, h0

    def self_barrier(self, q: np.ndarray) -> float:
        """Minimum non-adjacent link clearance above the self margin."""
        distances = self.kin.compute_self_distances(q)
        if len(distances) == 0:
            return np.inf
        return float(np.min(distances)) - self.self_d_safe

    def _gradient_of(self, q: np.ndarray, barrier_fn) -> tuple[np.ndarray, float]:
        """Central-difference gradient shared by obstacle and self barriers."""
        h0 = barrier_fn(q)
        if not np.isfinite(h0):
            return np.zeros(self.n), h0
        grad = np.zeros(self.n)
        for i in range(self.n):
            q_plus, q_minus = q.copy(), q.copy()
            q_plus[i] += self.eps
            q_minus[i] -= self.eps
            grad[i] = (barrier_fn(q_plus) - barrier_fn(q_minus)) / (2 * self.eps)
        norm = np.linalg.norm(grad)
        if norm > 100.0:
            grad *= 100.0 / norm
        return grad, h0

    def _project_constraint(self, dq_cmd: np.ndarray, grad: np.ndarray,
                            h: float) -> tuple[np.ndarray, bool]:
        """Project one velocity command onto one CBF half-space."""
        norm_sq = float(np.dot(grad, grad))
        if not np.isfinite(h) or norm_sq < 1e-12:
            return dq_cmd, False
        lhs = float(np.dot(grad, dq_cmd))
        rhs = -self.alpha * max(h, -0.5)
        if lhs >= rhs:
            return dq_cmd, False
        return dq_cmd + ((rhs - lhs) / norm_sq) * grad, True

    def filter(self, dq_cmd: np.ndarray, q: np.ndarray) -> tuple:
        """
        Apply CBF-QP safety filter.

        Parameters
        ----------
        dq_cmd : nominal joint velocity command (n,)
        q      : current joint positions (n,)

        Returns
        -------
        dq_filtered : CBF-modified joint velocity (n,)
        info        : dict with keys:
            - active: bool, whether CBF constraint was active
            - h     : float, current barrier value
            - dq_norm: float, ‖dq_filtered - dq_cmd‖ (modification magnitude)
        """
        obstacle_grad, obstacle_h = self.compute_gradient(q)
        self_grad, self_h = self._gradient_of(q, self.self_barrier)
        dq_filtered, obstacle_active = self._project_constraint(
            dq_cmd.copy(), obstacle_grad, obstacle_h
        )
        dq_filtered, self_active = self._project_constraint(
            dq_filtered, self_grad, self_h
        )
        info = {
            "active": obstacle_active or self_active,
            "obstacle_active": obstacle_active,
            "self_active": self_active,
            "h": min(obstacle_h, self_h),
            "obstacle_h": obstacle_h,
            "self_h": self_h,
            "dq_norm": float(np.linalg.norm(dq_filtered - dq_cmd)),
        }

        return dq_filtered, info


if __name__ == "__main__":
    print("=== CBF unit test ===")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.kinematics import ManipulatorKinematics
    from utils.sdf import ObstacleSDF

    # Try Pinocchio URDF for proper q-dependent capsules; fall back to simplified
    _urdf_candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     ".venv/lib/python3.12/site-packages/cmeel.prefix"
                     "/share/example-robot-data/robots/panda_description/urdf/panda.urdf"),
    ]
    _chosen_urdf = None
    for _p in _urdf_candidates:
        if os.path.exists(_p):
            _chosen_urdf = _p
            break
    kin = ManipulatorKinematics(_chosen_urdf)

    sdf = ObstacleSDF(n_obstacles=3, radius=0.1)
    obs_centers = [[0.4, 0.0, 0.5], [0.4, 0.0, 0.6], [0.4, 0.1, 0.5]]
    sdf.set_static_obstacles(obs_centers)

    cbf = CBFController(sdf, kin, d_safe=0.06, alpha=1.0)

    # Test safe configuration
    q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
    dq = np.ones(7) * 0.1

    h = cbf.barrier(q)
    print(f"h(q) = {h:.4f} (d_obs - d_safe)")

    grad, h_val = cbf.compute_gradient(q)
    has_ok_grad = np.linalg.norm(grad) > 1e-6 or not np.isfinite(h_val)
    print(f"grad norm: {np.linalg.norm(grad):.6f}  (ok={has_ok_grad})")

    dq_filt, info = cbf.filter(dq, q)
    print(f"CBF active: {info['active']}")
    print(f"dq modification: {info['dq_norm']:.6f}")

    # Verify CBF condition satisfied after filtering for unsafe case
    print("\n-- Unsafe test --")
    # Move toward obstacle
    if np.linalg.norm(grad) > 1e-6:
        dq_toward = -grad * 0.5
    else:
        dq_toward = -np.ones(7) * 0.2
    dq_filt2, info2 = cbf.filter(dq_toward, q)
    print(f"CBF active: {info2['active']}")

    grad2, h2 = cbf.compute_gradient(q)
    if info2["active"]:
        cbf_lhs = np.dot(grad2, dq_filt2)
        cbf_rhs = -cbf.alpha * h2
        print(f"∇h·dq_filtered = {cbf_lhs:.6f}  ≥  -α·h = {cbf_rhs:.6f}: "
              f"{cbf_lhs >= cbf_rhs - 1e-6}")

    # Test with no obstacles
    print("\n-- No-obstacle test --")
    empty_sdf = ObstacleSDF(n_obstacles=0)
    cbf_no_obs = CBFController(empty_sdf, kin)
    dq_filt3, info3 = cbf_no_obs.filter(dq, q)
    print(f"Active: {info3['active']} (should be False)")
    assert not info3["active"], "No obstacles should not activate CBF"

    print("\nCBF unit test PASSED")
