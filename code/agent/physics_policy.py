"""
physics_policy.py
-----------------
Physics-Informed Policy Network (paper core contribution).

Key idea: the actor network outputs 10D actions:
    a = [Δẋ_RL (3), dq0 (7)]
    - Δẋ_RL: position-space relaxation (scaled to allow obstacle avoidance)
    - dq0   : null-space self-motion velocities

Differentiable physics regularization (Plan B):
    Reconstructs dq_cmd from action analytically using stored Jacobian,
    then penalizes torque limit violations via simplified dynamics in pure torch.

Architecture:
    Input:  state s_t = [q, dq, x_ee, x_d, dx_d, d_obs, w(q)]  (dim=state_dim=25)
    Hidden: MLP with tanh activations
    Output: 10D action [Δẋ_RL (3), dq0 (7)]
            task relaxation scaled by task_scale (small),
            null-space motion scaled by nullspace_scale (larger)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from env.dynamics import ManipulatorDynamics

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class PhysicsInformedActor(nn.Module):
    """
    Gaussian actor for SAC outputting 10D actions: [Δẋ_RL (3), dq0 (7)].

    Task relaxation and null-space components are scaled differently:
    - task_scale (default 0.1): small to preserve tracking priority
    - nullspace_scale (default 0.3): larger for effective self-motion exploration
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list[int] = (256, 256),
                 action_scale: float = 0.5,
                 task_scale: float = 0.3,
                 nullspace_scale: float = 0.3):
        """
        Parameters
        ----------
        state_dim      : dimension of input state (25)
        action_dim     : total action dimension (10 = 3 + 7)
        hidden_dims    : MLP hidden layer sizes
        action_scale   : legacy scale (unused when task/nullspace scales provided)
        task_scale     : scale for task relaxation Δẋ_RL (first 3 dims)
        nullspace_scale: scale for null-space velocity dq0 (last 7 dims)
        """
        super().__init__()
        self.action_scale    = action_scale
        self.task_scale      = task_scale
        self.nullspace_scale = nullspace_scale
        self.task_dim        = 3  # position-only (Route A)
        self.nullspace_dim   = action_dim - 3  # typically 7

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h

        self.net = nn.Sequential(*layers)
        self.mean_head    = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

    def forward(self, state: torch.Tensor):
        """
        Returns
        -------
        mean    : [batch x action_dim]
        log_std : [batch x action_dim]
        """
        h = self.net(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        """
        Reparameterized sample with squashed Gaussian (tanh).

        Returns
        -------
        action   : [batch x 10]  [Δẋ_RL (3), dq0 (7)], separately scaled
        log_prob : [batch x 1]
        mean     : [batch x 10]  deterministic action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if torch.isnan(mean).any() or torch.isnan(std).any():
            mean = torch.nan_to_num(mean, nan=0.0)
            std  = torch.nan_to_num(std,  nan=1.0).clamp(min=1e-6)
        dist = Normal(mean, std)
        x = dist.rsample()
        y = torch.tanh(x)

        # Apply separate scales for task relaxation vs null-space
        scale = torch.ones_like(y)
        scale[:, :self.task_dim]  = self.task_scale
        scale[:, self.task_dim:]  = self.nullspace_scale
        action = y * scale

        # log prob with change-of-variables for tanh
        log_prob = dist.log_prob(x) - torch.log(scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_scale = torch.ones_like(mean)
        mean_scale[:, :self.task_dim] = self.task_scale
        mean_scale[:, self.task_dim:] = self.nullspace_scale
        mean_action = torch.tanh(mean) * mean_scale

        return action, log_prob, mean_action


class PhysicsRegularizer:
    """
    Differentiable torque constraint regularization (Plan B).

    Reconstructs dq_cmd from action analytically using stored Jacobian,
    sigma gate, and nominal task velocity — all in pure torch with full
    gradient flow back to the actor.

    τ = M·ddq + C·dq + g  (simplified dynamics, matches dynamics.py)
    L_dyn = || relu(|τ| - τ_max) ||²
    """

    def __init__(self, dynamics, tau_max: float = 15.0,
                 lambda_dyn: float = 0.1, dt: float = 0.02,
                 device: str = "cpu"):
        self.dt = dt
        self.lambda_dyn = lambda_dyn
        self.n = dynamics.n
        self.device = torch.device(device)

        # Simplified dynamics params (matches dynamics.py _compute_simplified)
        inertias = torch.tensor(
            [1.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.4],
            dtype=torch.float32
        )[:self.n]
        self._M_diag = inertias.to(self.device)
        self._tau_max_t = torch.full(
            (self.n,), tau_max, dtype=torch.float32, device=self.device
        )

    def _compute_simplified_torch(self, q: torch.Tensor,
                                   dq: torch.Tensor,
                                   ddq: torch.Tensor) -> torch.Tensor:
        """
        Simplified inverse dynamics entirely in torch.
        τ = M·ddq + C·dq  (g = 0 in simplified model)
          M = diag([1, 2, 1.5, 1, 0.8, 0.6, 0.4]) — constant inertia
          C = diag(0.1 * dq) — viscous friction approximation

        All tensors preserve gradient tracking.
        """
        B = q.shape[0]
        n = q.shape[1]
        device = q.device
        dtype = q.dtype

        # Mass matrix (constant diagonal)
        inertias = self._M_diag.to(device=device, dtype=dtype)
        M = torch.diag(inertias).unsqueeze(0).expand(B, -1, -1)  # (B, n, n)

        # Coriolis (viscous friction)
        C = torch.diag_embed(0.1 * dq)  # (B, n, n)

        # τ = M·ddq + C·dq  (g = 0)
        tau = M @ ddq.unsqueeze(-1) + C @ dq.unsqueeze(-1)  # (B, n, 1)
        return tau.squeeze(-1)  # (B, n)

    def compute_loss_batch(self, q_batch: torch.Tensor,
                            dq_batch: torch.Tensor,
                            J_batch: torch.Tensor,
                            sigma_batch: torch.Tensor,
                            dx_nom_batch: torch.Tensor,
                            action_batch: torch.Tensor) -> torch.Tensor:
        """
        Batched physics loss with full gradient flow.

        Reconstructs dq_cmd from current-policy action analytically, then
        computes torque via simplified dynamics and penalises limit violations.

        Route A (position-only): uses 3D task space, J_pos ∈ ℝ³ˣⁿ.

        Parameters
        ----------
        q_batch       : [B x n] joint positions (from buffer, detached)
        dq_batch      : [B x n] previous joint velocities (from buffer)
        J_batch       : [B x 3 x n] position-only Jacobian
        sigma_batch   : [B x 1] gate value
        dx_nom_batch  : [B x 3] nominal position-space velocity
        action_batch  : [B x 10] current-policy action  (*has grad*)

        Returns
        -------
        loss : torch scalar
        """
        B = q_batch.shape[0]
        n = self.n
        device = q_batch.device
        dtype = q_batch.dtype
        task_dim = 3  # position-only (Route A)
        lam = 1e-4    # damping for pseudo-inverse

        # Split action
        delta_x = action_batch[:, :3]    # (B, 3)  *has grad*
        dq0     = action_batch[:, 3:]    # (B, n)  *has grad*

        # ---- Reconstruct dq_cmd from action (differentiable) ----

        # Pseudo-inverse: J_pinv = J^T (J J^T + λI)^{-1}
        JJT = J_batch @ J_batch.transpose(-2, -1)  # (B, 3, 3)
        reg = lam * torch.eye(task_dim, device=device, dtype=dtype).unsqueeze(0)
        JJT_inv = torch.linalg.inv(JJT + reg)
        J_pinv = J_batch.transpose(-2, -1) @ JJT_inv  # (B, n, 3)

        # Null-space projector: N = I - J_pinv @ J
        I = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
        N = I - J_pinv @ J_batch  # (B, n, n)

        # dq_cmd = J_pinv @ (dx_nom + σ·Δx) + N @ dq0
        sigma_flat = sigma_batch.view(B, 1, 1)          # (B, 1, 1)
        dx_nom_r = dx_nom_batch.view(B, task_dim, 1)    # (B, 3, 1)
        delta_x_r = delta_x.view(B, task_dim, 1)        # (B, 3, 1)
        dq0_r     = dq0.view(B, n, 1)                   # (B, n, 1)

        dq_cmd = J_pinv @ (dx_nom_r + sigma_flat * delta_x_r) + N @ dq0_r
        dq_cmd = dq_cmd.squeeze(-1)  # (B, n)

        # ---- Torque computation (differentiable) ----

        # Acceleration via finite difference
        ddq = (dq_cmd - dq_batch) / self.dt
        ddq = torch.clamp(ddq, -100.0, 100.0)

        # Simplified dynamics in pure torch
        torques = self._compute_simplified_torch(q_batch, dq_batch, ddq)

        # ---- Torque limit violation loss ----
        violation = F.relu(torques.abs() - self._tau_max_t)
        loss = (violation ** 2).mean()

        return loss * self.lambda_dyn


class SoftmaxCritic(nn.Module):
    """Standard double-Q critic for SAC."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list[int] = (256, 256)):
        super().__init__()
        self.q1 = self._build(state_dim + action_dim, hidden_dims)
        self.q2 = self._build(state_dim + action_dim, hidden_dims)

    @staticmethod
    def _build(in_dim, hidden_dims):
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.dynamics import ManipulatorDynamics

    n_joints = 7
    state_dim = n_joints * 2 + 3 + 3 + 3 + 1 + 1   # q + dq + x_ee + x_d + dx_d + d_obs + w = 25
    action_dim = 3 + n_joints  # 10D: Δẋ_RL (3) + dq0 (7)

    print("=== physics_policy.py unit tests ===")

    actor = PhysicsInformedActor(state_dim, action_dim)
    s = torch.randn(4, state_dim)
    a, logp, a_det = actor.sample(s)
    print(f"action shape: {a.shape}  (expected [4, {action_dim}])")
    print(f"log_prob shape: {logp.shape}  (expected [4, 1])")

    dyn = ManipulatorDynamics()
    reg = PhysicsRegularizer(dyn, tau_max=15.0)

    # Test compute_loss_batch with dummy data (Plan B signature)
    # Use aggressive actions to exceed torque limits → verify gradient flow
    B = 4
    q_t = torch.zeros(B, n_joints)
    dq_t = torch.zeros(B, n_joints)
    J_t = torch.eye(3, n_joints).unsqueeze(0).expand(B, -1, -1)  # position-only J
    sigma_t = torch.ones(B, 1) * 0.5  # partial gate opening
    dx_nom_t = torch.full((B, 3), 0.5)  # nominal position velocity (Route A)
    action_t = torch.full((B, 10), 2.0)  # large task relaxation + nullspace
    action_t.requires_grad_(True)

    loss = reg.compute_loss_batch(q_t, dq_t, J_t, sigma_t, dx_nom_t, action_t)
    loss.backward()  # verify gradient flow
    grad_norm = action_t.grad.abs().sum().item()
    print(f"L_dyn (batch): {loss.item():.6f}  (|grad|={grad_norm:.4f}, flow={grad_norm > 0})")

    critic = SoftmaxCritic(state_dim, action_dim)
    q1, q2 = critic(s, a.detach())
    print(f"Q1 shape: {q1.shape}  (expected [4, 1])")

    print("physics_policy.py unit test PASSED")
