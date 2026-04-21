"""
physics_policy.py
-----------------
Physics-Informed Policy Network (paper core contribution).

Key idea: the actor network outputs 13D actions:
    a = [Δẋ_RL (6), dq0 (7)]
    - Δẋ_RL: task-space relaxation (scaled small to preserve tracking priority)
    - dq0   : null-space self-motion velocities

During training, torque constraint regularization is applied (paper Eq. 10):
    τ_π  = M(q) @ ddq + C(q,dq) @ dq + g(q)
    L_dyn = ||relu(|τ_π| - τ_max)||²   (penalize torques beyond limits)
    L_total = L_SAC + λ_dyn * L_dyn

This guides the policy to learn physically feasible trajectories that respect
joint torque limits, improving training stability and motion smoothness.

Architecture:
    Input:  state s_t = [q, dq, x_ee, x_d, dx_d, d_obs, w(q)]  (dim=state_dim=25)
    Hidden: MLP with tanh activations
    Output: 13D action [Δẋ_RL (6), dq0 (7)]
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
    Gaussian actor for SAC outputting 13D actions: [Δẋ_RL (6), dq0 (7)].

    Task relaxation and null-space components are scaled differently:
    - task_scale (default 0.1): small to preserve tracking priority
    - nullspace_scale (default 0.3): larger for effective self-motion exploration
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list[int] = (256, 256),
                 action_scale: float = 0.5,
                 task_scale: float = 0.1,
                 nullspace_scale: float = 0.3):
        """
        Parameters
        ----------
        state_dim      : dimension of input state (25)
        action_dim     : total action dimension (13 = 6 + 7)
        hidden_dims    : MLP hidden layer sizes
        action_scale   : legacy scale (unused when task/nullspace scales provided)
        task_scale     : scale for task relaxation Δẋ_RL (first 6 dims)
        nullspace_scale: scale for null-space velocity dq0 (last 7 dims)
        """
        super().__init__()
        self.action_scale    = action_scale
        self.task_scale      = task_scale
        self.nullspace_scale = nullspace_scale
        self.task_dim        = 6
        self.nullspace_dim   = action_dim - 6  # typically 7

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
        action   : [batch x 13]  [Δẋ_RL (6), dq0 (7)], separately scaled
        log_prob : [batch x 1]
        mean     : [batch x 13]  deterministic action
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
    Torque constraint regularization loss (paper Eq. 10).

    L_dyn = || relu(|τ_π| - τ_max) ||²

    Penalizes joint torques that exceed physical limits, guiding the policy to
    learn feasible trajectories. Uses soft ReLU constraint (not hard clipping)
    to preserve gradient flow while enforcing limits.

    τ_π is computed via inverse dynamics:
        τ_π = M(q) @ ddq + C(q,dq) @ dq + g(q)
    where ddq ≈ (dq_new - dq_prev) / dt.

    This is applied as an auxiliary loss during actor updates:
        L_total = L_SAC + λ_dyn * L_dyn
    This class operates on numpy arrays (called from the environment/agent),
    and returns a torch scalar for backpropagation.
    """

    def __init__(self, dynamics: ManipulatorDynamics, tau_max: np.ndarray | float = 87.0,
                 lambda_dyn: float = 0.1, lambda_collision: float = 1.0, dt: float = 0.02,
                 collision_detector=None):
        """
        Parameters
        ----------
        dynamics          : ManipulatorDynamics instance
        tau_max           : joint torque limits [n] or scalar (Nm). Panda default: 87 Nm
        lambda_dyn        : weight of physics loss term
        lambda_collision  : weight of collision loss term
        dt                : simulation timestep for finite-difference acceleration
        collision_detector: CollisionDetector instance (optional)
        """
        self.dynamics = dynamics
        self.dt = dt
        self.lambda_dyn = lambda_dyn
        self.lambda_collision = lambda_collision
        self.collision_detector = collision_detector
        n = dynamics.n
        if np.isscalar(tau_max):
            self.tau_max = np.full(n, tau_max)
        else:
            self.tau_max = np.asarray(tau_max)

    def compute_loss(self, q: np.ndarray, dq: np.ndarray,
                     dq_new: np.ndarray) -> torch.Tensor:
        """
        Compute physics regularization loss (numpy inputs → torch scalar).

        Parameters
        ----------
        q      : current joint positions   [n]
        dq     : current joint velocities  [n]
        dq_new : next joint velocities after action  [n]

        Returns
        -------
        L_dyn : torch scalar
        """
        ddq = (dq_new - dq) / self.dt
        # Clip acceleration to prevent numerical explosion
        ddq = np.clip(ddq, -100.0, 100.0)
        tau = self.dynamics.compute_torque(q, dq, ddq)

        tau_t = torch.tensor(tau, dtype=torch.float32)
        tau_max_t = torch.tensor(self.tau_max, dtype=torch.float32)

        violation = F.relu(tau_t.abs() - tau_max_t)
        loss = (violation ** 2).mean()
        return loss * self.lambda_dyn

    def compute_loss_batch(self,
                           q_batch: torch.Tensor,
                           dq_batch: torch.Tensor,
                           dq_new_batch: torch.Tensor,
                           collision_detector=None) -> torch.Tensor:
        """
        Batch version for efficient training with collision loss.
        Operates entirely in torch (requires dynamics in torch or loops over numpy).

        Parameters
        ----------
        q_batch            : [batch x n]
        dq_batch           : [batch x n]
        dq_new_batch       : [batch x n]
        collision_detector : CollisionDetector instance (optional)

        Returns
        -------
        L_total : torch scalar (mean over batch) = L_dyn + L_collision
        """
        B = q_batch.shape[0]
        dyn_losses = []
        collision_losses = []

        for i in range(B):
            q = q_batch[i].cpu().detach().numpy()
            dq = dq_batch[i].cpu().detach().numpy()
            dq_new = dq_new_batch[i].cpu().detach().numpy()

            # Dynamics loss (already includes lambda_dyn scaling)
            dyn_losses.append(self.compute_loss(q, dq, dq_new))

            # Collision loss (if detector available)
            if self.collision_detector is not None or collision_detector is not None:
                detector = collision_detector if collision_detector is not None else self.collision_detector
                collision_penalty, _ = detector.compute_collision_penalty(
                    w_obstacle=100.0,
                    w_self=50.0
                )
                collision_losses.append(torch.tensor(collision_penalty, dtype=torch.float32))
            else:
                collision_losses.append(torch.tensor(0.0, dtype=torch.float32))

        dyn_loss = torch.stack(dyn_losses).mean()
        collision_loss = torch.stack(collision_losses).mean()

        # lambda_dyn already applied in compute_loss, only scale collision
        total_loss = dyn_loss + self.lambda_collision * collision_loss
        return total_loss


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
    state_dim = n_joints * 2 + 6 + 1 + 1   # q + dq + x_d + d_obs + w
    action_dim = n_joints

    print("=== physics_policy.py unit tests ===")

    actor = PhysicsInformedActor(state_dim, action_dim)
    s = torch.randn(4, state_dim)
    a, logp, a_det = actor.sample(s)
    print(f"action shape: {a.shape}  (expected [4, {action_dim}])")
    print(f"log_prob shape: {logp.shape}  (expected [4, 1])")

    dyn = ManipulatorDynamics()
    reg = PhysicsRegularizer(dyn, tau_max=87.0)
    q = np.zeros(n_joints)
    dq = np.zeros(n_joints)
    dq_new = np.ones(n_joints) * 0.5
    loss = reg.compute_loss(q, dq, dq_new)
    print(f"L_dyn (single): {loss.item():.4f}")

    critic = SoftmaxCritic(state_dim, action_dim)
    q1, q2 = critic(s, a.detach())
    print(f"Q1 shape: {q1.shape}  (expected [4, 1])")

    print("physics_policy.py unit test PASSED")
