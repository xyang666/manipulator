"""
sac_agent.py
------------
Soft Actor-Critic (SAC) agent with physics-informed policy regularization.

Key modification over standard SAC:
    actor_loss = -Q_min(s, a) + alpha * log_pi(a|s) + lambda_dyn * L_dyn

where L_dyn penalizes torques that exceed joint limits.

References:
    Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2018
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy

from agent.physics_policy import PhysicsInformedActor, SoftmaxCritic, PhysicsRegularizer


class SACAgent:
    """
    SAC agent with physics-informed actor loss.
    """

    def __init__(self,
                 state_dim:    int,
                 action_dim:   int,
                 dynamics,
                 lr:           float = 1e-4,
                 gamma:        float = 0.99,
                 tau:          float = 0.005,
                 alpha:        float = 0.2,
                 lambda_dyn:   float = 0.1,
                 action_scale: float = 0.3,
                 hidden_dims:  tuple = (256, 256),
                 device:       str   = "cpu"):
        self.gamma       = gamma
        self.tau         = tau
        self.alpha       = alpha
        self.lambda_dyn  = lambda_dyn
        self.device      = torch.device(device)

        # Networks
        self.actor   = PhysicsInformedActor(state_dim, action_dim,
                                            list(hidden_dims), action_scale).to(self.device)
        self.critic  = SoftmaxCritic(state_dim, action_dim, list(hidden_dims)).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # Differentiable physics regularizer (Plan B: pure torch, preserves grad)
        self.physics = PhysicsRegularizer(dynamics, lambda_dyn=lambda_dyn,
                                          dt=self._get_dt_default(),
                                          device=self.device)

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

    def _get_dt_default(self):
        """Get simulation timestep (matches env default)."""
        return 0.02

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _, mean = self.actor.sample(s)
        if deterministic:
            return mean.squeeze(0).cpu().numpy()
        return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def update(self, batch: dict, batch_size: int = 256):
        """
        One gradient update step from a sampled batch.

        Returns dict with loss values for logging.
        """
        s  = torch.FloatTensor(batch["state"]).to(self.device)
        a  = torch.FloatTensor(batch["action"]).to(self.device)
        r  = torch.FloatTensor(batch["reward"]).to(self.device)
        s_ = torch.FloatTensor(batch["next_state"]).to(self.device)
        d  = torch.FloatTensor(batch["done"]).to(self.device)

        # -------- Critic update --------
        with torch.no_grad():
            a_, log_pi_, _ = self.actor.sample(s_)
            q1_t, q2_t = self.critic_target(s_, a_)
            q_target = torch.min(q1_t, q2_t) - self.alpha * log_pi_
            q_backup = r + self.gamma * (1 - d) * q_target

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # -------- Actor update (with differentiable physics loss) --------
        a_new, log_pi, _ = self.actor.sample(s)
        q_min = self.critic.q_min(s, a_new)

        actor_rl_loss = (self.alpha * log_pi - q_min).mean()

        # Differentiable physics regularization (Plan B)
        # Reconstruct dq_cmd from current-policy action analytically
        q_t  = torch.FloatTensor(batch["q"]).to(self.device)
        dq_t = torch.FloatTensor(batch["dq"]).to(self.device)
        J_t  = torch.FloatTensor(batch["J"]).to(self.device)
        sigma_t = torch.FloatTensor(batch["sigma"]).to(self.device)
        dx_nom_t = torch.FloatTensor(batch["dx_nom"]).to(self.device)

        physics_loss = self.physics.compute_loss_batch(
            q_batch=q_t, dq_batch=dq_t,
            J_batch=J_t, sigma_batch=sigma_t, dx_nom_batch=dx_nom_t,
            action_batch=a_new,  # current-policy action — has gradients!
        )

        # Safety check for NaN/Inf
        if torch.isnan(physics_loss) or torch.isinf(physics_loss):
            physics_loss = torch.tensor(0.0, device=self.device)

        actor_loss = actor_rl_loss + physics_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # -------- Alpha (entropy) update --------
        with torch.no_grad():
            _, log_pi_new, _ = self.actor.sample(s)
        alpha_loss = -(self.log_alpha * (log_pi_new + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # -------- Soft update target critic --------
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

        return {
            "critic_loss":  critic_loss.item(),
            "actor_rl_loss": actor_rl_loss.item(),
            "physics_loss": physics_loss.item(),
            "actor_loss":   actor_loss.item(),
            "alpha":        self.alpha,
        }

    def save(self, path: str, metadata: dict = None):
        torch.save({
            "actor":      self.actor.state_dict(),
            "critic":     self.critic.state_dict(),
            "actor_opt":  self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt":  self.alpha_opt.state_dict(),
            "log_alpha":  self.log_alpha.item(),
            "metadata":   metadata or {},
        }, path)

    def load(self, path: str, load_optimizers: bool = True) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if load_optimizers:
            if "actor_opt" in ckpt:
                self.actor_opt.load_state_dict(ckpt["actor_opt"])
                self.critic_opt.load_state_dict(ckpt["critic_opt"])
                self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
                self.log_alpha.data.fill_(ckpt["log_alpha"])
                self.alpha = self.log_alpha.exp().item()
        return ckpt.get("metadata", {})
