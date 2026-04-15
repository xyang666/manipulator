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
                 lr:           float = 3e-4,
                 gamma:        float = 0.99,
                 tau:          float = 0.005,
                 alpha:        float = 0.2,
                 lambda_dyn:   float = 0.1,
                 lambda_collision: float = 1.0,
                 tau_max:      float = 87.0,
                 dt:           float = 0.02,
                 action_scale: float = 0.5,
                 hidden_dims:  tuple = (256, 256),
                 device:       str   = "cpu",
                 collision_detector = None):
        self.gamma       = gamma
        self.tau         = tau
        self.alpha       = alpha
        self.lambda_dyn  = lambda_dyn
        self.lambda_collision = lambda_collision
        self.device      = torch.device(device)
        self.collision_detector = collision_detector

        # Networks
        self.actor   = PhysicsInformedActor(state_dim, action_dim,
                                            list(hidden_dims), action_scale).to(self.device)
        self.critic  = SoftmaxCritic(state_dim, action_dim, list(hidden_dims)).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # Physics regularizer with collision detection
        self.physics = PhysicsRegularizer(dynamics, tau_max=tau_max,
                                          lambda_dyn=lambda_dyn,
                                          lambda_collision=lambda_collision,
                                          dt=dt,
                                          collision_detector=collision_detector)

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

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
<<<<<<< HEAD
=======
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
>>>>>>> 1363fe7e1c704579a1bc953fb59aa96a9c819dea
        self.critic_opt.step()

        # -------- Actor update (with physics loss) --------
        a_new, log_pi, _ = self.actor.sample(s)
        q_min = self.critic.q_min(s, a_new)

        actor_rl_loss = (self.alpha * log_pi - q_min).mean()

        # Physics regularization
        q_np   = batch["q"]
        dq_np  = batch["dq"]
        dq_new_np = batch["dq_next"]

        q_t  = torch.FloatTensor(q_np).to(self.device)
        dq_t = torch.FloatTensor(dq_np).to(self.device)
        dq_new_t = torch.FloatTensor(dq_new_np).to(self.device)

        physics_loss = self.physics.compute_loss_batch(q_t, dq_t, dq_new_t,
                                                       collision_detector=self.collision_detector)

        actor_loss = actor_rl_loss + physics_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
<<<<<<< HEAD
=======
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
>>>>>>> 1363fe7e1c704579a1bc953fb59aa96a9c819dea
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

<<<<<<< HEAD
    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
=======
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
>>>>>>> 1363fe7e1c704579a1bc953fb59aa96a9c819dea
