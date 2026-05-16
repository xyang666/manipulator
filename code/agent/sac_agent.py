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
from utils.normalizer import RunningMeanStd


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
                 target_entropy: float | None = None,
                 lambda_dyn:   float = 0.1,
                 task_scale:   float = 1.0,
                 nullspace_scale: float = 0.5,
                 hidden_dims:  tuple = (256, 256),
                 device:       str   = "cpu",
                 critic_warmup: int = 5000,
                 total_steps:  int   = 0,
                 n_critics:    int   = 2):
        self.gamma       = gamma
        self.tau         = tau
        self.alpha       = alpha
        self.lambda_dyn  = lambda_dyn
        self.device      = torch.device(device)
        self.critic_warmup = critic_warmup
        self._update_count = 0

        # Networks
        self.actor   = PhysicsInformedActor(state_dim, action_dim,
                                            list(hidden_dims),
                                            task_scale=task_scale,
                                            nullspace_scale=nullspace_scale).to(self.device)
        self.critic  = SoftmaxCritic(state_dim, action_dim, list(hidden_dims),
                                     n_critics=n_critics).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # Differentiable physics regularizer (Plan B: pure torch, preserves grad)
        self.physics = PhysicsRegularizer(dynamics, lambda_dyn=lambda_dyn,
                                          dt=self._get_dt_default(),
                                          device=self.device)

        # Automatic entropy tuning
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim
        self.min_alpha = 0.02  # prevent entropy collapse
        initial_log_alpha = max(np.log(alpha), np.log(self.min_alpha))
        self.log_alpha = torch.tensor(initial_log_alpha, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

        # Observation normalization
        self.obs_normalizer = RunningMeanStd(shape=(state_dim,))

        # Cosine learning rate annealing
        if total_steps > 0:
            self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.actor_opt, T_max=total_steps, eta_min=lr * 0.1)
            self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.critic_opt, T_max=total_steps, eta_min=lr * 0.1)
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None

    def _get_dt_default(self):
        """Get simulation timestep (matches env default)."""
        return 0.02

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = self.obs_normalizer(state)  # updates running stats + normalizes
        s = torch.FloatTensor(s).unsqueeze(0).to(self.device)
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

        Supports ensemble critic (N Q-networks) and prioritized replay weights.

        Returns dict with loss values for logging. If batch contains PER indices,
        caller should pass them to update_priorities().
        """
        # Normalize observations in batch
        s  = self.obs_normalizer.normalize(batch["state"])
        s_ = self.obs_normalizer.normalize(batch["next_state"])
        s  = torch.FloatTensor(s).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        a  = torch.FloatTensor(batch["action"]).to(self.device)
        r  = torch.FloatTensor(batch["reward"]).to(self.device)
        d  = torch.FloatTensor(batch["done"]).to(self.device)
        is_weights = torch.FloatTensor(batch.get("weights", np.ones(len(r)))).to(self.device)

        # -------- Critic update (ensemble of N Q-networks) --------
        with torch.no_grad():
            a_, log_pi_, _ = self.actor.sample(s_)
            q_targets = self.critic_target(s_, a_)  # tuple of N
            q_target = torch.min(torch.cat(q_targets, dim=-1), dim=-1, keepdim=True).values
            q_backup = r + self.gamma * (1 - d) * (q_target - self.alpha * log_pi_)

        q_values = self.critic(s, a)  # tuple of N
        critic_loss = 0.0
        td_errors = []
        for q in q_values:
            per_sample_loss = F.mse_loss(q, q_backup, reduction='none')
            critic_loss += (per_sample_loss * is_weights.unsqueeze(-1)).mean()
            td_errors.append((q - q_backup).abs())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        # Average TD-errors across N critics for PER priority update
        td_error_avg = torch.stack(td_errors, dim=-1).mean(dim=-1).detach().cpu().numpy().flatten()

        self._update_count += 1
        doing_warmup = self._update_count < self.critic_warmup

        if not doing_warmup:
            # -------- Actor update (with differentiable physics loss) --------
            a_new, log_pi, _ = self.actor.sample(s)
            q_min = self.critic.q_min(s, a_new)

            actor_rl_loss = (self.alpha * log_pi - q_min).mean()

            # Differentiable physics regularization (Plan B)
            q_t  = torch.FloatTensor(batch["q"]).to(self.device)
            dq_t = torch.FloatTensor(batch["dq"]).to(self.device)
            J_t  = torch.FloatTensor(batch["J"]).to(self.device)
            sigma_t = torch.FloatTensor(batch["sigma"]).to(self.device)
            dx_nom_t = torch.FloatTensor(batch["dx_nom"]).to(self.device)

            physics_loss = self.physics.compute_loss_batch(
                q_batch=q_t, dq_batch=dq_t,
                J_batch=J_t, sigma_batch=sigma_t, dx_nom_batch=dx_nom_t,
                action_batch=a_new,
            )

            if torch.isnan(physics_loss) or torch.isinf(physics_loss):
                physics_loss = torch.tensor(0.0, device=self.device)

            actor_loss = actor_rl_loss + physics_loss

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()
            if self.actor_scheduler is not None:
                self.actor_scheduler.step()

            # -------- Alpha (entropy) update --------
            with torch.no_grad():
                _, log_pi_new, _ = self.actor.sample(s)
            alpha = self.log_alpha.exp()
            alpha_loss = -(alpha * (log_pi_new + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.log_alpha.data.clamp_(min=np.log(self.min_alpha))
            self.alpha = self.log_alpha.exp().item()

        # -------- Soft update target critic --------
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

        return {
            "critic_loss":  critic_loss.item(),
            "actor_rl_loss": actor_rl_loss.item() if not doing_warmup else 0.0,
            "physics_loss": physics_loss.item() if not doing_warmup else 0.0,
            "actor_loss":   actor_loss.item() if not doing_warmup else 0.0,
            "alpha":        self.alpha,
            "td_error":     float(td_error_avg.mean()),
        }, td_error_avg

    def save(self, path: str, metadata: dict = None):
        state = {
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "actor_opt":      self.actor_opt.state_dict(),
            "critic_opt":     self.critic_opt.state_dict(),
            "alpha_opt":      self.alpha_opt.state_dict(),
            "log_alpha":      self.log_alpha.item(),
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "metadata":       metadata or {},
        }
        if self.actor_scheduler is not None:
            state["actor_scheduler"] = self.actor_scheduler.state_dict()
        if self.critic_scheduler is not None:
            state["critic_scheduler"] = self.critic_scheduler.state_dict()
        torch.save(state, path)

    def load(self, path: str, load_optimizers: bool = True, reset_alpha: bool = False,
             reset_critic: bool = False, reset_actor: bool = False,
             lr: float | None = None) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if not reset_actor:
            self.actor.load_state_dict(ckpt["actor"])
        if reset_critic:
            # Don't load critic weights — use fresh LayerNorm init
            # (old checkpoint doesn't have LayerNorm params)
            # Use provided lr, fall back to actor_opt's current lr, last resort 3e-4
            _lr = lr if lr is not None else self.actor_opt.param_groups[0]['lr']
            self.critic_opt = optim.Adam(self.critic.parameters(), lr=_lr)
        else:
            self.critic.load_state_dict(ckpt["critic"], strict=False)
        if load_optimizers:
            if "actor_opt" in ckpt and not reset_actor:
                self.actor_opt.load_state_dict(ckpt["actor_opt"])
                if self.actor_scheduler is not None and "actor_scheduler" in ckpt:
                    self.actor_scheduler.load_state_dict(ckpt["actor_scheduler"])
            if "critic_opt" in ckpt and not reset_critic:
                self.critic_opt.load_state_dict(ckpt["critic_opt"])
                if self.critic_scheduler is not None and "critic_scheduler" in ckpt:
                    self.critic_scheduler.load_state_dict(ckpt["critic_scheduler"])
            if not reset_alpha:
                self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
                self.log_alpha.data.fill_(ckpt["log_alpha"])
                self.alpha = self.log_alpha.exp().item()
        if "obs_normalizer" in ckpt:
            self.obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        return ckpt.get("metadata", {})
