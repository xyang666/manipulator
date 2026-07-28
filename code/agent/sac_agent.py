"""
sac_agent.py
------------
Soft Actor-Critic (SAC) agent with physics-informed policy regularization.

Key modification over standard SAC:
    actor_loss = -Q_min(s, a) + alpha * log_pi(a|s) + lambda_dyn * L_dyn

where L_dyn penalizes torques that exceed joint limits.

Author: xie yang
Date:   2025-06

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


def scaled_sigmoid_inverse(value: float, maximum: float) -> float:
    """Raw parameter whose scaled sigmoid is exactly ``value``."""
    if not 0.0 < value < maximum:
        raise ValueError("value must be strictly between zero and maximum")
    ratio = value / maximum
    return float(np.log(ratio / (1.0 - ratio)))


def normalized_discounted_cost(q_cost, gamma: float):
    """Convert discounted cumulative cost to an average per-step scale."""
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    return (1.0 - gamma) * q_cost


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
                 n_critics:    int   = 2,
                 cost_limit:   float = 0.05,
                 cost_scale:   float = 1.0,
                 backbone:     str   = "mlp",
                 frame_stack:  int   = 1,
                 action_horizon: int = 1,
                 single_dim:   int   = None,
                 d_model:      int   = 128,
                 n_heads:      int   = 4,
                 n_enc_layers: int   = 2,
                 n_dec_layers: int   = 2,
                 dropout:      float = 0.1,
                 grad_steps:   int   = 1,
                 lr_lag:       float = 1e-4,
                 lag_init:     float = 0.1,
                 lag_max:      float = 10.0,
                 use_safety_critic: bool = True):
        self.gamma       = gamma
        self.use_safety_critic = use_safety_critic
        self.tau         = tau
        self.alpha       = alpha
        self.lambda_dyn  = lambda_dyn
        self.device      = torch.device(device)
        self.critic_warmup = critic_warmup
        self._update_count = 0
        self.cost_limit  = cost_limit
        self.cost_scale  = cost_scale
        self._frame_stack = frame_stack
        self._single_dim = single_dim if single_dim is not None else state_dim
        self._backbone = backbone
        self._grad_steps = max(1, grad_steps)

        # Scale tau by 1/grad_steps so the effective Polyak averaging rate is
        # τ per env step (not per gradient step). Otherwise when grad_steps > 1,
        # the target network tracks the online network too closely, collapsing
        # the stabilizing lag between Q(s,a) and the TD target and triggering
        # bootstrap divergence (Q-value explosion to infinity).
        #   effective movement per env step ≈ 1 - (1 - τ/grad_steps)^grad_steps ≈ τ
        self.tau = tau / max(1, grad_steps)

        # Actor: may take stacked frames (transformer) or single frame (mlp)
        actor_state_dim = state_dim

        # Critic: always single-frame input
        critic_state_dim = self._single_dim

        # Networks
        self.actor   = PhysicsInformedActor(actor_state_dim, action_dim,
                                            list(hidden_dims),
                                            task_scale=task_scale,
                                            nullspace_scale=nullspace_scale,
                                            backbone=backbone,
                                            frame_stack=frame_stack,
                                            action_horizon=action_horizon,
                                            d_model=d_model,
                                            n_heads=n_heads,
                                            n_enc_layers=n_enc_layers,
                                            n_dec_layers=n_dec_layers,
                                            dropout=dropout).to(self.device)
        self.critic  = SoftmaxCritic(critic_state_dim, action_dim, list(hidden_dims),
                                     n_critics=n_critics).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # Safety Critic (optional — disable via use_safety_critic=False for standard SAC)
        self.safety_critic = None
        self.safety_critic_target = None
        self.safety_critic_opt = None
        self._lag_raw = None
        self.lag_opt = None
        if lag_max <= 0.0:
            raise ValueError("lag_max must be positive")
        self._lambda_max = float(lag_max)
        if self.use_safety_critic:
            self.safety_critic = SoftmaxCritic(critic_state_dim, action_dim, list(hidden_dims),
                                               n_critics=n_critics).to(self.device)
            self.safety_critic_target = copy.deepcopy(self.safety_critic).to(self.device)
            self.safety_critic_opt = optim.Adam(self.safety_critic.parameters(), lr=lr)
            # Lagrange multiplier — sigmoid param avoids dead λ
            lag_raw = scaled_sigmoid_inverse(lag_init, self._lambda_max)
            self._lag_raw = torch.tensor(lag_raw, device=self.device, requires_grad=True)
            self.lag_opt = optim.Adam([self._lag_raw], lr=lr_lag)

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
        # NOTE: scheduler.step() is called inside agent.update() for every gradient update.
        # When grad_steps > 1 (gradient updates per env step), T_max must be multiplied
        # by grad_steps so the LR anneals over the intended number of env steps, not
        # gradient steps. Otherwise the LR decays to eta_min in only 1/grad_steps of
        # the training duration, stranding the critic at low LR and enabling divergence.
        if total_steps > 0:
            _grad_mult = max(1, grad_steps)
            self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.actor_opt, T_max=total_steps * _grad_mult, eta_min=lr * 0.1)
            self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.critic_opt, T_max=total_steps * _grad_mult, eta_min=lr * 0.1)
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
        s = self.obs_normalizer(state)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, mean = self.actor.sample(s)
        if deterministic:
            return mean.squeeze(0).cpu().numpy()
        return action.squeeze(0).cpu().numpy()

    def select_action_batch(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Batch action selection. states: (batch, dim) -> actions: (batch, act_dim)."""
        s = torch.as_tensor(self.obs_normalizer(states), dtype=torch.float32, device=self.device)
        action, _, mean = self.actor.sample(s)
        if deterministic:
            return mean.detach().cpu().numpy()
        return action.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _to_tensor(self, x, dtype=torch.float32):
        """Convert numpy/JAX → torch tensor on device. No-op if already tensor."""
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=self.device)
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    @staticmethod
    def _to_numpy(x):
        """Convert to numpy (for normalizer)."""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    def update(self, batch: dict, batch_size: int = 256, is_last: bool = False,
               actor_enabled: bool | None = None):
        """
        One gradient update step from a sampled batch.

        Accepts both numpy arrays and GPU torch tensors.

        Parameters
        ----------
        batch : dict from replay buffer
        batch_size : int (unused, kept for compat)
        is_last : bool
            If True (last of grad_steps iterations), also update α (entropy
            temperature) and λ (Lagrange multiplier).  These are single-scalar
            optimisers that should step once per env step, not once per
            gradient step — otherwise they overshoot when grad_steps > 1.

        Supports ensemble critic (N Q-networks) and prioritized replay weights.

        Returns dict with loss values for logging. If batch contains PER indices,
        caller should pass them to update_priorities().
        """
        # Normalize observations in batch (expects numpy)
        s_full  = self.obs_normalizer.normalize(self._to_numpy(batch["state"]))
        s_full_ = self.obs_normalizer.normalize(self._to_numpy(batch["next_state"]))

        # For transformer backbone: actor takes stacked frames, critic takes current frame only
        if self._frame_stack > 1:
            s_critic  = s_full[:, -self._single_dim:]
            s_critic_ = s_full_[:, -self._single_dim:]
        else:
            s_critic  = s_full
            s_critic_ = s_full_

        s  = self._to_tensor(s_full)
        s_ = self._to_tensor(s_full_)
        s_critic  = self._to_tensor(s_critic)
        s_critic_ = self._to_tensor(s_critic_)
        a  = self._to_tensor(batch["action"])
        r  = self._to_tensor(batch["reward"])
        d  = self._to_tensor(batch["done"])
        is_weights = self._to_tensor(np.ones(len(r)))
        if "weights" in batch:
            is_weights = self._to_tensor(batch["weights"])
        cost = self._to_tensor(batch["cost"])

        # -------- Critic update (ensemble of N Q-networks) --------
        with torch.no_grad():
            a_, log_pi_, _ = self.actor.sample(s_)
            q_targets = self.critic_target(s_critic_, a_)  # tuple of N
            q_target = torch.min(torch.cat(q_targets, dim=-1), dim=-1, keepdim=True).values
            # Clip the bootstrap term only (γ·Q_target), not the reward r.
            # This preserves terminal-bonus learning (q_backup=r when done=1)
            # while preventing bootstrap-diffused Q inflation from the large
            # success_bonus (=500) propagating backward through γ·Q.
            boot = self.gamma * (1 - d) * (q_target - self.alpha * log_pi_)
            q_backup = r + boot.clamp(-100.0, 100.0)

        q_values = self.critic(s_critic, a)  # tuple of N
        critic_loss = 0.0
        td_errors = []
        for q in q_values:
            per_sample_loss = F.huber_loss(q, q_backup, reduction='none', delta=100.0)
            # Clamp per-sample loss to prevent any single OOD sample (e.g. a
            # terminal transition with success_bonus=500) from dominating the
            # gradient.  Without this, an episode where Q-error = 500 produces
            # per-sample loss ≈ 45000, which at batch_size=512 averages to ~88
            # per critic → ×5 ensemble ≈ 440 — enough to swing the critic
            # weights significantly away from the non-terminal distribution.
            per_sample_loss = per_sample_loss.clamp(max=5000.0)
            critic_loss += (per_sample_loss * is_weights.unsqueeze(-1)).mean()
            td_errors.append((q - q_backup).abs())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        # -------- Safety Critic update (optional) --------
        safety_critic_loss = 0.0
        if self.use_safety_critic:
            with torch.no_grad():
                a_, log_pi_, _ = self.actor.sample(s_)
                qc_targets = self.safety_critic_target(s_critic_, a_)
                qc_target = torch.min(torch.cat(qc_targets, dim=-1), dim=-1, keepdim=True).values
                qc_backup = cost * self.cost_scale + self.gamma * (1 - d) * qc_target
            for qc in self.safety_critic(s_critic, a):
                safety_critic_loss += F.mse_loss(qc, qc_backup)
            self.safety_critic_opt.zero_grad()
            safety_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.safety_critic.parameters(), max_norm=1.0)
            self.safety_critic_opt.step()

        # Average TD-errors across N critics for PER priority update
        td_error_avg = torch.stack(td_errors, dim=-1).mean(dim=-1).detach().cpu().numpy().flatten()

        self._update_count += 1
        doing_warmup = (self._update_count < self.critic_warmup
                        if actor_enabled is None else not actor_enabled)
        lag_loss = torch.tensor(0.0, device=self.device)

        if not doing_warmup:
            # -------- Actor update (with differentiable physics loss) --------
            a_new, log_pi, _ = self.actor.sample(s)
            q_min = self.critic.q_min(s_critic, a_new)

            # Standard SAC objective or constrained RL (with safety critic)
            if self.use_safety_critic:
                q_c_max = self.safety_critic.q_max(s_critic, a_new)
                normalized_q_c = normalized_discounted_cost(q_c_max, self.gamma)
                lag = self._lambda_max * torch.sigmoid(self._lag_raw)
                actor_rl_loss = (
                    self.alpha * log_pi - q_min + lag.detach() * normalized_q_c
                ).mean()
            else:
                actor_rl_loss = (self.alpha * log_pi - q_min).mean()

            # Differentiable physics regularization
            q_t  = self._to_tensor(batch["q"])
            dq_t = self._to_tensor(batch["dq"])
            J_t  = self._to_tensor(batch["J"])
            sigma_t = self._to_tensor(batch["sigma"])
            dx_nom_t = self._to_tensor(batch["dx_nom"])

            physics_loss = self.physics.compute_loss_batch(
                q_batch=q_t, dq_batch=dq_t,
                J_batch=J_t, sigma_batch=sigma_t, dx_nom_batch=dx_nom_t,
                action_batch=a_new,
            )

            if torch.isnan(physics_loss) or torch.isinf(physics_loss):
                physics_loss = torch.tensor(0.0, device=self.device)

            actor_loss = actor_rl_loss + physics_loss

            # Chunk smoothness regularization (transformer only)
            if self._backbone == "transformer" and self._frame_stack > 1:
                chunk_mean, _ = self.actor._transformer.forward(s)
                smooth_loss = 0.01 * ((chunk_mean[:, 1:] - chunk_mean[:, :-1]) ** 2).mean()
                actor_loss = actor_loss + smooth_loss

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()
            if self.actor_scheduler is not None:
                self.actor_scheduler.step()

            # -------- Alpha (entropy) update (once per env step) --------
            if is_last:
                with torch.no_grad():
                    _, log_pi_new, _ = self.actor.sample(s)
                alpha = self.log_alpha.exp()
                alpha_loss = -(alpha * (log_pi_new + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()
                self.log_alpha.data.clamp_(min=np.log(self.min_alpha))
                self.alpha = self.log_alpha.exp().item()
            else:
                alpha_loss = torch.tensor(0.0, device=self.device)

            # -------- Lagrange multiplier λ update (safety critic only) --------
            if self.use_safety_critic and is_last:
                with torch.no_grad():
                    q_c_max_detach = self.safety_critic.q_max(s_critic, a_new)
                    normalized_q_c = normalized_discounted_cost(
                        q_c_max_detach, self.gamma
                    )
                lag_loss = -lag * (
                    normalized_q_c - self.cost_limit * self.cost_scale
                ).detach().mean()
                self.lag_opt.zero_grad()
                lag_loss.backward()
                self.lag_opt.step()
            else:
                lag_loss = torch.tensor(0.0, device=self.device)

        # -------- Soft update target critics (task + safety) --------
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)
        if self.use_safety_critic:
            for p, p_t in zip(self.safety_critic.parameters(), self.safety_critic_target.parameters()):
                p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

        return {
            "critic_loss":  critic_loss.item(),
            "safety_critic_loss": safety_critic_loss.item() if isinstance(safety_critic_loss, torch.Tensor) else safety_critic_loss,
            "actor_rl_loss": actor_rl_loss.item() if not doing_warmup else 0.0,
            "physics_loss": physics_loss.item() if not doing_warmup else 0.0,
            "actor_loss":   actor_loss.item() if not doing_warmup else 0.0,
            "lag_loss":     lag_loss.item() if isinstance(lag_loss, torch.Tensor) else lag_loss,
            "alpha":        self.alpha,
            "lag":          (self._lambda_max * torch.sigmoid(self._lag_raw)).item() if self.use_safety_critic else 0.0,
            "td_error":     float(td_error_avg.mean()),
        }, td_error_avg

    def save(self, path: str, metadata: dict = None):
        state = {
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
            "actor_opt":      self.actor_opt.state_dict(),
            "critic_opt":     self.critic_opt.state_dict(),
            "alpha_opt":      self.alpha_opt.state_dict(),
            "log_alpha":      self.log_alpha.item(),
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "metadata":       metadata or {},
        }
        if self.use_safety_critic:
            state["safety_critic"] = self.safety_critic.state_dict()
            state["safety_critic_target"] = self.safety_critic_target.state_dict()
            state["safety_critic_opt"] = self.safety_critic_opt.state_dict()
            state["lag_raw"] = self._lag_raw.item()
            state["lag_opt"] = self.lag_opt.state_dict()
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
            if "critic_target" in ckpt:
                self.critic_target.load_state_dict(ckpt["critic_target"], strict=False)
            else:
                self.critic_target.load_state_dict(self.critic.state_dict())
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
        if self.use_safety_critic and "safety_critic" in ckpt:
            self.safety_critic.load_state_dict(ckpt["safety_critic"], strict=False)
            if "safety_critic_target" in ckpt:
                self.safety_critic_target.load_state_dict(ckpt["safety_critic_target"], strict=False)
            if load_optimizers and "safety_critic_opt" in ckpt:
                self.safety_critic_opt.load_state_dict(ckpt["safety_critic_opt"])
        if self.use_safety_critic and "lag_raw" in ckpt:
            self._lag_raw.data.fill_(ckpt["lag_raw"])
        elif self.use_safety_critic and "log_lag" in ckpt:
            self._lag_raw.data.fill_(np.exp(ckpt["log_lag"]))
        if self.use_safety_critic and load_optimizers and "lag_opt" in ckpt:
            self.lag_opt.load_state_dict(ckpt["lag_opt"])
        return ckpt.get("metadata", {})
