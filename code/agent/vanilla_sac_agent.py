"""
vanilla_sac_agent.py
--------------------
Standard Soft Actor-Critic (SAC) agent with simple MLP actor/critic.
No physics-informed decomposition, no task/nullspace action split,
no physics regularization.

This serves as a baseline comparison for the physics-informed SAC agent.

Reference:
    Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2018
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from utils.normalizer import RunningMeanStd

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class VanillaActor(nn.Module):
    """
    Simple MLP Gaussian actor outputting 7D tanh-squashed joint velocities.
    All dimensions share the same action_scale (uniform scaling).

    Architecture: [state_dim -> 256 -> Tanh -> 256 -> Tanh -> action_dim * 2]
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list[int] = (256, 256),
                 action_scale: float = 2.175):
        super().__init__()
        self.action_scale = action_scale

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h

        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

    def forward(self, state: torch.Tensor):
        h = self.net(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        """
        Returns
        -------
        action   : tanh-squashed, scaled by action_scale
        log_prob : [batch x 1]
        mean     : deterministic action (tanh(mean) * scale)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if torch.isnan(mean).any() or torch.isnan(std).any():
            mean = torch.nan_to_num(mean, nan=0.0)
            std = torch.nan_to_num(std, nan=1.0).clamp(min=1e-6)

        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale

        # log_prob with tanh change-of-variables
        log_prob = dist.log_prob(x) - torch.log(
            self.action_scale * (1 - y.pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale
        return action, log_prob, mean_action


class VanillaCritic(nn.Module):
    """
    Standard double-Q network for SAC.
    No LayerNorm — standard MLP + ReLU.
    """

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


class VanillaSACAgent:
    """
    Standard Soft Actor-Critic agent.
    No physics-informed components; simple MLP actor + double-Q critic.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 action_scale: float = 2.175,
                 hidden_dims: tuple = (256, 256),
                 device: str = "cpu",
                 critic_warmup: int = 5000,
                 total_steps: int = 0,
                 n_critics: int = 2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device(device)
        self.critic_warmup = critic_warmup
        self._update_count = 0

        # Networks
        self.actor = VanillaActor(state_dim, action_dim,
                                  list(hidden_dims), action_scale).to(self.device)
        self.critic = VanillaCritic(state_dim, action_dim,
                                    list(hidden_dims)).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.min_alpha = 0.02
        initial_log_alpha = max(np.log(alpha), np.log(self.min_alpha))
        self.log_alpha = torch.tensor(initial_log_alpha, requires_grad=True,
                                       device=self.device)
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

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        s = self.obs_normalizer.normalize(state)
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

        Standard SAC loss: J_Q = MSE(Q(s,a), r + gamma * (V)(s'))
                            J_pi = alpha * log_pi - Q_min(s, a)

        Returns dict with loss values for logging.
        """
        # Normalize observations
        s = self.obs_normalizer.normalize(batch["state"])
        s_ = self.obs_normalizer.normalize(batch["next_state"])
        s = torch.FloatTensor(s).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        a = torch.FloatTensor(batch["action"]).to(self.device)
        r = torch.FloatTensor(batch["reward"]).to(self.device).view(-1, 1)
        d = torch.FloatTensor(batch["done"]).to(self.device).view(-1, 1)

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

        self._update_count += 1
        doing_warmup = self._update_count < self.critic_warmup

        if not doing_warmup:
            # -------- Actor update (no physics loss) --------
            a_new, log_pi, _ = self.actor.sample(s)
            q_min = self.critic.q_min(s, a_new)
            actor_rl_loss = (self.alpha * log_pi - q_min).mean()
            actor_loss = actor_rl_loss

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()

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

        if self.actor_scheduler is not None and not doing_warmup:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_rl_loss": actor_rl_loss.item() if not doing_warmup else 0.0,
            "actor_loss": actor_loss.item() if not doing_warmup else 0.0,
            "alpha": self.alpha,
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str, metadata: dict = None):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "metadata": metadata or {},
        }, path)

    def load(self, path: str, load_optimizers: bool = True,
             reset_alpha: bool = False) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if load_optimizers:
            if "actor_opt" in ckpt:
                self.actor_opt.load_state_dict(ckpt["actor_opt"])
            if "critic_opt" in ckpt:
                self.critic_opt.load_state_dict(ckpt["critic_opt"])
            if not reset_alpha:
                self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
                self.log_alpha.data.fill_(ckpt["log_alpha"])
                self.alpha = self.log_alpha.exp().item()
        if "obs_normalizer" in ckpt:
            self.obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        return ckpt.get("metadata", {})


# ------------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    n_joints = 7
    state_dim = 28
    action_dim = 7

    print("=== vanilla_sac_agent.py unit tests ===")

    # Test networks
    actor = VanillaActor(state_dim, action_dim)
    s = torch.randn(4, state_dim)
    a, logp, a_det = actor.sample(s)
    print(f"actor output:  action={list(a.shape)}  (expected [4, 7])")
    print(f"               log_prob={list(logp.shape)}  (expected [4, 1])")

    critic = VanillaCritic(state_dim, action_dim)
    q1, q2 = critic(s, a.detach())
    print(f"critic output: q1={list(q1.shape)}, q2={list(q2.shape)}  (expected [4, 1])")

    # Test agent
    agent = VanillaSACAgent(
        state_dim=state_dim, action_dim=action_dim,
        critic_warmup=10, total_steps=1000,
    )

    # Test action selection
    s_np = np.random.randn(state_dim).astype(np.float32)
    a_np = agent.select_action(s_np, deterministic=True)
    print(f"select_action: shape={list(a_np.shape)}  (expected [{action_dim}])")

    # Test update with dummy batch
    batch = {
        "state":     np.random.randn(32, state_dim).astype(np.float32),
        "next_state": np.random.randn(32, state_dim).astype(np.float32),
        "action":    np.random.randn(32, action_dim).astype(np.float32),
        "reward":    np.random.randn(32).astype(np.float32),
        "done":      np.zeros(32, dtype=np.float32),
    }
    losses = agent.update(batch)
    print(f"update: critic_loss={losses['critic_loss']:.6f}, "
          f"actor_loss={losses['actor_rl_loss']:.6f}, "
          f"alpha={losses['alpha']:.4f}")

    # Test save/load
    agent.save("/tmp/test_vanilla_sac.pt")
    agent2 = VanillaSACAgent(
        state_dim=state_dim, action_dim=action_dim,
        critic_warmup=10,
    )
    agent2.load("/tmp/test_vanilla_sac.pt")
    os.remove("/tmp/test_vanilla_sac.pt")
    print("save/load: OK")

    print("vanilla_sac_agent.py unit test PASSED")
