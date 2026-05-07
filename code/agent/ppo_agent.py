"""
ppo_agent.py
------------
Proximal Policy Optimization (PPO) agent with physics-informed policy regularization.

Shares the same PhysicsInformedActor and PhysicsRegularizer as SAC,
enabling fair comparison in ablation studies.

Key differences from SACAgent:
  - On-policy: uses RolloutBuffer (not ReplayBuffer)
  - Value network instead of double-Q critic
  - GAE for advantage estimation
  - Clipped surrogate objective for policy update
  - Multi-epoch mini-batch updates per rollout

Reference:
    Schulman et al., "Proximal Policy Optimization Algorithms", 2017
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent.physics_policy import PhysicsInformedActor, PhysicsRegularizer
from agent.value_network import ValueNetwork
from utils.normalizer import RunningMeanStd
from utils.rollout_buffer import RolloutBuffer


class PPOAgent:
    """
    PPO agent with physics-informed actor loss.

    Interface compatible with SACAgent (select_action, update, save, load)
    for drop-in use in the training pipeline.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        dynamics,
        n_envs: int = 1,
        rollout_steps: int = 200,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        lambda_dyn: float = 0.1,
        action_scale: float = 0.3,
        hidden_dims: tuple = (256, 256),
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.lambda_dyn = lambda_dyn
        self.device = torch.device(device)

        # Networks
        self.actor = PhysicsInformedActor(
            state_dim, action_dim, list(hidden_dims), action_scale
        ).to(self.device)
        self.value = ValueNetwork(state_dim, list(hidden_dims)).to(self.device)

        # Optimizers with separate learning rates if needed
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.value_opt = optim.Adam(self.value.parameters(), lr=lr)

        # Differentiable physics regularizer (Plan B: pure torch, preserves grad)
        self.physics = PhysicsRegularizer(
            dynamics, lambda_dyn=lambda_dyn, dt=0.02, device=self.device
        )

        # On-policy rollout buffer
        self.buffer = RolloutBuffer(
            n_envs=n_envs,
            rollout_steps=rollout_steps,
            state_dim=state_dim,
            action_dim=action_dim,
            joints=dynamics.n,
            gae_lambda=gae_lambda,
            gamma=gamma,
        )

        # Observation normalization
        self.obs_normalizer = RunningMeanStd(shape=(state_dim,))

    # ------------------------------------------------------------------
    # Action selection (evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Returns action only (for evaluation / validation).

        Parameters
        ----------
        state       : raw observation
        deterministic: if True, returns mean action without exploration noise
        """
        s = self.obs_normalizer.normalize(state)
        s_t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        _, _, mean = self.actor.sample(s_t)
        if deterministic:
            return mean.squeeze(0).cpu().numpy()
        action, _, _ = self.actor.sample(s_t)
        return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, state: np.ndarray):
        """
        Get action, log_prob, and value for a single state (training mode).

        Returns
        -------
        action   : np.ndarray (action_dim,)
        log_prob : float
        value    : float
        """
        s = self.obs_normalizer.normalize(state)
        s_t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        action, log_prob, _ = self.actor.sample(s_t)
        value = self.value(s_t)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    @torch.no_grad()
    def get_value(self, state: np.ndarray) -> np.ndarray:
        """Batched value prediction for advantage bootstrapping."""
        s = self.obs_normalizer.normalize(state)
        s_t = torch.FloatTensor(s).to(self.device)
        return self.value(s_t).cpu().numpy()

    def update(self) -> dict:
        """
        Full PPO update: multi-epoch mini-batch from rollout buffer.

        Returns dict with average loss values for logging.
        """
        losses = {
            "actor_rl_loss": 0.0,
            "critic_loss": 0.0,
            "physics_loss": 0.0,
            "actor_loss": 0.0,
            "alpha": 0.0,
        }
        n_updates = 0

        for _ in range(self.ppo_epochs):
            num_batches = max(1, self.buffer.__len__() // self.batch_size)
            for _ in range(num_batches):
                batch = self.buffer.sample(self.batch_size)
                if not batch:
                    continue

                # --- Convert to tensors ---
                # Normalize observations
                s_np = self.obs_normalizer.normalize(batch["state"])
                s_t = torch.FloatTensor(s_np).to(self.device)
                actions = torch.FloatTensor(batch["action"]).to(self.device)
                old_log_probs = torch.FloatTensor(batch["old_log_prob"]).to(self.device)
                adv = torch.FloatTensor(batch["advantages"]).to(self.device)
                ret = torch.FloatTensor(batch["returns"]).to(self.device)

                # Normalize advantages (per mini-batch)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # --- Actor forward ---
                mean, log_std = self.actor(s_t)
                std = log_std.exp()
                if torch.isnan(mean).any() or torch.isnan(std).any():
                    mean = torch.nan_to_num(mean, nan=0.0)
                    std = torch.nan_to_num(std, nan=1.0).clamp(min=1e-6)
                dist = torch.distributions.Normal(mean, std)

                # Log probability of stored actions (inverse tanh transform)
                scale = torch.ones_like(mean)
                scale[:, : self.actor.task_dim] = self.actor.task_scale
                scale[:, self.actor.task_dim :] = self.actor.nullspace_scale
                clamped = torch.clamp(actions / scale, -0.999, 0.999)
                x = 0.5 * (torch.log(1 + clamped) - torch.log(1 - clamped))

                log_prob = dist.log_prob(x) - torch.log(
                    scale * (1 - clamped.pow(2)) + 1e-6
                )
                log_prob = log_prob.sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)

                # --- PPO clipped surrogate ---
                ratio = (log_prob - old_log_probs).exp()
                surr1 = ratio * adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * adv
                )
                actor_rl_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -self.entropy_coef * entropy.mean()
                policy_loss = actor_rl_loss + entropy_loss

                # --- Value loss ---
                values_pred = self.value(s_t)
                value_loss = 0.5 * ((values_pred - ret) ** 2).mean()

                # --- Physics regularization ---
                # Use mean action for deterministic gradient flow
                mean_action = torch.tanh(mean) * scale
                physics_loss = self.physics.compute_loss_batch(
                    q_batch=torch.FloatTensor(batch["q"]).to(self.device),
                    dq_batch=torch.FloatTensor(batch["dq"]).to(self.device),
                    J_batch=torch.FloatTensor(batch["J"]).to(self.device),
                    sigma_batch=torch.FloatTensor(batch["sigma"]).to(self.device),
                    dx_nom_batch=torch.FloatTensor(batch["dx_nom"]).to(self.device),
                    action_batch=mean_action,
                )
                if torch.isnan(physics_loss) or torch.isinf(physics_loss):
                    physics_loss = torch.tensor(0.0, device=self.device)

                # --- Total loss ---
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.lambda_dyn * physics_loss
                )

                # --- Gradient step ---
                self.actor_opt.zero_grad()
                self.value_opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
                self.actor_opt.step()
                self.value_opt.step()

                # Accumulate logging stats
                losses["actor_rl_loss"] += actor_rl_loss.item()
                losses["critic_loss"] += value_loss.item()
                losses["physics_loss"] += physics_loss.item()
                losses["actor_loss"] += total_loss.item()
                n_updates += 1

        # Average across all updates
        if n_updates > 0:
            for k in losses:
                losses[k] /= n_updates

        return losses

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str, metadata: dict = None):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "value": self.value.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "value_opt": self.value_opt.state_dict(),
                "obs_normalizer": self.obs_normalizer.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path: str, load_optimizers: bool = True) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.value.load_state_dict(ckpt["value"])
        if load_optimizers:
            if "actor_opt" in ckpt:
                self.actor_opt.load_state_dict(ckpt["actor_opt"])
                self.value_opt.load_state_dict(ckpt["value_opt"])
        if "obs_normalizer" in ckpt:
            self.obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        return ckpt.get("metadata", {})


# ------------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from env.dynamics import ManipulatorDynamics

    n_joints = 7
    state_dim = n_joints * 2 + 3 + 3 + 3 + 1 + 1 + 3  # 28
    action_dim = 3 + (n_joints - 3)  # 7 = task(3) + nullspace(4)

    print("=== ppo_agent.py unit tests ===")

    dyn = ManipulatorDynamics()
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        dynamics=dyn,
        n_envs=2,
        rollout_steps=8,
        lr=3e-4,
        ppo_epochs=3,
        batch_size=4,
    )

    # Test select_action (evaluation)
    s = np.random.randn(state_dim).astype(np.float32)
    a = agent.select_action(s, deterministic=True)
    print(f"select_action deterministic: shape={a.shape}  (expected ({action_dim},))")

    # Test act (training)
    a2, lp, v = agent.act(s)
    print(f"act: action={a2.shape}, log_prob={lp:.4f}, value={v:.4f}")

    # Test buffer + update
    for step in range(8):
        states_row = np.random.randn(2, state_dim).astype(np.float32)
        actions_row = np.random.randn(2, action_dim).astype(np.float32)
        rewards_row = np.random.randn(2).astype(np.float32)
        dones_row = np.zeros(2, dtype=np.float32)
        log_probs_row = np.random.randn(2).astype(np.float32)
        values_row = np.random.randn(2).astype(np.float32)

        agent.buffer.push(
            states_row,
            actions_row,
            rewards_row,
            dones_row,
            log_probs_row,
            values_row,
            q=np.random.randn(2, n_joints).astype(np.float32),
            dq=np.random.randn(2, n_joints).astype(np.float32),
            dq_next=np.random.randn(2, n_joints).astype(np.float32),
            J=np.random.randn(2, 3, n_joints).astype(np.float32),
            sigma=np.random.rand(2, 1).astype(np.float32),
            dx_nom=np.random.randn(2, 3).astype(np.float32),
        )

    last_values = np.random.randn(2, 1).astype(np.float32)
    agent.buffer.compute_advantages(last_values)
    print(
        f"buffer size: {len(agent.buffer)}  (expected 16)"
    )

    losses = agent.update()
    print(
        f"update: actor_rl_loss={losses['actor_rl_loss']:.6f}, "
        f"critic_loss={losses['critic_loss']:.6f}, "
        f"physics_loss={losses['physics_loss']:.6f}"
    )
    assert losses["actor_rl_loss"] > 0, "actor_rl_loss should be > 0"
    print("ppo_agent.py unit test PASSED")
