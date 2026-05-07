"""
rollout_buffer.py
-----------------
On-policy rollout buffer for PPO with Generalized Advantage Estimation (GAE).

Stores transitions in a 2D array [rollout_steps, n_envs, ...].
After collecting a full rollout, computes GAE advantages via compute_advantages(),
then samples mini-batches for multi-epoch PPO updates.
"""

import numpy as np


class RolloutBuffer:
    """
    Buffer for on-policy rollout collection.

    Transitions are stored as [step, env, ...] to keep per-environment
    trajectories separable for GAE computation.
    """

    def __init__(
        self,
        n_envs: int,
        rollout_steps: int,
        state_dim: int,
        action_dim: int,
        joints: int = 7,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
    ):
        self.n_envs = n_envs
        self.rollout_steps = rollout_steps
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.states = np.zeros((rollout_steps, n_envs, state_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, n_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, n_envs, 1), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, n_envs, 1), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps, n_envs, 1), dtype=np.float32)
        self.values = np.zeros((rollout_steps, n_envs, 1), dtype=np.float32)

        # Physics loss fields (stored per-step for differentiable regularization)
        self.q_prev = np.zeros((rollout_steps, n_envs, joints), dtype=np.float32)
        self.dq_prev = np.zeros((rollout_steps, n_envs, joints), dtype=np.float32)
        self.dq_next = np.zeros((rollout_steps, n_envs, joints), dtype=np.float32)
        self.J = np.zeros((rollout_steps, n_envs, 3, joints), dtype=np.float32)
        self.sigma = np.zeros((rollout_steps, n_envs, 1), dtype=np.float32)
        self.dx_nom = np.zeros((rollout_steps, n_envs, 3), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((rollout_steps, n_envs, 1), dtype=np.float32)
        self.returns = np.zeros((rollout_steps, n_envs, 1), dtype=np.float32)

        self._step = 0  # current step index (0..rollout_steps-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        states_row,
        actions_row,
        rewards_row,
        dones_row,
        log_probs_row,
        values_row,
        q=None,
        dq=None,
        dq_next=None,
        J=None,
        sigma=None,
        dx_nom=None,
    ):
        """
        Push one step across all envs into the buffer.

        Each *_row should have shape (n_envs, ...) or (n_envs,).
        """
        s = self._step
        if s >= self.rollout_steps:
            raise IndexError(
                f"RolloutBuffer step {s} exceeds capacity {self.rollout_steps}"
            )

        self.states[s] = states_row
        self.actions[s] = actions_row
        self.rewards[s] = np.asarray(rewards_row).reshape(-1, 1)
        self.dones[s] = np.asarray(dones_row, dtype=np.float32).reshape(-1, 1)
        self.log_probs[s] = np.asarray(log_probs_row).reshape(-1, 1)
        self.values[s] = np.asarray(values_row).reshape(-1, 1)

        if q is not None:
            self.q_prev[s] = q
            self.dq_prev[s] = dq
            self.dq_next[s] = dq_next
        if J is not None:
            self.J[s] = J
            self.sigma[s] = np.asarray(sigma).reshape(-1, 1)
            self.dx_nom[s] = dx_nom

        self._step += 1

    def compute_advantages(self, last_values: np.ndarray):
        """
        Compute GAE advantages and discounted returns for each env.

        Parameters
        ----------
        last_values : ndarray of shape (n_envs, 1)
            V(s) for each env's observation *after* the last step,
            used as bootstrap for unfinished trajectories.
        """
        n_steps = self._step
        n_envs = self.n_envs

        for env_idx in range(n_envs):
            env_values = self.values[:n_steps, env_idx, 0]  # (n_steps,)
            env_rewards = self.rewards[:n_steps, env_idx, 0]
            env_dones = self.dones[:n_steps, env_idx, 0]
            next_val = float(last_values[env_idx])

            gae = 0.0
            for t in reversed(range(n_steps)):
                delta = (
                    env_rewards[t]
                    + self.gamma * next_val * (1.0 - env_dones[t])
                    - env_values[t]
                )
                gae = delta + self.gamma * self.gae_lambda * (1.0 - env_dones[t]) * gae
                self.advantages[t, env_idx, 0] = gae
                next_val = env_values[t]

            # returns = advantages + values
            self.returns[:n_steps, env_idx, 0] = (
                self.advantages[:n_steps, env_idx, 0] + env_values
            )

    def sample(self, batch_size: int) -> dict:
        """
        Random mini-batch from the buffer (with replacement).

        Flattens the [step, env] dimensions and samples.
        """
        n_steps = self._step
        n_envs = self.n_envs
        total = n_steps * n_envs
        if total < 1:
            return {}

        idx = np.random.choice(total, min(batch_size, total), replace=True)
        step_idx = idx // n_envs
        env_idx = idx % n_envs

        return {
            "state": self.states[step_idx, env_idx],
            "action": self.actions[step_idx, env_idx],
            "old_log_prob": self.log_probs[step_idx, env_idx],
            "advantages": self.advantages[step_idx, env_idx],
            "returns": self.returns[step_idx, env_idx],
            "q": self.q_prev[step_idx, env_idx],
            "dq": self.dq_prev[step_idx, env_idx],
            "dq_next": self.dq_next[step_idx, env_idx],
            "J": self.J[step_idx, env_idx],
            "sigma": self.sigma[step_idx, env_idx],
            "dx_nom": self.dx_nom[step_idx, env_idx],
        }

    def clear(self):
        """Reset buffer for the next rollout."""
        self._step = 0

    def __len__(self) -> int:
        return self._step * self.n_envs
