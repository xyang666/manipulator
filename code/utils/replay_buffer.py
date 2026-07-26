"""
replay_buffer.py
----------------
Experience replay buffer for SAC.
Supports both uniform sampling (ReplayBuffer) and
prioritized experience replay (PrioritizedReplayBuffer).

Author: xie yang
Date:   2025-06

"""

import numpy as np
import gzip
import pickle


class SumTree:
    """Binary SumTree for O(log N) priority-based sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)   # internal nodes + leaves
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self.size = 0

    def add(self, priority: float, data):
        idx = self.ptr
        self.data[idx] = data
        self._update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _update(self, idx: int, priority: float):
        """Set leaf priority and propagate up."""
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = priority
        tree_idx //= 2
        while tree_idx >= 1:
            self.tree[tree_idx] = self.tree[2 * tree_idx] + self.tree[2 * tree_idx + 1]
            tree_idx //= 2

    def update_batch(self, indices: np.ndarray, priorities: np.ndarray):
        """Batch update priorities at given data indices."""
        for idx, prio in zip(indices, priorities):
            prio = max(float(prio), 1e-6)
            self._update(int(idx), prio)

    def sample(self, batch_size: int):
        """Sample batch_size items proportional to priority."""
        indices = np.zeros(batch_size, dtype=np.int32)
        batch = []
        total = self.tree[1]
        segment = total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            idx = self._retrieve(v)
            indices[i] = idx
            batch.append(self.data[idx])
        return indices, batch

    def _retrieve(self, v: float) -> int:
        """Find leaf index for cumulative probability v."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if v <= self.tree[left]:
                idx = left
            else:
                v -= self.tree[left]
                idx = right
        return idx - self.capacity

    @property
    def max_priority(self) -> float:
        return float(np.max(self.tree[-self.capacity:]))


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay (Schaul et al., 2016).
    Samples transitions with probability p_i ∝ |TD-error_i| + ε.
    Importance-sampling weights correct the bias: w_i = (1/N * 1/P(i))^β.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int, joints: int = 7,
                 alpha_prio: float = 0.6, beta_prio: float = 0.4, beta_anneal: float = 0):
        self.capacity = capacity
        self.alpha_prio = alpha_prio          # how much prioritization (0=uniform, 1=full)
        self.beta = beta_prio                 # importance sampling correction (0=none, 1=full)
        self.beta_init = beta_prio
        self.beta_anneal = beta_anneal        # beta increment per sample (for annealing to 1)
        self.eps = 1e-6                       # small constant to avoid zero priority
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.joints = joints

        self.tree = SumTree(capacity)

    def push(self, state, action, reward, next_state, done,
             q=None, dq=None, dq_next=None,
             J=None, sigma=None, dx_nom=None, cost=None):
        data = {
            "state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.float64(reward),
            "next_state": np.asarray(next_state, dtype=np.float32),
            "done": np.float32(done),
            "q": np.asarray(q, dtype=np.float32) if q is not None else None,
            "dq": np.asarray(dq, dtype=np.float32) if dq is not None else None,
            "dq_next": np.asarray(dq_next, dtype=np.float32) if dq_next is not None else None,
            "J": np.asarray(J, dtype=np.float32) if J is not None else None,
            "sigma": np.float32(sigma) if sigma is not None else None,
            "dx_nom": np.asarray(dx_nom, dtype=np.float32) if dx_nom is not None else None,
            "cost": np.float64(cost) if cost is not None else np.float64(0.0),
        }
        # New transitions get max priority to ensure they're sampled at least once
        prio = self.tree.max_priority if self.tree.size > 0 else 1.0
        self.tree.add(prio, data)

    def sample(self, batch_size: int):
        """Return batch + indices + importance weights."""
        indices, batch_data = self.tree.sample(batch_size)
        self._n_samples = getattr(self, '_n_samples', 0) + 1

        # Anneal beta towards 1
        if self.beta_anneal > 0:
            self.beta = min(1.0, self.beta_init + self._n_samples * self.beta_anneal)

        # Importance-sampling weights
        total = self.tree.tree[1]
        N = self.tree.size
        weights = np.zeros(batch_size, dtype=np.float32)
        for i, idx in enumerate(indices):
            p = self.tree.tree[idx + self.tree.capacity] / total
            w = (N * p) ** (-self.beta)
            weights[i] = w
        weights /= weights.max()  # normalize for stability

        # Assemble batch dict
        batch = {
            "state":      np.array([d["state"] for d in batch_data], dtype=np.float32),
            "action":     np.array([d["action"] for d in batch_data], dtype=np.float32),
            "reward":     np.array([d["reward"] for d in batch_data], dtype=np.float64).reshape(-1, 1),
            "next_state": np.array([d["next_state"] for d in batch_data], dtype=np.float32),
            "done":       np.array([d["done"] for d in batch_data], dtype=np.float32).reshape(-1, 1),
            "weights":    weights,
            "indices":    indices,
        }
        # Optional fields
        for key in ("q", "dq", "dq_next", "J", "sigma", "dx_nom"):
            vals = [d[key] for d in batch_data]
            if any(v is not None for v in vals):
                arr = np.stack([v if v is not None else np.zeros(1) for v in vals])
                if key in ("sigma",):
                    arr = arr.reshape(-1, 1)
                batch[key] = arr.astype(np.float32)
        # Cost is always stored (defaults to 0.0 if not provided)
        batch["cost"] = np.array([d["cost"] for d in batch_data], dtype=np.float64).reshape(-1, 1)

        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """After computing TD-errors, update priorities in the tree."""
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha_prio
        self.tree.update_batch(indices, priorities)

    def clear(self):
        self.tree = SumTree(self.capacity)

    def __len__(self):
        return self.tree.size

    def save(self, path: str) -> None:
        with gzip.open(path, "wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        with gzip.open(path, "rb") as stream:
            restored = pickle.load(stream)
        if not isinstance(restored, PrioritizedReplayBuffer):
            raise TypeError("replay file does not contain a prioritized buffer")
        if (restored.capacity, restored.state_dim, restored.action_dim) != \
                (self.capacity, self.state_dim, self.action_dim):
            raise ValueError("replay buffer shape/capacity mismatch")
        self.__dict__.update(restored.__dict__)


class ReplayBuffer:
    """Uniform replay buffer (original implementation, unchanged)."""

    def __init__(self, capacity: int = 100_000,
                 state_dim: int = 22, action_dim: int = 13, joints: int = 7):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.joints = joints

        self.states   = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),          dtype=np.float64)
        self.next_s   = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),          dtype=np.float32)

        self.q_prev   = np.zeros((capacity, joints), dtype=np.float32)
        self.dq_prev  = np.zeros((capacity, joints), dtype=np.float32)
        self.dq_next  = np.zeros((capacity, joints), dtype=np.float32)
        self.J        = np.zeros((capacity, 3, joints), dtype=np.float32)
        self.sigma    = np.zeros((capacity, 1),        dtype=np.float32)
        self.dx_nom   = np.zeros((capacity, 3),        dtype=np.float32)
        self.costs    = np.zeros((capacity, 1),        dtype=np.float64)

        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done,
             q=None, dq=None, dq_next=None,
             J=None, sigma=None, dx_nom=None, cost=None):
        i = self.ptr
        self.states[i]  = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_s[i]  = next_state
        self.dones[i]   = float(done)
        if cost is not None:
            self.costs[i]  = cost
        if q is not None:
            self.q_prev[i]  = q
            self.dq_prev[i] = dq
            self.dq_next[i] = dq_next
        if J is not None:
            self.J[i]      = J
            self.sigma[i]  = sigma
            self.dx_nom[i] = dx_nom
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, state, action, reward, next_state, done,
                   q=None, dq=None, dq_next=None,
                   J=None, sigma=None, dx_nom=None, cost=None):
        """Insert a vector-environment batch with vectorized ring writes."""
        states = np.asarray(state, dtype=np.float32)
        n = len(states)
        if n == 0:
            return
        if n > self.capacity:
            start = n - self.capacity
            return self.push_batch(
                states[start:], np.asarray(action)[start:],
                np.asarray(reward)[start:], np.asarray(next_state)[start:],
                np.asarray(done)[start:],
                q=None if q is None else np.asarray(q)[start:],
                dq=None if dq is None else np.asarray(dq)[start:],
                dq_next=None if dq_next is None else np.asarray(dq_next)[start:],
                J=None if J is None else np.asarray(J)[start:],
                sigma=None if sigma is None else np.asarray(sigma)[start:],
                dx_nom=None if dx_nom is None else np.asarray(dx_nom)[start:],
                cost=None if cost is None else np.asarray(cost)[start:],
            )

        idx = (self.ptr + np.arange(n)) % self.capacity
        self.states[idx] = states
        self.actions[idx] = np.asarray(action, dtype=np.float32)
        self.rewards[idx, 0] = np.asarray(reward, dtype=np.float64).reshape(-1)
        self.next_s[idx] = np.asarray(next_state, dtype=np.float32)
        self.dones[idx, 0] = np.asarray(done, dtype=np.float32).reshape(-1)
        if cost is not None:
            self.costs[idx, 0] = np.asarray(cost, dtype=np.float64).reshape(-1)
        if q is not None:
            self.q_prev[idx] = np.asarray(q, dtype=np.float32)
            self.dq_prev[idx] = np.asarray(dq, dtype=np.float32)
            self.dq_next[idx] = np.asarray(dq_next, dtype=np.float32)
        if J is not None:
            self.J[idx] = np.asarray(J, dtype=np.float32)
            self.sigma[idx, 0] = np.asarray(sigma, dtype=np.float32).reshape(-1)
            self.dx_nom[idx] = np.asarray(dx_nom, dtype=np.float32)
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def clear(self):
        """Clear all data (reset ptr and size) to discard random-exploration buffer."""
        self.ptr = 0
        self.size = 0

    def sample(self, batch_size: int):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return dict(
            state      = self.states[idx],
            action     = self.actions[idx],
            reward     = self.rewards[idx],
            next_state = self.next_s[idx],
            done       = self.dones[idx],
            cost       = self.costs[idx],
            q          = self.q_prev[idx],
            dq         = self.dq_prev[idx],
            dq_next    = self.dq_next[idx],
            J          = self.J[idx],
            sigma      = self.sigma[idx],
            dx_nom     = self.dx_nom[idx],
        )

    def __len__(self):
        return self.size

    def save(self, path: str) -> None:
        """Persist active transitions and ring-buffer position."""
        count = self.capacity if self.size == self.capacity else self.size
        np.savez_compressed(
            path, capacity=self.capacity, state_dim=self.state_dim,
            action_dim=self.action_dim, joints=self.joints,
            ptr=self.ptr, size=self.size,
            states=self.states[:count], actions=self.actions[:count],
            rewards=self.rewards[:count], next_s=self.next_s[:count],
            dones=self.dones[:count], q_prev=self.q_prev[:count],
            dq_prev=self.dq_prev[:count], dq_next=self.dq_next[:count],
            J=self.J[:count], sigma=self.sigma[:count],
            dx_nom=self.dx_nom[:count], costs=self.costs[:count],
        )

    def load(self, path: str) -> None:
        with np.load(path, allow_pickle=False) as state:
            expected = (self.capacity, self.state_dim, self.action_dim, self.joints)
            actual = tuple(int(state[k]) for k in
                           ("capacity", "state_dim", "action_dim", "joints"))
            if actual != expected:
                raise ValueError(f"replay buffer mismatch: expected {expected}, got {actual}")
            self.ptr, self.size = int(state["ptr"]), int(state["size"])
            count = self.capacity if self.size == self.capacity else self.size
            for name in ("states", "actions", "rewards", "next_s", "dones",
                         "q_prev", "dq_prev", "dq_next", "J", "sigma",
                         "dx_nom", "costs"):
                getattr(self, name)[:count] = state[name]
