"""
gpu_buffer.py
-------------
GPU-accelerated replay buffer that stores transitions as PyTorch tensors
on CUDA, avoiding GPU→CPU→GPU round trips.

Accepts JAX arrays (zero-copy via __cuda_array_interface__) or numpy arrays.
Samples return dict of torch CUDA tensors ready for agent.update().
"""

import numpy as np
import torch


class GpuReplayBuffer:
    """Circular replay buffer backed by PyTorch CUDA tensors."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, joints: int = 7,
                 device: str = "cuda"):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.joints = joints
        self.device = torch.device(device)
        self.ptr = 0
        self._size = 0

        # Pre-allocate GPU tensors
        float32 = torch.float32
        float64 = torch.float64
        kwargs = dict(device=self.device)

        self.states   = torch.zeros((capacity, state_dim),  dtype=float32, **kwargs)
        self.actions  = torch.zeros((capacity, action_dim), dtype=float32, **kwargs)
        self.rewards  = torch.zeros((capacity, 1),          dtype=float64, **kwargs)
        self.next_s   = torch.zeros((capacity, state_dim),  dtype=float32, **kwargs)
        self.dones    = torch.zeros((capacity, 1),          dtype=float32, **kwargs)
        self.q_prev   = torch.zeros((capacity, joints),     dtype=float32, **kwargs)
        self.dq_prev  = torch.zeros((capacity, joints),     dtype=float32, **kwargs)
        self.dq_next  = torch.zeros((capacity, joints),     dtype=float32, **kwargs)
        self.J        = torch.zeros((capacity, 3, joints),  dtype=float32, **kwargs)
        self.sigma    = torch.zeros((capacity, 1),          dtype=float32, **kwargs)
        self.dx_nom   = torch.zeros((capacity, 3),          dtype=float32, **kwargs)
        self.costs    = torch.zeros((capacity, 1),          dtype=float64, **kwargs)

    def _to(self, x, dtype):
        """Convert JAX/numpy/torch to torch tensor on device."""
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=dtype)
        # JAX arrays expose __cuda_array_interface__ — zero-copy
        if hasattr(x, '__cuda_array_interface__'):
            return torch.as_tensor(x, device=self.device).to(dtype=dtype)
        # numpy or other
        return torch.as_tensor(np.asarray(x), device=self.device, dtype=dtype)

    def push(self, state, action, reward, next_state, done,
             q=None, dq=None, dq_next=None,
             J=None, sigma=None, dx_nom=None, cost=None):
        """Single transition push."""
        i = self.ptr
        self.states[i]  = self._to(state, torch.float32)
        self.actions[i] = self._to(action, torch.float32)
        self.rewards[i] = self._to(reward, torch.float64)
        self.next_s[i]  = self._to(next_state, torch.float32)
        self.dones[i]   = self._to(float(done), torch.float32)
        if cost is not None:
            self.costs[i] = self._to(cost, torch.float64)
        if q is not None:
            self.q_prev[i]  = self._to(q, torch.float32)
            self.dq_prev[i] = self._to(dq, torch.float32)
            self.dq_next[i] = self._to(dq_next, torch.float32)
        if J is not None:
            self.J[i]      = self._to(J, torch.float32)
            self.sigma[i]  = self._to(sigma, torch.float32)
            self.dx_nom[i] = self._to(dx_nom, torch.float32)
        self.ptr  = (self.ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push_batch(self, states, actions, rewards, next_states, dones,
                   infos, keys=None):
        """Push N transitions at once. All inputs are arrays with batch dim 0.

        This is 5-10x faster than calling push() N times.
        """
        n = len(states)
        idx = (self.ptr + np.arange(n)) % self.capacity

        self.states[idx]  = self._to(states, torch.float32)
        self.actions[idx] = self._to(actions, torch.float32)
        self.rewards[idx] = self._to(rewards, torch.float64).reshape(-1, 1)
        self.next_s[idx]  = self._to(next_states, torch.float32)
        self.dones[idx]   = self._to(dones, torch.float32).reshape(-1, 1)

        # Info fields (per-env per-step metadata)
        if infos and len(infos) > 0:
            # Stack list-of-dicts into per-key arrays
            if keys is None:
                keys = ['q_before', 'dq_before', 'J_before', 'sigma',
                        'dx_nom_before', 'cost']
            for key in keys:
                if key not in infos[0]:
                    continue
                vals = [info[key] for info in infos]
                arr = np.stack(vals) if not isinstance(vals[0], (int, float)) else np.array(vals)
                if key == 'q_before':
                    self.q_prev[idx] = self._to(arr, torch.float32)
                elif key == 'dq_before':
                    self.dq_prev[idx] = self._to(arr, torch.float32)
                    self.dq_next[idx] = self._to(arr, torch.float32)
                elif key == 'J_before':
                    self.J[idx] = self._to(arr, torch.float32)
                elif key == 'sigma':
                    self.sigma[idx] = self._to(np.array(vals, dtype=np.float32).reshape(-1, 1), torch.float32)
                elif key == 'dx_nom_before':
                    self.dx_nom[idx] = self._to(arr, torch.float32)
                elif key == 'cost':
                    self.costs[idx] = self._to(np.array(vals).reshape(-1, 1), torch.float64)

        self.ptr  = (self.ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int):
        """Return batch of GPU tensors, already on device.

        Compatible with SACAgent.update() — skip FloatTensor+to(device) there.
        """
        idx = np.random.choice(self._size, batch_size, replace=False)
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

    def sample_numpy(self, batch_size: int):
        """Return batch as numpy (CPU). For obs normalizer which needs numpy."""
        idx = np.random.choice(self._size, batch_size, replace=False)
        return dict(
            state      = self.states[idx].cpu().numpy(),
            action     = self.actions[idx].cpu().numpy(),
            reward     = self.rewards[idx].cpu().numpy(),
            next_state = self.next_s[idx].cpu().numpy(),
            done       = self.dones[idx].cpu().numpy(),
            cost       = self.costs[idx].cpu().numpy(),
            q          = self.q_prev[idx].cpu().numpy(),
            dq         = self.dq_prev[idx].cpu().numpy(),
            dq_next    = self.dq_next[idx].cpu().numpy(),
            J          = self.J[idx].cpu().numpy(),
            sigma      = self.sigma[idx].cpu().numpy(),
            dx_nom     = self.dx_nom[idx].cpu().numpy(),
        )

    def __len__(self):
        return self._size
