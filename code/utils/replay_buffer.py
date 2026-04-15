"""
replay_buffer.py
----------------
Experience replay buffer for SAC.
Stores (s, a, r, s', done) transitions.
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:

    def __init__(self, capacity: int = 100_000,
                 state_dim: int = 22, action_dim: int = 7):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states   = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_s   = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),          dtype=np.float32)

        # Extra fields for physics loss computation
        self.q_prev   = np.zeros((capacity, action_dim), dtype=np.float32)
        self.dq_prev  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.dq_next  = np.zeros((capacity, action_dim), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done,
             q=None, dq=None, dq_next=None):
        i = self.ptr
        self.states[i]  = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_s[i]  = next_state
        self.dones[i]   = float(done)

        if q is not None:
            self.q_prev[i]  = q
            self.dq_prev[i] = dq
            self.dq_next[i] = dq_next

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return dict(
            state      = self.states[idx],
            action     = self.actions[idx],
            reward     = self.rewards[idx],
            next_state = self.next_s[idx],
            done       = self.dones[idx],
            q          = self.q_prev[idx],
            dq         = self.dq_prev[idx],
            dq_next    = self.dq_next[idx],
        )

    def __len__(self):
        return self.size
