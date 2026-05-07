"""
value_network.py
----------------
Single-head value network V(s) for PPO.
Returns a scalar value estimate for a given state.
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """V(s) -> scalar value estimate."""

    def __init__(self, state_dim: int, hidden_dims: list[int] = (256, 256)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
