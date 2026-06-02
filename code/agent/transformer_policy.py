"""
transformer_policy.py
---------------------
Transformer-based actor with action chunking (ACT-style).

Encoder-decoder architecture:
  - Encoder: TransformerEncoder over K history frames
  - Decoder: TransformerDecoder with H learnable query tokens → H action steps
  - Only a_0 receives RL gradient; positions 1..H-1 get L2 smoothness regularization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fallback); overridden by learnable pos_embed."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ActionChunkActor(nn.Module):
    """
    Encoder-decoder transformer actor that generates H-step action chunks from K-frame history.

    Encoder: K frames → Linear projection + pos_embed → TransformerEncoder → memory
    Decoder: H learnable query tokens → TransformerDecoder (cross-attn to memory) → H actions
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 frame_stack: int = 4,
                 action_horizon: int = 8,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_enc_layers: int = 2,
                 n_dec_layers: int = 2,
                 dropout: float = 0.1,
                 task_scale: float = 1.0,
                 nullspace_scale: float = 0.15):
        super().__init__()
        self.obs_dim = obs_dim
        self.per_frame_dim = obs_dim // frame_stack
        self.action_dim = action_dim
        self.frame_stack = frame_stack
        self.action_horizon = action_horizon
        self.d_model = d_model
        self.task_dim = 3
        self.task_scale = task_scale
        self.nullspace_scale = nullspace_scale

        # Shared frame projection
        self.frame_proj = nn.Linear(self.per_frame_dim, d_model)

        # Learnable position embeddings
        self.enc_pos_embed = nn.Parameter(torch.randn(1, frame_stack, d_model) * 0.02)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # Decoder query tokens (learnable)
        self.query_tokens = nn.Parameter(torch.randn(1, action_horizon, d_model) * 0.02)

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)

        # Action heads (shared across positions)
        self.mean_head = nn.Linear(d_model, action_dim)
        self.log_std_head = nn.Linear(d_model, action_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def _encode(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """Encode stacked observation frames into memory.

        Args:
            obs_flat: [B, obs_dim * frame_stack]

        Returns:
            memory: [B, frame_stack, d_model]
        """
        B = obs_flat.size(0)
        x = obs_flat.view(B, self.frame_stack, self.per_frame_dim)
        x = self.frame_proj(x)                       # [B, K, d]
        x = x + self.enc_pos_embed[:, :self.frame_stack]
        x = self.encoder(x)                          # [B, K, d]
        return x

    def _decode(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode memory into action chunk.

        Args:
            memory: [B, frame_stack, d_model]

        Returns:
            chunk_mean:    [B, H, action_dim]
            chunk_log_std: [B, H, action_dim]
        """
        B = memory.size(0)
        queries = self.query_tokens.expand(B, -1, -1)  # [B, H, d]
        out = self.decoder(queries, memory)             # [B, H, d]
        chunk_mean = self.mean_head(out)                # [B, H, action_dim]
        chunk_log_std = self.log_std_head(out).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return chunk_mean, chunk_log_std

    def forward(self, obs_flat: torch.Tensor):
        """Full encoder-decoder forward pass.

        Args:
            obs_flat: [B, obs_dim * frame_stack]

        Returns:
            chunk_mean:    [B, H, action_dim]
            chunk_log_std: [B, H, action_dim]
        """
        memory = self._encode(obs_flat)
        return self._decode(memory)

    def sample(self, obs_flat: torch.Tensor):
        """Sample action chunk. Returns first action with RL-compatible signature.

        Args:
            obs_flat: [B, obs_dim * frame_stack]

        Returns:
            action:      [B, action_dim]  — first step of chunk (for env execution)
            log_prob:    [B, 1]           — log prob of first action
            mean_action: [B, action_dim]  — deterministic first action
        """
        chunk_mean, chunk_log_std = self.forward(obs_flat)

        # Only position 0 participates in RL
        mean_0 = chunk_mean[:, 0, :]       # [B, action_dim]
        log_std_0 = chunk_log_std[:, 0, :]

        std_0 = log_std_0.exp()
        if torch.isnan(mean_0).any() or torch.isnan(std_0).any():
            mean_0 = torch.nan_to_num(mean_0, nan=0.0)
            std_0 = torch.nan_to_num(std_0, nan=1.0).clamp(min=1e-6)

        dist = Normal(mean_0, std_0)
        x = dist.rsample()
        y = torch.tanh(x)

        # Apply per-dimension scaling (task vs nullspace)
        scale = torch.ones_like(y)
        scale[:, :self.task_dim] = self.task_scale
        scale[:, self.task_dim:] = self.nullspace_scale
        action = y * scale

        # Log prob with tanh change-of-variables
        log_prob = dist.log_prob(x) - torch.log(scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob * (scale > 0).float()
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Deterministic mean (also position 0)
        mean_scale = torch.ones_like(mean_0)
        mean_scale[:, :self.task_dim] = self.task_scale
        mean_scale[:, self.task_dim:] = self.nullspace_scale
        mean_action = torch.tanh(mean_0) * mean_scale

        return action, log_prob, mean_action

    def sample_chunk(self, obs_flat: torch.Tensor):
        """Sample full action chunk (all H steps with stochasticity).

        Returns:
            chunk:  [B, H, action_dim]
        """
        chunk_mean, chunk_log_std = self.forward(obs_flat)
        B, H, A = chunk_mean.shape

        std = chunk_log_std.exp()
        std = torch.nan_to_num(std, nan=1.0).clamp(min=1e-6)
        dist = Normal(chunk_mean, std)
        x = dist.rsample()
        y = torch.tanh(x)

        scale = torch.ones_like(y)
        scale[:, :, :self.task_dim] = self.task_scale
        scale[:, :, self.task_dim:] = self.nullspace_scale

        return y * scale

    def compute_log_prob(self, obs_flat: torch.Tensor, action: torch.Tensor):
        """Compute log_prob of a given action (position 0 only).

        Args:
            obs_flat: [B, obs_dim * frame_stack]
            action:   [B, action_dim]

        Returns:
            log_prob: [B, 1]
        """
        chunk_mean, chunk_log_std = self.forward(obs_flat)
        mean_0 = chunk_mean[:, 0, :]
        log_std_0 = chunk_log_std[:, 0, :]

        std_0 = log_std_0.exp().clamp(min=1e-6)
        dist = Normal(mean_0, std_0)

        # Invert tanh squashing to get pre-squashed action
        scale = torch.ones_like(action)
        scale[:, :self.task_dim] = self.task_scale
        scale[:, self.task_dim:] = self.nullspace_scale

        y = action / scale.clamp(min=1e-6)
        y = y.clamp(-0.999, 0.999)
        x = 0.5 * torch.log((1 + y) / (1 - y + 1e-6))

        log_prob = dist.log_prob(x) - torch.log(scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob * (scale > 0).float()
        return log_prob.sum(dim=-1, keepdim=True)

    def compute_chunk_smoothness_loss(self, chunk: torch.Tensor) -> torch.Tensor:
        """L2 penalty on consecutive action differences within chunk."""
        diffs = chunk[:, 1:] - chunk[:, :-1]
        return (diffs ** 2).mean()
