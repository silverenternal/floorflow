"""Low-rank tensor-ring transport over room graph tokens."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RoomGraphTensorRing(nn.Module):
    def __init__(self, hidden_dim: int, rank: int = 32, order: int = 3, residual_scale: float = 0.2) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank)
        self.cores = nn.Parameter(torch.empty(order, rank, rank))
        self.up = nn.Linear(rank, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.residual_scale = float(residual_scale)
        nn.init.xavier_uniform_(self.cores)

    def forward(self, tokens: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return tokens
        z = torch.tanh(self.down(tokens))
        if adjacency is not None:
            weights = adjacency.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            z = torch.bmm(weights, z)
        residual = z
        for core in self.cores:
            z = torch.tanh(z @ core / math.sqrt(core.shape[0]) + residual)
        return self.norm(tokens + self.residual_scale * self.up(z))
