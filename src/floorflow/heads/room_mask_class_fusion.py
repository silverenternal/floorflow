"""Class-only query adaptation with optional room-program tokens."""

from __future__ import annotations

import torch
from torch import nn


class RoomMaskClassFusion(nn.Module):
    """Adapt semantic queries without changing the mask-bearing query path."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        *,
        program_token_dim: int = 0,
    ) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.normal_(self.adapter[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.adapter[-1].bias)
        self.program_projection = (
            nn.Sequential(
                nn.Linear(int(program_token_dim), hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            if program_token_dim > 0
            else None
        )
        self.program_attention = (
            nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
            if program_token_dim > 0
            else None
        )
        self.program_gate = (
            nn.Parameter(torch.tensor(1.0e-3))
            if program_token_dim > 0
            else None
        )

    def forward(
        self,
        queries: torch.Tensor,
        program_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        adapted = queries + self.adapter(queries)
        if program_tokens is None:
            return adapted
        if self.program_projection is None:
            raise ValueError("decoder was not initialized for program tokens")
        if program_tokens.shape[0] != queries.shape[0]:
            raise ValueError("program token batch must match query batch")
        memory = self.program_projection(program_tokens.to(queries.dtype))
        context = self.program_attention(
            adapted,
            memory,
            memory,
            need_weights=False,
        )[0]
        return adapted + torch.tanh(self.program_gate).to(queries.dtype) * context


__all__ = ["RoomMaskClassFusion"]
