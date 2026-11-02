"""Typed structural memory consumed by the floorplan token decoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from floorflow.objectives.token_grammar import TOKEN_STAGES


class TypedTokenMemory(nn.Module):
    """Fuse boundary, room, CAD-action, and vector-stage tokens."""

    def __init__(self, hidden_dim: int, vector_stage_names: Sequence[str]) -> None:
        super().__init__()
        self.vector_stage_names = tuple(vector_stage_names)
        self.type_embedding = nn.Embedding(4, int(hidden_dim))
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.out_norm = nn.LayerNorm(int(hidden_dim))

    def forward(
        self,
        boundary_tokens: torch.Tensor,
        boundary_valid: torch.Tensor,
        room_tokens: torch.Tensor,
        room_valid: torch.Tensor,
        stage_tokens: torch.Tensor,
        vector_stage_tokens: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if boundary_tokens.ndim != 3 or room_tokens.ndim != 3 or stage_tokens.ndim != 3:
            raise ValueError("typed memory inputs must be [B,N,D]")
        if boundary_valid.shape != boundary_tokens.shape[:2]:
            raise ValueError("boundary_valid must align with boundary tokens")
        if room_valid.shape != room_tokens.shape[:2]:
            raise ValueError("room_valid must align with room tokens")
        if stage_tokens.shape[1] != len(TOKEN_STAGES):
            raise ValueError("stage tokens must follow TOKEN_STAGES")
        missing = set(self.vector_stage_names) - set(vector_stage_tokens)
        if missing:
            raise ValueError(f"missing vector stage tokens: {sorted(missing)}")
        vector_tokens = torch.stack(
            [vector_stage_tokens[name] for name in self.vector_stage_names],
            dim=1,
        )
        device = boundary_tokens.device
        groups = [
            (boundary_tokens, boundary_valid, 0),
            (room_tokens, room_valid, 1),
            (
                stage_tokens,
                torch.ones(
                    stage_tokens.shape[:2],
                    dtype=torch.bool,
                    device=device,
                ),
                2,
            ),
            (
                vector_tokens,
                torch.ones(
                    vector_tokens.shape[:2],
                    dtype=torch.bool,
                    device=device,
                ),
                3,
            ),
        ]
        memory = torch.cat(
            [
                self.norm(tokens) + self.type_embedding.weight[type_id]
                for tokens, _valid, type_id in groups
            ],
            dim=1,
        )
        valid = torch.cat([mask for _tokens, mask, _type_id in groups], dim=1)
        return self.out_norm(memory), valid


__all__ = ["TypedTokenMemory"]
