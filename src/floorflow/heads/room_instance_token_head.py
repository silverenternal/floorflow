"""Semantic room-instance token shared by graph and token branches."""

from __future__ import annotations

import torch
from torch import nn


class RoomInstanceTokenHead(nn.Module):
    def __init__(self, hidden_dim: int, room_classes: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.class_embedding = nn.Embedding(int(room_classes), self.hidden_dim)
        self.semantic_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.gate = nn.Linear(self.hidden_dim * 2, 1)

    def forward(
        self,
        room_features: torch.Tensor,
        class_logits: torch.Tensor,
        room_valid: torch.Tensor,
    ) -> torch.Tensor:
        if room_features.ndim != 3 or class_logits.shape[:2] != room_features.shape[:2]:
            raise ValueError("room features and class logits must align")
        semantic = class_logits[..., :-1].softmax(dim=-1) @ self.class_embedding.weight
        semantic = self.semantic_projection(semantic)
        gate = self.gate(
            torch.cat([room_features, semantic], dim=-1)
        ).sigmoid()
        tokens = self.norm(room_features + gate * semantic)
        return tokens * room_valid[..., None].to(tokens.dtype)


__all__ = ["RoomInstanceTokenHead"]
