"""Semantic-token conditioned refinement of coarse room regions."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoomRegionRefiner(nn.Module):
    """Refine every coarse room region using its graph token as a local condition."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
            nn.SiLU(),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.shared_mask = nn.Conv2d(hidden_dim, num_room_classes, 1)
        self.class_scale = nn.Linear(hidden_dim, 1)
        self.class_bias = nn.Linear(hidden_dim, 1)
        self.residual_scale = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.shared_mask.weight)
        nn.init.zeros_(self.shared_mask.bias)
        nn.init.zeros_(self.class_scale.weight)
        nn.init.zeros_(self.class_scale.bias)
        nn.init.zeros_(self.class_bias.weight)
        nn.init.zeros_(self.class_bias.bias)

    def forward(
        self,
        features: torch.Tensor,
        coarse_logits: torch.Tensor,
        room_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = F.adaptive_avg_pool2d(
            features,
            coarse_logits.shape[-2:],
        )
        hidden = self.feature(hidden)
        query = self.query(room_tokens)
        scores = torch.einsum("bkd,bdhw->bkhw", query, hidden)
        scores = scores / math.sqrt(hidden.shape[1])
        attention = torch.softmax(scores.flatten(2), dim=-1).reshape_as(scores)
        class_context = torch.einsum("bkhw,bdhw->bkd", attention, hidden)
        scale = self.class_scale(class_context).sigmoid()[..., 0][:, :, None, None]
        bias = self.class_bias(class_context)[..., 0][:, :, None, None]
        residual = (
            self.shared_mask(hidden)
            * scale
            + bias
        )
        refined_logits = coarse_logits + self.residual_scale.to(residual.dtype) * residual
        return {
            "refined_room_logits": refined_logits,
            "refined_room_probabilities": refined_logits.softmax(dim=1),
            "room_region_attention": attention,
        }
