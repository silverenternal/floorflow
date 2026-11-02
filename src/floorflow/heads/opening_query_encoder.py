"""Encode detected opening centers into stage-ordered opening tokens."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class OpeningQueryEncoder(nn.Module):
    """Sample spatial evidence at detected centers and expose typed queries."""

    def __init__(self, in_channels: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.feature_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
            nn.SiLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim + 10, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        features: torch.Tensor,
        centers: torch.Tensor,
        widths: torch.Tensor,
        family_logits: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.feature_projection(features)
        if centers.shape[1] == 0:
            return hidden.new_zeros(
                hidden.shape[0],
                0,
                hidden.shape[1],
            )
        grid = centers * 2.0 - 1.0
        sampled = F.grid_sample(
            hidden,
            grid[:, :, None],
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[:, :, :, 0].permute(0, 2, 1)
        angle = centers * (2.0 * math.pi)
        geometry = torch.cat(
            [
                centers,
                torch.sin(angle),
                torch.cos(angle),
                widths[..., None],
                family_logits.softmax(dim=-1),
                valid[..., None].to(centers.dtype),
            ],
            dim=-1,
        )
        tokens = self.encoder(torch.cat([sampled, geometry], dim=-1))
        return tokens * valid[..., None].to(tokens.dtype)


__all__ = ["OpeningQueryEncoder"]
