"""Source-aware residual refinement for the distribution CFM velocity."""

from __future__ import annotations

import math

import torch
from torch import nn


class SourceAwareVelocityRefiner(nn.Module):
    """Predict a zero-initialized velocity correction from x_t and contour source."""

    def __init__(self, feature_channels: int, target_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.target_channels = int(target_channels)
        input_channels = int(feature_channels) + 2 * self.target_channels + 5
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(hidden, self.target_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        features: torch.Tensor,
        x_t: torch.Tensor,
        source: torch.Tensor,
        footprint: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if source.shape != x_t.shape or source.shape[1] != self.target_channels:
            raise ValueError("source and x_t must share [B,C,H,W]")
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        dtype = features.dtype
        time_value = time.to(device=x_t.device, dtype=dtype)[:, None, None, None]
        time_channels = torch.cat(
            [
                time_value.expand_as(x_t[:, :1]),
                (1.0 - time_value).expand_as(x_t[:, :1]),
                torch.sin(math.pi * time_value).expand_as(x_t[:, :1]),
                torch.cos(math.pi * time_value).expand_as(x_t[:, :1]),
            ],
            dim=1,
        )
        conditioned = torch.cat(
            [
                features.to(dtype),
                x_t.to(dtype),
                source.to(dtype),
                footprint.to(dtype).expand(-1, 1, *x_t.shape[-2:]),
                time_channels.to(dtype),
            ],
            dim=1,
        )
        return self.output(self.encoder(conditioned))


__all__ = ["SourceAwareVelocityRefiner"]
