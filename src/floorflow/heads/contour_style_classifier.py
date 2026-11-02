"""Lightweight contour-only style classifier."""

from __future__ import annotations

import torch
from torch import nn


class ContourStyleClassifier(nn.Module):
    def __init__(self, num_styles: int, hidden_channels: int = 32) -> None:
        super().__init__()
        channels = (hidden_channels, hidden_channels * 2, hidden_channels * 4)
        self.backbone = nn.Sequential(
            self._block(1, channels[0]),
            self._block(channels[0], channels[1]),
            self._block(channels[1], channels[2]),
        )
        self.head = nn.Sequential(
            nn.Linear(channels[2] * 2, hidden_channels * 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 4, num_styles),
        )

    @staticmethod
    def _block(left: int, right: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(left, right, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, right // 8), right),
            nn.SiLU(),
            nn.Conv2d(right, right, 3, padding=1),
            nn.GroupNorm(max(1, right // 8), right),
            nn.SiLU(),
        )

    def forward(self, footprint: torch.Tensor) -> torch.Tensor:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        features = self.backbone(footprint.float())
        pooled = torch.cat(
            [features.mean(dim=(2, 3)), features.amax(dim=(2, 3))],
            dim=-1,
        )
        return self.head(pooled)


__all__ = ["ContourStyleClassifier"]
