"""Synchronous semantic endpoint head for floorplan channels."""

from __future__ import annotations

import torch
import torch.nn as nn


class SemanticEndpointHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = max(32, in_channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
