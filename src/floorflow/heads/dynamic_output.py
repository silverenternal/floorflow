"""Learned multi-scale output routing."""

from __future__ import annotations

import torch
from torch import nn


class DynamicMultiScaleOutputHead(nn.Module):
    """Learned per-channel, per-pixel mixture of multi-scale convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: tuple[int, ...]) -> None:
        super().__init__()
        kernel_sizes = tuple(
            sorted({int(size) for size in kernel_sizes if int(size) >= 1 and int(size) % 2 == 1})
        )
        if not kernel_sizes:
            raise ValueError("at least one positive odd kernel size is required")
        self.kernel_sizes = kernel_sizes
        self.out_channels = int(out_channels)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, size, padding=size // 2, groups=in_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels, out_channels, 1),
                )
                for size in kernel_sizes
            ]
        )
        for branch in self.branches:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
        router_channels = max(16, in_channels // 2)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, router_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(router_channels, out_channels * len(kernel_sizes), 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branches = torch.stack([branch(features) for branch in self.branches], dim=1)
        batch, _scales, channels, height, width = branches.shape
        logits = self.router(features).reshape(batch, len(self.kernel_sizes), channels, height, width)
        weights = torch.softmax(logits, dim=1)
        self.last_scale_weights = weights.detach()
        self.last_scale_weights_raw = weights
        return (weights * branches).sum(dim=1)
