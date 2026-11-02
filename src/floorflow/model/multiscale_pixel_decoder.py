"""Lightweight multi-scale pixel decoder adapted from EoMT ScaleBlock."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleBlockLite(nn.Module):
    """EoMT-style 2x scale block without the timm dependency."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(max(1, channels // 8), channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv2(self.act(self.conv1(value))))


class MultiScalePixelDecoder(nn.Module):
    """Build a top-down pyramid from one backbone feature map."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 96,
        levels: int = 3,
        upscale_steps: int = 0,
    ) -> None:
        super().__init__()
        if levels <= 0:
            raise ValueError("levels must be positive")
        self.levels = int(levels)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(int(in_channels), hidden_channels, 1),
                    nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
                    nn.SiLU(),
                )
                for _ in range(self.levels)
            ]
        )
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
                    nn.SiLU(),
                )
                for _ in range(self.levels)
            ]
        )
        self.coarse_to_fine = nn.ModuleList(
            [
                ScaleBlockLite(hidden_channels)
                for _ in range(max(0, int(upscale_steps)))
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(max(1, hidden_channels // 16), hidden_channels),
            nn.SiLU(),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        levels = []
        for index, projection in enumerate(self.projections):
            level = features
            if index:
                level = F.avg_pool2d(features, 2**index)
            levels.append(projection(level))
        pyramid = [levels[-1]]
        for index in range(self.levels - 2, -1, -1):
            top = F.interpolate(
                pyramid[0],
                size=levels[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pyramid.insert(0, self.smooth[index](top + levels[index]))
        output = pyramid[0]
        for scale_block in self.coarse_to_fine:
            output = scale_block(output)
        return self.output(output), pyramid


__all__ = ["MultiScalePixelDecoder", "ScaleBlockLite"]
