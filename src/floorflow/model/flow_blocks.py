"""Reusable time-conditioned U-Net blocks for floorplan flow models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class TimestepBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time(time_embedding).to(hidden.dtype)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class TimeEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.net = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(1, half - 1)
        )
        phases = t.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)
        if embedding.shape[1] < self.channels:
            embedding = F.pad(embedding, (0, self.channels - embedding.shape[1]))
        return self.net(embedding[:, : self.channels])


class GatedConditionCrossAttention(nn.Module):
    """Long-range condition fusion adapted from multiflow's gated attention."""

    def __init__(self, hidden_channels: int, heads: int = 4) -> None:
        super().__init__()
        if hidden_channels % heads:
            heads = 1
        self.query_norm = nn.LayerNorm(hidden_channels)
        self.key_norm = nn.LayerNorm(hidden_channels)
        self.attention = nn.MultiheadAttention(hidden_channels, heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_channels)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = hidden.shape
        query = hidden.flatten(2).transpose(1, 2)
        key = condition.flatten(2).transpose(1, 2)
        attended, _ = self.attention(
            self.query_norm(query),
            self.key_norm(key),
            self.key_norm(key),
            need_weights=False,
        )
        mixed = query + self.gate(torch.cat([query, attended], dim=-1)) * attended
        mixed = self.output_norm(mixed)
        return mixed.transpose(1, 2).reshape(batch, channels, height, width)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        condition_channels: int,
        base_channels: int,
        levels: int,
        *,
        use_time: bool = True,
    ) -> None:
        super().__init__()
        self.use_time = bool(use_time)
        channels = [base_channels * (2**level) for level in range(levels)]
        self.stem = nn.Conv2d(condition_channels, channels[0], 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                TimestepBlock(
                    channels[level],
                    channels[level],
                    base_channels * 4,
                )
                for level in range(levels)
            ]
        )
        self.downs = nn.ModuleList(
            [
                Downsample(channels[level], channels[level + 1])
                for level in range(levels - 1)
            ]
        )

    def forward(
        self,
        condition: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> list[torch.Tensor]:
        hidden = self.stem(condition)
        if not self.use_time:
            time_embedding = time_embedding * 0.0
        features: list[torch.Tensor] = []
        for level, block in enumerate(self.blocks):
            hidden = block(hidden, time_embedding)
            features.append(hidden)
            if level < len(self.downs):
                hidden = self.downs[level](hidden)
        return features
