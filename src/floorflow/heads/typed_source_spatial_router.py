"""Spatially dynamic, channel-family-aware routing over control sources."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

FAMILIES = ("rooms", "walls", "openings", "furniture")


class TypedSourceSpatialRouter(nn.Module):
    """Choose evidence sources independently for each CAD channel family."""

    def __init__(
        self,
        channels: int,
        source_count: int = 4,
        *,
        bottleneck: int = 16,
    ) -> None:
        super().__init__()
        channels = int(channels)
        source_count = int(source_count)
        bottleneck = max(4, int(bottleneck))
        self.channels = channels
        self.source_count = source_count
        self.bottleneck = bottleneck
        self.source_values = nn.ModuleList(
            [
                nn.Conv2d(channels, bottleneck, 1, bias=False)
                for _ in range(source_count)
            ]
        )
        self.context = nn.Sequential(
            nn.Conv2d(
                channels * (source_count + 1) + 1,
                bottleneck,
                1,
                bias=False,
            ),
            nn.GroupNorm(max(1, bottleneck // 8), bottleneck),
            nn.SiLU(),
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, bottleneck // 8), bottleneck),
            nn.SiLU(),
        )
        self.family_logits = nn.ModuleDict(
            {
                family: nn.Conv2d(bottleneck, source_count, 1)
                for family in FAMILIES
            }
        )
        self.family_values = nn.ModuleDict(
            {
                family: nn.Conv2d(bottleneck, channels, 1)
                for family in FAMILIES
            }
        )
        self.family_fusion = nn.Conv2d(
            channels * len(FAMILIES),
            channels,
            1,
        )
        self.scale = nn.Parameter(torch.tensor(0.1))
        prior = torch.tensor(
            [
                [2.0, 1.0, 0.2, 0.5],
                [0.5, 3.0, 0.2, 0.5],
                [0.3, 3.0, 0.2, 0.7],
                [0.8, 0.5, 2.0, 0.5],
            ],
            dtype=torch.float32,
        )[:, :source_count]
        self.register_buffer("source_prior", prior)
        for logits, values in zip(
            self.family_logits.values(),
            self.family_values.values(),
            strict=True,
        ):
            nn.init.normal_(logits.weight, std=0.01)
            nn.init.zeros_(logits.bias)
            nn.init.normal_(values.weight, std=0.01)
            nn.init.zeros_(values.bias)
        nn.init.normal_(self.family_fusion.weight, std=0.01)
        nn.init.zeros_(self.family_fusion.bias)

    def forward(
        self,
        sources: tuple[torch.Tensor, ...],
        context: torch.Tensor,
        footprint: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(sources) != self.source_count:
            raise ValueError("source count does not match router")
        if any(source.shape != sources[0].shape for source in sources):
            raise ValueError("all source features must share a shape")
        if context.shape != sources[0].shape:
            raise ValueError("context features must match source features")
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        footprint = footprint.float()
        if footprint.shape[-2:] != sources[0].shape[-2:]:
            footprint = F.interpolate(
                footprint,
                size=sources[0].shape[-2:],
                mode="nearest",
            )
        router = self.context(
            torch.cat([*sources, context, footprint], dim=1)
        )
        values = torch.stack(
            [projection(source) for projection, source in zip(
                self.source_values,
                sources,
                strict=True,
            )],
            dim=1,
        )
        family_weights = []
        family_deltas = []
        for family_index, family in enumerate(FAMILIES):
            logits = self.family_logits[family](router)
            weights = (
                logits
                + self.source_prior[family_index][None, :, None, None]
            ).softmax(dim=1)
            routed = (weights[:, :, None] * values).sum(dim=1)
            family_weights.append(weights)
            family_deltas.append(self.family_values[family](routed))
        weights = torch.stack(family_weights, dim=1)
        delta = self.family_fusion(torch.cat(family_deltas, dim=1))
        output = context + torch.tanh(self.scale).to(context.dtype) * delta
        entropy = -(
            weights.clamp_min(1.0e-8) * weights.clamp_min(1.0e-8).log()
        ).sum(dim=2)
        entropy = entropy / math.log(max(2, self.source_count))
        return output, {
            "source_weights": weights,
            "source_entropy": entropy.mean(),
            "source_usage": weights.mean(dim=(0, 3, 4)),
            "source_routed_delta": delta,
        }


__all__ = ["FAMILIES", "TypedSourceSpatialRouter"]
