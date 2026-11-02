"""Interior-only wall head, loss, and metrics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from floorflow.geometry.core import footprint_band
from floorflow.data.taxonomy import FloorplanTaxonomy


class InteriorWallHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        hidden = max(32, int(in_channels) // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


def interior_region(footprint: torch.Tensor, boundary_width: int = 7) -> torch.Tensor:
    footprint_bchw = footprint[:, None] if footprint.ndim == 3 else footprint
    band = footprint_band(footprint_bchw, boundary_width)
    return (footprint_bchw - band).clamp(0.0, 1.0)[:, 0].bool()


def interior_wall_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    target_dilation: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    interior = interior_region(footprint)
    expected = (target == taxonomy.class_id("wall")).float() * interior.float()
    if target_dilation > 0:
        expected = F.max_pool2d(
            expected[:, None],
            2 * int(target_dilation) + 1,
            stride=1,
            padding=int(target_dilation),
        )[:, 0] * interior.float()
    positive = expected.sum().clamp_min(1.0)
    negative = (interior & ~expected.bool()).sum().clamp_min(1.0).float()
    bce = F.binary_cross_entropy_with_logits(
        logits,
        expected,
        pos_weight=(negative / positive).clamp(1.0, 50.0),
        reduction="none",
    )
    bce = bce[interior].mean()
    probability = torch.sigmoid(logits) * interior.float()
    true_positive = (probability * expected).sum()
    false_negative = ((1.0 - probability) * expected).sum()
    false_positive = (probability * (interior.float() - expected)).sum()
    tversky = 1.0 - (true_positive + 1.0) / (
        true_positive + 4.0 * false_negative + false_positive + 1.0
    )
    return bce + tversky, bce.detach(), tversky.detach()


@torch.no_grad()
def interior_wall_iou(
    logits: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    threshold: float = 0.0,
) -> float:
    interior = interior_region(footprint)
    predicted = (logits > threshold) & interior
    expected = (target == taxonomy.class_id("wall")) & interior
    return float((predicted & expected).sum().float() / (predicted | expected).sum().clamp_min(1).float())
