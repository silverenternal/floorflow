"""Mask-region evidence fusion for room-query class prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskPooledClassFusion(nn.Module):
    """Add class evidence pooled inside each predicted query mask.

    The residual projection is zero-initialized, so enabling the module starts
    from the exact existing class logits. Features and masks are pooled to a
    bounded resolution to avoid a full-resolution B x Q x H x W allocation.
    """

    def __init__(
        self,
        query_dim: int,
        feature_channels: int,
        num_room_classes: int,
        *,
        hidden_dim: int = 64,
        pool_size: int = 64,
    ) -> None:
        super().__init__()
        self.num_room_classes = int(num_room_classes)
        self.pool_size = max(1, int(pool_size))
        query_dim = int(query_dim)
        self.feature_projection = nn.Conv2d(
            int(feature_channels),
            query_dim,
            1,
        )
        self.evidence = nn.Sequential(
            nn.Linear(query_dim * 2 + 3, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.output = nn.Linear(
            int(hidden_dim),
            self.num_room_classes + 1,
        )
        self.residual_scale = nn.Parameter(torch.full((1,), 0.1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        query_features: torch.Tensor,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if query_features.ndim != 3:
            raise ValueError("query_features must be [B,Q,D]")
        if mask_logits.ndim != 4:
            raise ValueError("mask_logits must be [B,Q,H,W]")
        if class_logits.shape[:2] != query_features.shape[:2]:
            raise ValueError("class logits and query features must align")
        if features.ndim != 4 or features.shape[0] != mask_logits.shape[0]:
            raise ValueError("features must be [B,C,H,W] matching mask batch")
        spatial_size = tuple(
            min(self.pool_size, dimension)
            for dimension in features.shape[-2:]
        )
        projected = F.adaptive_avg_pool2d(
            self.feature_projection(features.float()),
            spatial_size,
        )
        probability = F.adaptive_avg_pool2d(
            mask_logits.float().sigmoid(),
            spatial_size,
        )
        weights = probability / probability.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(1.0e-6)
        pooled = torch.einsum(
            "bqhw,bchw->bqc",
            weights,
            projected,
        )
        mass = probability.mean(dim=(2, 3))
        peak = probability.amax(dim=(2, 3))
        area = (probability >= 0.5).float().mean(dim=(2, 3))
        evidence = torch.cat(
            [
                query_features.float(),
                pooled,
                torch.stack([mass, peak, area], dim=-1),
            ],
            dim=-1,
        )
        residual = self.output(self.evidence(evidence))
        residual = (
            torch.tanh(self.residual_scale.to(residual.dtype))
            * residual
        )
        return class_logits + residual, {
            "residual_norm": residual.norm(dim=-1).mean(),
            "pooled_feature_norm": pooled.norm(dim=-1).mean(),
            "mask_mass": mass.mean(),
        }


__all__ = ["MaskPooledClassFusion"]
