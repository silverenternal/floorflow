"""Local full-resolution feature evidence for room-query classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FullResolutionClassSampler(nn.Module):
    """Sample decoder features at each soft room-mask centroid."""

    def __init__(
        self,
        query_dim: int,
        feature_channels: int,
        num_room_classes: int,
        *,
        hidden_dim: int = 64,
        pool_size: int = 128,
    ) -> None:
        super().__init__()
        self.pool_size = int(pool_size)
        self.feature_projection = nn.Conv2d(
            int(feature_channels),
            int(query_dim),
            1,
        )
        self.evidence = nn.Sequential(
            nn.Linear(int(query_dim) * 2 + 3, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.output = nn.Linear(int(hidden_dim), int(num_room_classes) + 1)
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
        if query_features.ndim != 3 or mask_logits.ndim != 4:
            raise ValueError("query features and masks must be [B,Q,D]/[B,Q,H,W]")
        if class_logits.shape[:2] != query_features.shape[:2]:
            raise ValueError("class logits and query features must align")
        if features.ndim != 4 or features.shape[0] != query_features.shape[0]:
            raise ValueError("features must be [B,C,H,W]")
        size = (
            min(self.pool_size, features.shape[-2]),
            min(self.pool_size, features.shape[-1]),
        )
        projected = F.adaptive_avg_pool2d(
            self.feature_projection(features.float()),
            size,
        )
        probability = F.interpolate(
            mask_logits.float().sigmoid(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        mass = probability.sum(dim=(2, 3))
        normalized = probability / mass[..., None, None].clamp_min(1.0e-6)
        x_coordinate = torch.linspace(
            -1.0,
            1.0,
            size[1],
            device=features.device,
            dtype=projected.dtype,
        )
        y_coordinate = torch.linspace(
            -1.0,
            1.0,
            size[0],
            device=features.device,
            dtype=projected.dtype,
        )
        centroid_x = (normalized.sum(dim=2) * x_coordinate).sum(dim=2)
        centroid_y = (normalized.sum(dim=3) * y_coordinate).sum(dim=2)
        grid = torch.stack([centroid_x, centroid_y], dim=-1).reshape(
            query_features.shape[0],
            query_features.shape[1],
            1,
            2,
        )
        local = F.grid_sample(
            projected,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[:, :, :, 0].transpose(1, 2)
        peak = probability.amax(dim=(2, 3))
        area = (probability >= 0.5).float().mean(dim=(2, 3))
        stats = torch.stack(
            [
                mass / float(size[0] * size[1]),
                peak,
                area,
            ],
            dim=-1,
        )
        evidence = torch.cat(
            [query_features.float(), local, stats],
            dim=-1,
        )
        residual = self.output(self.evidence(evidence))
        residual = torch.tanh(self.residual_scale.to(residual.dtype)) * residual
        return class_logits + residual, {
            "residual_norm": residual.norm(dim=-1).mean(),
            "local_feature_norm": local.norm(dim=-1).mean(),
            "mask_mass": mass.mean(),
            "centroid_x_abs": centroid_x.abs().mean(),
            "centroid_y_abs": centroid_y.abs().mean(),
        }


__all__ = ["FullResolutionClassSampler"]
