"""Joint full-resolution room boundary and owner refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RoomBoundaryOwnerRefiner(nn.Module):
    """Refine query masks through a mutually exclusive owner map."""

    def __init__(
        self,
        feature_channels: int,
        num_queries: int,
        *,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        self.feature = nn.Sequential(
            nn.Conv2d(int(feature_channels), hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
        )
        self.owner_residual = nn.Conv2d(hidden, int(num_queries) + 1, 1)
        self.boundary = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.owner_residual.weight)
        nn.init.zeros_(self.owner_residual.bias)
        nn.init.zeros_(self.boundary.weight)
        nn.init.zeros_(self.boundary.bias)

    def forward(
        self,
        features: torch.Tensor,
        mask_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 4 or mask_logits.ndim != 4:
            raise ValueError("features and masks must both be [B,C,H,W]")
        hidden = self.feature(features.float())
        masks = F.interpolate(
            mask_logits.float(),
            size=hidden.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        owner_residual = self.owner_residual(hidden)
        refined_masks = masks + owner_residual[:, :-1]
        background = -refined_masks.amax(dim=1, keepdim=True)
        owner_logits = torch.cat(
            [refined_masks, background + owner_residual[:, -1:]],
            dim=1,
        )
        owner_probability = owner_logits.softmax(dim=1)
        foreground = owner_probability[:, :-1]
        background_probability = owner_probability[:, -1:]
        boundary_logits = self.boundary(hidden)[:, 0]
        return refined_masks, boundary_logits, {
            "owner_confidence": foreground.amax(dim=1).mean(),
            "owner_background_probability": background_probability.mean(),
            "owner_residual_norm": owner_residual.norm(dim=1).mean(),
        }


__all__ = ["RoomBoundaryOwnerRefiner"]
