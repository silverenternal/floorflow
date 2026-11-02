"""Assign dense room semantics to query-derived owner regions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class OwnerSemanticReassignment(nn.Module):
    """Use masks for ownership and dense pixels for semantic class evidence."""

    def __init__(
        self,
        *,
        support_threshold: float = 0.5,
        mask_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.support_threshold = float(support_threshold)
        self.mask_temperature = float(mask_temperature)

    def forward(
        self,
        dense_logits: torch.Tensor,
        mask_logits: torch.Tensor,
        footprint: torch.Tensor,
        query_existence_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if dense_logits.ndim != 4 or mask_logits.ndim != 4:
            raise ValueError("dense and mask logits must both be [B,C,H,W]")
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        size = dense_logits.shape[-2:]
        masks = F.interpolate(
            mask_logits.float() / max(self.mask_temperature, 1.0e-6),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        if query_existence_logits is not None:
            masks = masks * query_existence_logits.sigmoid()[..., None, None]
        owner = masks / masks.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        dense_probability = dense_logits.softmax(dim=1)
        pooled = torch.einsum(
            "bqhw,bchw->bqc",
            owner,
            dense_probability,
        )
        pooled = pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        reassigned = torch.einsum(
            "bqhw,bqc->bchw",
            owner,
            pooled,
        )
        support = (
            masks.max(dim=1).values >= self.support_threshold
        ) & footprint[:, 0].bool()
        probability = torch.where(
            support[:, None],
            reassigned,
            dense_probability,
        )
        probability = probability / probability.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        logits = probability.clamp_min(1.0e-8).log()
        return logits, {
            "support_fraction": support.float().mean(),
            "owner_confidence": masks.max(dim=1).values.mean(),
            "reassigned_mass": reassigned.sum(dim=1).mean(),
        }


__all__ = ["OwnerSemanticReassignment"]
