"""Coarse room-layout control logits for conditional generation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy


def coarse_room_layout_logits(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    size: int = 64,
) -> torch.Tensor:
    """Downsample a room layout, then upsample it as a soft control prior."""

    if labels.ndim != 3:
        raise ValueError("labels must be [B,H,W]")
    size = max(2, int(size))
    height, width = labels.shape[-2:]
    room_ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    one_hot = torch.stack(
        [(labels == class_id).float() for class_id in room_ids],
        dim=1,
    )
    occupied = one_hot.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    coarse = F.adaptive_max_pool2d(one_hot, (size, size))
    coarse_occupied = F.adaptive_max_pool2d(occupied, (size, size))
    coarse_labels = coarse.argmax(dim=1)
    coarse_onehot = F.one_hot(
        coarse_labels,
        len(room_ids),
    ).permute(0, 3, 1, 2).float()
    coarse_onehot = coarse_onehot * coarse_occupied
    logits = coarse_onehot * 2.0 - coarse_occupied
    return F.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )


__all__ = ["coarse_room_layout_logits"]
