"""Fuse a high-precision wall-centerline heatmap into explicit-plan labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy


def fuse_wall_centerline(
    labels: torch.Tensor,
    room_labels: torch.Tensor,
    footprint: torch.Tensor,
    wall_probability: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float = 0.5,
    dilation: int = 1,
    opening_protect_radius: int = 3,
    mode: str = "consensus",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use centerline evidence to remove off-line walls and bridge thin gaps."""

    if labels.shape != room_labels.shape or labels.shape != footprint.shape:
        raise ValueError("labels, room labels, and footprint must share a shape")
    wall_probability = F.interpolate(
        wall_probability[:, None].float(),
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    centerline = (wall_probability >= float(threshold)) & footprint.bool()
    radius = 2 * max(0, int(dilation)) + 1
    band = F.max_pool2d(
        centerline.float()[:, None],
        radius,
        stride=1,
        padding=max(0, int(dilation)),
    )[:, 0] > 0.5
    room_ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=labels.device)
    wall_id = taxonomy.class_id("wall")
    room_support = torch.isin(room_labels, room_ids)
    opening_support = torch.isin(labels, opening_ids)
    protect_radius = max(0, int(opening_protect_radius))
    protect_kernel = 2 * protect_radius + 1
    opening_band = F.max_pool2d(
        opening_support.float()[:, None],
        protect_kernel,
        stride=1,
        padding=protect_radius,
    )[:, 0] > 0.5
    existing = labels == wall_id
    allowed = ~opening_support
    if mode == "add":
        wall = existing | (centerline & room_support & allowed)
    elif mode == "consensus":
        wall = (
            (existing & (band | opening_band))
            | (centerline & room_support & allowed)
        )
    else:
        raise ValueError(f"unsupported wall-centerline fusion mode: {mode}")
    output = torch.where(wall, torch.full_like(labels, wall_id), labels)
    restore_room = existing & ~wall & room_support
    output = torch.where(restore_room, room_labels, output)
    removed = int((existing & ~wall).sum())
    added = int((wall & ~existing).sum())
    return output, {
        "wall_pixels_removed": float(removed),
        "wall_pixels_added": float(added),
        "wall_centerline_fraction": float(
            ((centerline & footprint.bool()).sum() / footprint.sum().clamp_min(1.0)).item()
        ),
    }


__all__ = ["fuse_wall_centerline"]
