"""Geometry diagnostics shared by room-query objectives."""

from __future__ import annotations

import torch


def mask_centroid(masks: torch.Tensor) -> torch.Tensor:
    """Return normalized mask centroids for a [Q,H,W] tensor."""

    if masks.ndim != 3:
        raise ValueError("masks must be [Q,H,W]")
    coordinates = torch.linspace(
        0.0,
        1.0,
        masks.shape[-1],
        device=masks.device,
        dtype=masks.dtype,
    )
    y_grid, x_grid = torch.meshgrid(
        coordinates,
        coordinates,
        indexing="ij",
    )
    mass = masks.sum(dim=(1, 2)).clamp_min(1.0e-6)
    return torch.stack(
        [
            (masks * x_grid).sum(dim=(1, 2)) / mass,
            (masks * y_grid).sum(dim=(1, 2)) / mass,
        ],
        dim=-1,
    )


def matched_mask_centroid_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return per-query normalized centroid distance for matched masks."""

    if prediction.shape != target.shape:
        raise ValueError("matched masks must share a shape")
    return (
        mask_centroid(prediction) - mask_centroid(target)
    ).norm(dim=-1)


def reference_anchor_loss(
    reference_points: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Match the mean learned query anchor to a target-mask centroid."""

    if reference_points.ndim != 3 or reference_points.shape[-1] != 2:
        raise ValueError("reference points must be [Q,P,2]")
    return torch.nn.functional.smooth_l1_loss(
        reference_points.mean(dim=1),
        mask_centroid(target_mask),
        reduction="none",
    ).mean(dim=-1)


__all__ = [
    "mask_centroid",
    "matched_mask_centroid_error",
    "reference_anchor_loss",
]
