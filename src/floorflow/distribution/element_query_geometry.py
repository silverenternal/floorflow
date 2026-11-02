"""Standalone query-anchor and mask-diagnostic helpers for element heads."""

from __future__ import annotations

import torch


def mask_diagnostics(masks: torch.Tensor) -> dict[str, torch.Tensor]:
    probability = masks.sigmoid()
    flat = probability.flatten(2)
    intersection = flat @ flat.transpose(1, 2)
    denominator = (
        flat.sum(dim=2)[:, :, None]
        + flat.sum(dim=2)[:, None]
        - intersection
    )
    pairwise = intersection / denominator.clamp_min(1.0e-6)
    off_diagonal = ~torch.eye(
        masks.shape[1],
        dtype=torch.bool,
        device=masks.device,
    )
    return {
        "mask_pairwise_iou": pairwise[:, off_diagonal].mean()
        if masks.shape[1] > 1
        else masks.sum() * 0.0,
        "mask_positive_fraction": (probability >= 0.5).float().mean(),
        "mask_mean_probability": probability.mean(),
        "mask_logit_std": masks.std(),
    }


def query_anchor_maps(count: int, grid: int) -> torch.Tensor:
    """Initialize one smooth spatial anchor per learned query."""

    columns = max(1, int(count**0.5))
    rows = max(1, (count + columns - 1) // columns)
    y = torch.linspace(0.0, 1.0, grid)
    x = torch.linspace(0.0, 1.0, grid)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    anchors = torch.empty(count, grid, grid)
    for index in range(count):
        center_x = (index % columns + 0.5) / columns
        center_y = (index // columns + 0.5) / rows
        distance = (xx - center_x).square() + (yy - center_y).square()
        anchors[index] = 2.0 * torch.exp(-distance / (2.0 * 0.16**2))
    return anchors


__all__ = ["mask_diagnostics", "query_anchor_maps"]
