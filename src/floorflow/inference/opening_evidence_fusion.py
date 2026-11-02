"""Fuse learned opening evidence into explicit-plan wall bands."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy


def fuse_opening_evidence(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    opening_prediction: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    wall_radius: int = 2,
) -> torch.Tensor:
    """Overlay typed openings only where they touch a predicted wall or opening."""

    if labels.shape != footprint.shape or labels.shape != opening_prediction.shape:
        raise ValueError("labels, footprint, and openings must share a shape")
    wall_id = taxonomy.class_id("wall")
    opening_ids = torch.tensor(taxonomy.opening_ids, device=labels.device)
    wall_or_opening = (labels == wall_id) | torch.isin(labels, opening_ids)
    kernel = 2 * max(0, int(wall_radius)) + 1
    band = F.max_pool2d(
        wall_or_opening.float()[:, None],
        kernel,
        stride=1,
        padding=max(0, int(wall_radius)),
    )[:, 0] > 0.5
    candidate = (opening_prediction != 0) & footprint.bool() & band
    return torch.where(candidate, opening_prediction, labels)


__all__ = ["fuse_opening_evidence"]
