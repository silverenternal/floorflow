"""Contour-inward CFM weighting and coarse-to-fine transport curriculum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from floorflow.data.taxonomy import FloorplanTaxonomy


def inside_distance(footprint: torch.Tensor) -> torch.Tensor:
    """Normalized distance from the exterior boundary to each footprint pixel."""

    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    masks = footprint.detach().cpu().numpy() > 0
    values = []
    for mask in masks:
        if not mask.any():
            values.append(np.zeros(mask.shape, dtype=np.float32))
            continue
        distance = distance_transform_edt(mask).astype(np.float32)
        scale = max(1.0, float(distance.max()))
        values.append(distance / scale)
    return torch.from_numpy(np.stack(values)).to(footprint.device)


def contour_inward_weight_map(
    footprint: torch.Tensor,
    progress: float,
    *,
    band_width: float = 0.22,
) -> torch.Tensor:
    """Weight a contour band that moves inward as training progresses."""

    distance = inside_distance(footprint)
    progress = min(1.0, max(0.0, float(progress)))
    width = max(1.0e-3, float(band_width))
    weight = torch.exp(-((distance - progress) / width).pow(2))
    mask = footprint if footprint.ndim == 3 else footprint[:, 0]
    return weight * mask.float()


def contour_inward_location(
    source: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    progress: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move interpolation toward the target from boundary to interior."""

    if source.shape != target.shape:
        raise ValueError("source and target shapes differ")
    weight = contour_inward_weight_map(footprint, progress)[:, None]
    location = source + weight * (target - source)
    velocity = (target - source) * weight
    return location, velocity


@dataclass(frozen=True)
class MultiStageWeights:
    room: float
    structure: float
    detail: float


def multi_stage_weights(
    progress: float,
    *,
    temperature: float = 0.22,
) -> MultiStageWeights:
    """Return room -> structure -> detail weights along training progress."""

    progress = min(1.0, max(0.0, float(progress)))
    centers = torch.tensor([0.18, 0.50, 0.82], dtype=torch.float32)
    distance = (torch.tensor(progress) - centers).abs()
    probability = torch.softmax(-distance / max(1.0e-4, temperature), dim=0)
    return MultiStageWeights(
        room=float(probability[0]),
        structure=float(probability[1]),
        detail=float(probability[2]),
    )


def stage_family_weights(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    weights: MultiStageWeights,
) -> torch.Tensor:
    """Build a per-pixel curriculum map for one semantic label raster."""

    result = torch.zeros_like(labels, dtype=torch.float32)
    room = torch.isin(
        labels,
        torch.tensor(taxonomy.room_ids, device=labels.device),
    )
    structure = labels == taxonomy.class_id("wall")
    detail = torch.isin(
        labels,
        torch.tensor(
            taxonomy.opening_ids + taxonomy.furniture_ids,
            device=labels.device,
        ),
    )
    result[room] = float(weights.room)
    result[structure] = float(weights.structure)
    result[detail] = float(weights.detail)
    return result


def stage_ordered_inward_weights(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    progress: float,
) -> torch.Tensor:
    """Combine semantic stage order with the moving contour band."""

    family = stage_family_weights(
        labels,
        taxonomy,
        multi_stage_weights(progress),
    )
    inward = contour_inward_weight_map(footprint, progress)
    if inward.ndim == 2:
        inward = inward[0]
    return family * inward


__all__ = [
    "MultiStageWeights",
    "contour_inward_location",
    "contour_inward_weight_map",
    "inside_distance",
    "multi_stage_weights",
    "stage_family_weights",
    "stage_ordered_inward_weights",
]
