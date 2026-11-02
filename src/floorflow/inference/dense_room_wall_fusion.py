"""Deterministic fusion of dense room classes with predicted wall evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.inference.room_repair import merge_room_fragments


@dataclass(frozen=True)
class DenseWallFusionResult:
    labels: np.ndarray
    mode: str
    wall_threshold: float | None
    wall_pixels: int
    boundary_pixels: int
    changed_room_pixels: int


def _wall_mask(
    wall_probability: np.ndarray,
    footprint: np.ndarray,
    threshold: float,
) -> np.ndarray:
    probability = np.asarray(wall_probability, dtype=np.float32)
    area = np.asarray(footprint, dtype=bool)
    if probability.shape != area.shape:
        raise ValueError("wall probability and footprint must share one shape")
    return area & (probability >= float(threshold))


def fuse_dense_room_walls(
    room_labels: np.ndarray,
    footprint: np.ndarray,
    wall_probability: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    mode: str = "predicted_wall",
    wall_threshold: float = 0.5,
    minimum_room_area: int = 32,
    owner_boundary: np.ndarray | None = None,
) -> DenseWallFusionResult:
    """Split dense room components with detected and/or derived wall seams."""

    labels = np.asarray(room_labels, dtype=np.int64).copy()
    area = np.asarray(footprint, dtype=bool)
    if labels.shape != area.shape:
        raise ValueError("room labels and footprint must share one shape")
    predicted = _wall_mask(wall_probability, area, wall_threshold)
    boundary = (
        partition_wall_mask(torch.from_numpy(labels)[None], radius=0)[0]
        .numpy()
        & area
        & (labels > 0)
    )
    if owner_boundary is not None:
        owner_mask = np.asarray(owner_boundary, dtype=bool)
        if owner_mask.shape != area.shape:
            raise ValueError("owner boundary and footprint must share one shape")
        boundary |= owner_mask & area
    if mode == "none":
        walls = np.zeros_like(area, dtype=bool)
    elif mode == "predicted_wall":
        walls = predicted
    elif mode == "boundary_wall":
        walls = boundary
    elif mode == "wall_gated":
        walls = boundary & predicted
    elif mode == "owner_boundary":
        if owner_boundary is None:
            raise ValueError("owner_boundary mode requires owner_boundary")
        walls = boundary
    elif mode == "owner_wall_gated":
        if owner_boundary is None:
            raise ValueError("owner_wall_gated mode requires owner_boundary")
        walls = boundary & predicted
    elif mode == "union":
        walls = predicted | boundary
    else:
        raise ValueError(f"unknown dense wall fusion mode: {mode}")
    before_rooms = np.isin(labels, np.asarray(taxonomy.room_ids))
    output = labels.copy()
    output[walls] = taxonomy.class_id("wall")
    output[~area] = 0
    output = merge_room_fragments(
        output,
        taxonomy,
        min_area=int(minimum_room_area),
    )
    after_rooms = np.isin(output, np.asarray(taxonomy.room_ids))
    return DenseWallFusionResult(
        labels=output,
        mode=mode,
        wall_threshold=(
            None if mode in {"none", "boundary_wall"} else float(wall_threshold)
        ),
        wall_pixels=int(predicted.sum()),
        boundary_pixels=int(boundary.sum()),
        changed_room_pixels=int(
            np.count_nonzero(before_rooms != after_rooms)
        ),
    )


__all__ = ["DenseWallFusionResult", "fuse_dense_room_walls"]
