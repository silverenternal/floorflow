"""Exact polygon readout for a dense labeled room partition."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as component_label

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.cell_polygons import component_cell_polygon
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    WallSkeletonPolicy,
    raster_labels_to_wall_plan,
)
from floorflow.heads.room_partition import build_room_partition_target
from floorflow.inference.room_repair import merge_room_fragments

_FOUR_CONNECTED = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    dtype=np.uint8,
)


def partition_polygon_rows(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    minimum_area: int = 32,
) -> list[dict]:
    """Read every 4-connected room component as an exact shared-boundary cell."""

    labels = merge_room_fragments(
        np.asarray(labels, dtype=np.int64),
        taxonomy,
        min_area=minimum_area,
    )
    footprint_mask = np.asarray(footprint, dtype=bool)
    dense_indices = build_room_partition_target(
        labels,
        footprint_mask,
        taxonomy,
        fill_occluded=True,
    )
    large_mask = np.zeros_like(footprint_mask, dtype=bool)
    for room_index in range(len(taxonomy.room_ids)):
        components, count = component_label(
            dense_indices == room_index,
            structure=_FOUR_CONNECTED,
        )
        if count == 0:
            continue
        sizes = np.bincount(components.ravel())
        accepted = np.flatnonzero(sizes >= int(minimum_area))
        accepted = accepted[accepted != 0]
        if accepted.size:
            large_mask |= np.isin(components, accepted)
    small_mask = (dense_indices >= 0) & ~large_mask
    if small_mask.any() and large_mask.any():
        nearest = distance_transform_edt(
            ~large_mask,
            return_distances=False,
            return_indices=True,
        )
        nearest_rows, nearest_columns = nearest[:, small_mask]
        dense_indices[small_mask] = dense_indices[nearest_rows, nearest_columns]
    rows = []
    for room_index, class_id in enumerate(taxonomy.room_ids):
        components, count = component_label(
            dense_indices == room_index,
            structure=_FOUR_CONNECTED,
        )
        for component_id in range(1, int(count) + 1):
            mask = components == component_id
            if int(mask.sum()) < int(minimum_area):
                continue
            polygon = component_cell_polygon(mask)
            if polygon.is_empty or polygon.area <= 0.0:
                continue
            rows.append(
                {
                    "label": taxonomy.class_names[int(class_id)],
                    "walls": [],
                    "polygon": np.asarray(
                        polygon.exterior.coords[:-1],
                        dtype=np.float64,
                    ).tolist(),
                    "holes": [
                        np.asarray(interior.coords[:-1], dtype=np.float64).tolist()
                        for interior in polygon.interiors
                    ],
                }
            )
    return rows


def explicit_partition_plan(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    minimum_area: int = 32,
    wall_policy: WallSkeletonPolicy | None = None,
) -> RasterWallPlan:
    """Keep the wall/opening readout and replace collapsed faces by room cells."""

    plan = raster_labels_to_wall_plan(
        labels,
        taxonomy,
        wall_policy=wall_policy,
    )
    rooms = partition_polygon_rows(
        labels,
        footprint,
        taxonomy,
        minimum_area=minimum_area,
    )
    if rooms:
        plan.rooms = rooms
        plan.room_source = "explicit_partition"
    return plan


__all__ = [
    "component_cell_polygon",
    "explicit_partition_plan",
    "partition_polygon_rows",
]
