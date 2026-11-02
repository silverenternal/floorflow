"""Deterministic room-region repair and fragment merge."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from floorflow.data.taxonomy import FloorplanTaxonomy


def merge_room_fragments(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    min_area: int = 8,
) -> np.ndarray:
    output = labels.copy()
    for class_id in taxonomy.room_ids:
        components, count = label(output == class_id)
        for component_id in range(1, count + 1):
            mask = components == component_id
            if mask.sum() >= min_area:
                continue
            _, indices = distance_transform_edt(~mask, return_indices=True)
            neighbor = output[tuple(indices)]
            candidates = neighbor[np.isin(neighbor, taxonomy.room_ids)]
            if candidates.size:
                output[mask] = np.bincount(candidates).argmax()
            else:
                output[mask] = 0
    return output


def close_room_partition(labels: np.ndarray, footprint: np.ndarray, taxonomy: FloorplanTaxonomy) -> np.ndarray:
    output = labels.copy()
    empty = (output == 0) & (footprint > 0)
    if not empty.any():
        return output
    _, indices = distance_transform_edt(empty, return_indices=True)
    nearest = output[tuple(indices)]
    output[empty] = nearest[empty]
    return output


def fill_room_support(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Assign only unlabeled footprint pixels to their nearest room class."""

    output = labels.copy()
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    rooms = np.isin(output, room_ids)
    support = footprint > 0 if support_mask is None else support_mask > 0
    empty = (output == 0) & support & (footprint > 0)
    if not rooms.any() or not empty.any():
        return output
    _, indices = distance_transform_edt(~rooms, return_indices=True)
    nearest = output[tuple(indices)]
    assign = empty & np.isin(nearest, room_ids)
    output[assign] = nearest[assign]
    return output


def drop_unsupported_openings(labels: np.ndarray, taxonomy: FloorplanTaxonomy, radius: int = 2) -> np.ndarray:
    output = labels.copy()
    wall = labels == taxonomy.class_id("wall")
    wall_near = binary_dilation(wall, iterations=max(1, radius))
    opening = np.isin(output, taxonomy.opening_ids) & ~wall_near
    output[opening] = 0
    return output


def drop_outside_furniture(labels: np.ndarray, taxonomy: FloorplanTaxonomy, radius: int = 2) -> np.ndarray:
    output = labels.copy()
    rooms = np.isin(labels, taxonomy.room_ids)
    room_near = binary_dilation(rooms, iterations=max(1, radius))
    furniture = np.isin(output, taxonomy.furniture_ids) & ~room_near
    output[furniture] = 0
    return output


def apply_floorplan_repairs(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    preserve_furniture: bool = True,
) -> np.ndarray:
    output = merge_room_fragments(labels, taxonomy)
    output = drop_unsupported_openings(output, taxonomy)
    if not preserve_furniture:
        output = drop_outside_furniture(output, taxonomy)
    return close_room_partition(output, footprint, taxonomy)
