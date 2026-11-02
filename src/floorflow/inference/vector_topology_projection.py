"""Edit-local topology projection for floorplan label maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.vector_edit_protocol import VectorPlan, VectorRoom
from floorflow.heads.structured_objects import partition_wall_mask


@dataclass(frozen=True)
class TopologyProjection:
    labels: np.ndarray
    room_labels: np.ndarray
    changed_pixels: int
    outside_changed_pixels: int
    filled_room_pixels: int
    projected_wall_pixels: int


def _fill_room_holes(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    allowed: np.ndarray,
) -> tuple[np.ndarray, int]:
    output = labels.copy()
    room_seed = np.isin(output, np.asarray(taxonomy.room_ids))
    opening = np.isin(output, np.asarray(taxonomy.opening_ids))
    empty = (
        allowed
        & footprint
        & ~room_seed
        & ~opening
        & (output != taxonomy.class_id("wall"))
    )
    if not room_seed.any() or not empty.any():
        return output, 0
    indices = distance_transform_edt(
        ~room_seed,
        return_distances=False,
        return_indices=True,
    )
    nearest = output[tuple(indices)]
    output[empty] = nearest[empty]
    return output, int(empty.sum())


def _project_wall_seams(
    labels: np.ndarray,
    room_labels: np.ndarray,
    allowed: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> tuple[np.ndarray, int]:
    output = labels.copy()
    seam = (
        partition_wall_mask(
            torch.from_numpy(room_labels)[None],
            radius=1,
        )[0].numpy()
        & allowed
        & (room_labels > 0)
    )
    opening = np.isin(output, np.asarray(taxonomy.opening_ids))
    missing = seam & ~opening & (output != taxonomy.class_id("wall"))
    output[missing] = taxonomy.class_id("wall")
    return output, int(missing.sum())


def project_edit_topology(
    before_labels: np.ndarray,
    after_labels: np.ndarray,
    room_labels: np.ndarray,
    edit_region: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    fill_holes: bool = True,
    project_walls: bool = True,
) -> TopologyProjection:
    """Project topology only inside the declared edit region."""

    before = np.asarray(before_labels, dtype=np.int64)
    after = np.asarray(after_labels, dtype=np.int64)
    region = np.asarray(edit_region, dtype=bool) & np.asarray(footprint, dtype=bool)
    if before.shape != after.shape or before.shape != region.shape:
        raise ValueError("all label maps must share one shape")
    output = after.copy()
    output[~region] = before[~region]
    room_output = np.asarray(room_labels, dtype=np.int64).copy()
    filled = 0
    wall_pixels = 0
    if fill_holes:
        output, filled = _fill_room_holes(
            output,
            np.asarray(footprint, dtype=bool),
            taxonomy,
            region,
        )
        nearest_rooms = distance_transform_edt(
            room_output == 0,
            return_distances=False,
            return_indices=True,
        )
        room_filled = room_output[tuple(nearest_rooms)]
        room_output[region] = room_filled[region]
    if project_walls:
        output, wall_pixels = _project_wall_seams(
            output,
            room_output,
            region,
            taxonomy,
        )
    output[~region] = before[~region]
    outside = ~region
    return TopologyProjection(
        labels=output,
        room_labels=room_output,
        changed_pixels=int(np.count_nonzero(output != after)),
        outside_changed_pixels=int(
            np.count_nonzero(output[outside] != before[outside])
        ),
        filled_room_pixels=filled,
        projected_wall_pixels=wall_pixels,
    )


def connected_room_components(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> int:
    room_mask = np.isin(labels, np.asarray(taxonomy.room_ids))
    _components, count = label(room_mask)
    return int(count)


def _footprint_polygon(footprint: np.ndarray) -> Polygon:
    import cv2

    contours, _hierarchy = cv2.findContours(
        np.asarray(footprint, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        height, width = footprint.shape
        return Polygon([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)])
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        height, width = footprint.shape
        return Polygon([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)])
    return Polygon(contour[:, 0, :].astype(float)).buffer(0)


def _valid_polygon(value) -> Polygon | None:
    if value.is_empty:
        return None
    if isinstance(value, MultiPolygon):
        value = max(value.geoms, key=lambda item: item.area)
    if not isinstance(value, Polygon) or value.area <= 0.0:
        return None
    coordinates = np.rint(np.asarray(value.exterior.coords)).astype(float)
    candidate = Polygon(coordinates)
    return candidate if candidate.is_valid and candidate.area > 0.0 else None


def repair_edited_room_polygons(
    plan: VectorPlan,
    edit_region: np.ndarray,
    footprint: np.ndarray,
) -> dict:
    """Conservatively repair only edited room polygons."""

    region = np.asarray(edit_region, dtype=bool)
    footprint_polygon = _footprint_polygon(footprint)
    accepted = None
    rooms = list(plan.rooms)
    repaired_count = 0
    overlap_count = 0
    for index, room in enumerate(rooms):
        if not bool((room.mask & region).any()):
            continue
        repaired = _valid_polygon(room.polygon.buffer(0).intersection(footprint_polygon))
        if repaired is None:
            continue
        if accepted is not None:
            reduced = repaired.difference(accepted)
            if reduced.area < repaired.area - 1.0e-6:
                overlap_count += 1
            repaired = _valid_polygon(reduced)
            if repaired is None:
                continue
        accepted = repaired if accepted is None else unary_union([accepted, repaired])
        rooms[index] = VectorRoom(
            label=room.label,
            polygon=repaired,
            mask=room.mask,
        )
        repaired_count += 1
    return {
        "plan": VectorPlan(
            rooms=tuple(rooms),
            walls=plan.walls,
            openings=plan.openings,
            wall_mask=plan.wall_mask,
            opening_mask=plan.opening_mask,
            source=plan.source,
        ),
        "repaired_rooms": repaired_count,
        "overlap_adjustments": overlap_count,
    }


__all__ = [
    "TopologyProjection",
    "connected_room_components",
    "project_edit_topology",
    "repair_edited_room_polygons",
]
