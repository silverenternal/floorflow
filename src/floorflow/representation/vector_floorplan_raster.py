"""Derived raster diagnostics from a vector-first canonical floorplan."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from shapely.geometry import Polygon
from shapely.ops import unary_union

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    canonical_fingerprint,
)


def render_floorplan_rasters(
    plan: CanonicalFloorplan,
    taxonomy: FloorplanTaxonomy,
    *,
    strict: bool = True,
) -> dict[str, object]:
    """Render diagnostics while separating area overlap from boundary collision."""

    polygons = [
        (room.id, Polygon(room.polygon, room.holes))
        for room in plan.rooms
    ]
    area_overlap = []
    for index, (left_id, left) in enumerate(polygons):
        for right_id, right in polygons[index + 1 :]:
            overlap = float(left.intersection(right).area)
            if overlap > 0.0:
                area_overlap.append((left_id, right_id, overlap))
    if strict and area_overlap:
        raise ValueError(f"vector rooms overlap: {area_overlap}")

    labels = np.zeros((plan.height, plan.width), dtype=np.uint8)
    occupancy = np.zeros((plan.height, plan.width), dtype=np.uint8)
    wall_labels = np.zeros_like(labels)
    opening_labels = np.zeros_like(labels)
    furniture_labels = np.zeros_like(labels)
    overlap_pixels = 0
    room_geometry = unary_union(
        [polygon.buffer(0) for _room_id, polygon in polygons]
    ) if polygons else Polygon()
    geometry_components = (
        list(room_geometry.geoms)
        if hasattr(room_geometry, "geoms")
        else [room_geometry]
    )
    vector_is_hole_free = not any(
        component.interiors
        for component in geometry_components
        if not component.is_empty and component.geom_type == "Polygon"
    )
    for room in sorted(plan.rooms, key=lambda item: item.id):
        room_mask = np.zeros_like(labels)
        cv2.fillPoly(
            room_mask,
            [np.rint(np.asarray(room.polygon)).astype(np.int32)],
            color=1,
        )
        for hole in room.holes:
            cv2.fillPoly(
                room_mask,
                [np.rint(np.asarray(hole)).astype(np.int32)],
                color=0,
            )
        overlap = int(np.count_nonzero(room_mask & occupancy))
        overlap_pixels += overlap
        occupancy |= room_mask
        labels[room_mask.astype(bool)] = taxonomy.class_id(room.label)
    raster_gaps = (
        binary_fill_holes(occupancy.astype(bool)) & ~occupancy.astype(bool)
        if vector_is_hole_free and occupancy.any()
        else np.zeros_like(occupancy, dtype=bool)
    )
    if raster_gaps.any() and occupancy.any():
        _distances, indices = distance_transform_edt(
            ~occupancy.astype(bool),
            return_indices=True,
        )
        nearest_rows = indices[0][raster_gaps]
        nearest_columns = indices[1][raster_gaps]
        labels[raster_gaps] = labels[nearest_rows, nearest_columns]
        occupancy[raster_gaps] = 1
    wall_id = taxonomy.class_id("wall")
    for wall in plan.walls:
        cv2.line(
            wall_labels,
            tuple(np.rint(wall.start).astype(int)),
            tuple(np.rint(wall.end).astype(int)),
            wall_id,
            thickness=max(1, round(wall.thickness)),
        )
    wall_by_id = {wall.id: wall for wall in plan.walls}
    for opening in plan.openings:
        wall = wall_by_id.get(opening.wall_id)
        if wall is None:
            continue
        delta = np.asarray(wall.end) - np.asarray(wall.start)
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        direction = delta / length
        center = np.asarray(opening.center, dtype=np.float64)
        half = 0.5 * float(opening.width)
        start = np.rint(center - half * direction).astype(np.int32)
        end = np.rint(center + half * direction).astype(np.int32)
        cv2.line(
            opening_labels,
            tuple(start),
            tuple(end),
            taxonomy.class_id(opening.label),
            thickness=max(1, round(float(wall.thickness) + 2.0)),
        )
    for item in plan.furniture:
        polygon = np.rint(np.asarray(item.polygon)).astype(np.int32)
        cv2.fillPoly(
            furniture_labels,
            [polygon],
            taxonomy.class_id(item.label),
        )
    footprint = np.zeros_like(labels)
    cv2.fillPoly(
        footprint,
        [np.rint(np.asarray(plan.footprint)).astype(np.int32)],
        color=1,
    )
    outside = int(
        np.count_nonzero(labels.astype(bool) & ~footprint.astype(bool))
    )
    opening_outside = int(
        np.count_nonzero(opening_labels.astype(bool) & ~footprint.astype(bool))
    )
    furniture_outside = int(
        np.count_nonzero(
            furniture_labels.astype(bool) & ~footprint.astype(bool)
        )
    )
    visible = labels.copy()
    visible = np.where(furniture_labels != 0, furniture_labels, visible)
    visible = np.where(wall_labels != 0, wall_labels, visible)
    visible = np.where(opening_labels != 0, opening_labels, visible)
    visible = np.where(footprint, visible, 0)
    return {
        "room_labels": labels,
        "wall_labels": wall_labels,
        "opening_labels": opening_labels,
        "furniture_labels": furniture_labels,
        "visible_labels": visible,
        "footprint": footprint.astype(bool),
        "room_overlap_pixels": sum(item[2] for item in area_overlap),
        "room_boundary_collision_pixels": overlap_pixels,
        "outside_footprint_pixels": outside,
        "opening_outside_footprint_pixels": opening_outside,
        "furniture_outside_footprint_pixels": furniture_outside,
        "opening_wall_host_errors": sum(
            opening.wall_id not in wall_by_id for opening in plan.openings
        ),
        "fingerprint": canonical_fingerprint(plan),
    }


__all__ = ["render_floorplan_rasters"]
