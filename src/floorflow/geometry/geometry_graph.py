"""Build wall-first plans from geometry-cache wall skeletons and faces."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import label as component_label
from scipy.ndimage import binary_fill_holes
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from skimage.measure import find_contours

from floorflow.data.geometry_cache import (
    FloorplanGeometry,
    geometry_from_payload,
    load_geometry_cache,
)
from floorflow.heads.room_partition import build_room_partition_target
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    raster_labels_to_wall_plan,
)


def _assign_walls(polygon: Polygon, plan: RasterWallPlan) -> list[str]:
    boundary = polygon.boundary.buffer(2.0)
    return [
        f"wall_{index + 1}"
        for index, wall in enumerate(plan.walls)
        if LineString([wall.start, wall.end]).intersection(boundary).length
        >= min(3.0, 0.25 * math.dist(wall.start, wall.end))
    ]


def dense_room_polygons(
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    min_area: float,
) -> list[tuple[int, Polygon]]:
    dense = build_room_partition_target(target, footprint, taxonomy)
    polygons = []
    for local_index, class_id in enumerate(taxonomy.room_ids):
        components, count = component_label(dense == local_index)
        for component_id in range(1, int(count) + 1):
            mask = binary_fill_holes(components == component_id)
            if mask.sum() < min_area:
                continue
            padded = np.pad(mask.astype(np.float32), 1)
            contours = find_contours(padded, 0.5)
            if not contours:
                continue
            points = max(contours, key=len) - 1.0
            polygon = Polygon([(float(x), float(y)) for y, x in points])
            if polygon.is_valid and polygon.area >= min_area:
                polygons.append((int(class_id), polygon))
    return polygons


def _tiling_is_valid(polygons: list[Polygon], min_area: float) -> bool:
    if not polygons:
        return False
    for index, left in enumerate(polygons):
        for right in polygons[index + 1 :]:
            if left.intersection(right).area > min_area:
                return False
    return unary_union(polygons).geom_type == "Polygon"


def wall_plan_from_geometry(
    geometry: FloorplanGeometry,
    taxonomy: FloorplanTaxonomy,
) -> RasterWallPlan:
    """Use wall-graph faces, with dense reconciliation only for invalid tiling."""

    plan = raster_labels_to_wall_plan(geometry.target, taxonomy)
    min_area = geometry.target.size * 0.0004
    polygons = [Polygon(room["polygon"]) for room in plan.rooms]
    if _tiling_is_valid(polygons, min_area):
        return plan
    rooms = [
        {
            "label": taxonomy.class_names[class_id],
            "walls": _assign_walls(polygon, plan),
            "polygon": [
                [float(x), float(y)]
                for x, y in polygon.exterior.coords[:-1]
            ],
        }
        for class_id, polygon in dense_room_polygons(
            geometry.target,
            geometry.footprint,
            taxonomy,
            min_area,
        )
    ]
    return RasterWallPlan(
        frame=plan.frame,
        walls=plan.walls,
        rooms=rooms,
        openings=plan.openings,
        room_source="dense_reconciled_faces",
    )


def dense_room_plan(
    room_labels: np.ndarray,
    footprint: np.ndarray,
    wall_plan: RasterWallPlan,
    taxonomy: FloorplanTaxonomy,
) -> RasterWallPlan:
    min_area = room_labels.size * 0.0004
    rooms = [
        {
            "label": taxonomy.class_names[class_id],
            "walls": _assign_walls(polygon, wall_plan),
            "polygon": [
                [float(x), float(y)]
                for x, y in polygon.exterior.coords[:-1]
            ],
        }
        for class_id, polygon in dense_room_polygons(
            room_labels,
            footprint,
            taxonomy,
            min_area,
        )
    ]
    return RasterWallPlan(
        frame=wall_plan.frame,
        walls=wall_plan.walls,
        rooms=rooms,
        openings=wall_plan.openings,
        room_source="dense_room_support",
    )


def wall_plan_from_cache(
    path,
    taxonomy: FloorplanTaxonomy,
) -> RasterWallPlan:
    geometry = geometry_from_payload(load_geometry_cache(path))
    return wall_plan_from_geometry(geometry, taxonomy)
