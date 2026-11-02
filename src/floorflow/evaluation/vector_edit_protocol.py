"""FEVP-2 vector-level edit locality and preservation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.room_polygons import room_polygon_rows
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    extract_wall_segments,
    raster_labels_to_wall_plan,
)

POLYGON_EXACT_TOLERANCE = 0.5
WALL_ENDPOINT_TOLERANCE = 0.5
OPENING_CENTER_TOLERANCE = 1.0
ROOM_MATCH_IOU = 0.5


@dataclass(frozen=True)
class VectorRoom:
    label: str
    polygon: Polygon
    mask: np.ndarray


@dataclass(frozen=True)
class VectorWall:
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float
    wall_type: str

    @property
    def line(self) -> LineString:
        return LineString([self.start, self.end])

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


@dataclass(frozen=True)
class VectorOpening:
    center: tuple[float, float]
    width: float
    kind: str

    @property
    def point(self) -> Point:
        return Point(self.center)


@dataclass(frozen=True)
class VectorPlan:
    rooms: tuple[VectorRoom, ...]
    walls: tuple[VectorWall, ...]
    openings: tuple[VectorOpening, ...]
    wall_mask: np.ndarray
    opening_mask: np.ndarray
    source: RasterWallPlan


def _room_mask(
    labels: np.ndarray,
    polygon: Polygon,
) -> np.ndarray:
    import cv2

    mask = np.zeros(labels.shape, dtype=np.uint8)
    points = np.rint(np.asarray(polygon.exterior.coords[:-1])).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def extract_vector_plan(
    labels: np.ndarray,
    room_labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> VectorPlan:
    plan = raster_labels_to_wall_plan(
        labels,
        taxonomy,
        room_labels=room_labels,
    )
    rooms = []
    for row in room_polygon_rows(room_labels, taxonomy):
        polygon = Polygon(row["polygon"])
        if polygon.is_empty or polygon.area <= 0.0:
            continue
        rooms.append(
            VectorRoom(
                label=str(row["label"]),
                polygon=polygon,
                mask=_room_mask(room_labels, polygon),
            )
        )
    walls = tuple(
        VectorWall(
            start=wall.start,
            end=wall.end,
            thickness=float(wall.thickness),
            wall_type=str(wall.wall_type),
        )
        for wall in extract_wall_segments(
            labels,
            taxonomy,
            include_openings=False,
        )
    )
    openings = tuple(
        VectorOpening(
            center=opening.center,
            width=float(opening.width),
            kind=str(opening.kind),
        )
        for opening in plan.openings
    )
    return VectorPlan(
        rooms=tuple(rooms),
        walls=walls,
        openings=openings,
        wall_mask=np.asarray(labels) == taxonomy.class_id("wall"),
        opening_mask=np.isin(
            np.asarray(labels),
            np.asarray(taxonomy.opening_ids),
        ),
        source=plan,
    )


def _iou(left: Polygon, right: Polygon) -> float:
    union = left.union(right).area
    return float(left.intersection(right).area / union) if union > 0.0 else 0.0


def _room_mapping(
    before: tuple[VectorRoom, ...],
    after: tuple[VectorRoom, ...],
) -> dict[int, int]:
    if not before or not after:
        return {}
    similarity = np.zeros((len(before), len(after)), dtype=np.float64)
    for left, before_room in enumerate(before):
        for right, after_room in enumerate(after):
            if before_room.label == after_room.label:
                similarity[left, right] = _iou(
                    before_room.polygon,
                    after_room.polygon,
                )
    rows, columns = linear_sum_assignment(-similarity)
    return {
        int(row): int(column)
        for row, column in zip(rows, columns, strict=True)
        if similarity[row, column] >= ROOM_MATCH_IOU
    }


def _polygon_hausdorff(left: Polygon, right: Polygon) -> float:
    return float(left.boundary.hausdorff_distance(right.boundary))


def _wall_distance(left: VectorWall, right: VectorWall) -> float:
    direct = math.dist(left.start, right.start) + math.dist(left.end, right.end)
    reverse = math.dist(left.start, right.end) + math.dist(left.end, right.start)
    return min(direct, reverse) / 2.0


def _wall_mapping(
    before: tuple[VectorWall, ...],
    after: tuple[VectorWall, ...],
) -> dict[int, int]:
    if not before or not after:
        return {}
    cost = np.full((len(before), len(after)), 1.0e6, dtype=np.float64)
    for left, before_wall in enumerate(before):
        for right, after_wall in enumerate(after):
            if before_wall.wall_type != after_wall.wall_type:
                continue
            distance = _wall_distance(before_wall, after_wall)
            if (
                distance <= WALL_ENDPOINT_TOLERANCE
                and abs(before_wall.length - after_wall.length)
                <= WALL_ENDPOINT_TOLERANCE
            ):
                cost[left, right] = distance
    rows, columns = linear_sum_assignment(cost)
    return {
        int(row): int(column)
        for row, column in zip(rows, columns, strict=True)
        if cost[row, column] < 1.0e5
    }


def _opening_mapping(
    before: tuple[VectorOpening, ...],
    after: tuple[VectorOpening, ...],
) -> dict[int, int]:
    if not before or not after:
        return {}
    cost = np.full((len(before), len(after)), 1.0e6, dtype=np.float64)
    for left, before_opening in enumerate(before):
        for right, after_opening in enumerate(after):
            if before_opening.kind != after_opening.kind:
                continue
            distance = before_opening.point.distance(after_opening.point)
            if distance <= OPENING_CENTER_TOLERANCE:
                cost[left, right] = distance
    rows, columns = linear_sum_assignment(cost)
    return {
        int(row): int(column)
        for row, column in zip(rows, columns, strict=True)
        if cost[row, column] < 1.0e5
    }


def _unaffected_rooms(
    rooms: tuple[VectorRoom, ...],
    edit_region: np.ndarray,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, room in enumerate(rooms)
        if not bool((room.mask & edit_region).any())
    )


def _unaffected_walls(
    walls: tuple[VectorWall, ...],
    edit_region: np.ndarray,
) -> tuple[int, ...]:
    height, width = edit_region.shape
    return tuple(
        index
        for index, wall in enumerate(walls)
        if not _line_touches_mask(wall.line, edit_region, height, width)
    )


def _unaffected_openings(
    openings: tuple[VectorOpening, ...],
    edit_region: np.ndarray,
) -> tuple[int, ...]:
    height, width = edit_region.shape
    return tuple(
        index
        for index, opening in enumerate(openings)
        if not _circle_touches_mask(
            opening.center,
            max(1.0, opening.width / 2.0),
            edit_region,
            height,
            width,
        )
    )


def _circle_touches_mask(
    center: tuple[float, float],
    radius: float,
    mask: np.ndarray,
    height: int,
    width: int,
) -> bool:
    x, y = center
    left = max(0, math.floor(x - radius))
    right = min(width - 1, math.ceil(x + radius))
    top = max(0, math.floor(y - radius))
    bottom = min(height - 1, math.ceil(y + radius))
    yy, xx = np.mgrid[top : bottom + 1, left : right + 1]
    local = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    return bool((mask[top : bottom + 1, left : right + 1] & local).any())


def _point_in_mask(point: Point, mask: np.ndarray, height: int, width: int) -> bool:
    x = min(width - 1, max(0, round(point.x)))
    y = min(height - 1, max(0, round(point.y)))
    return bool(mask[y, x])


def _line_touches_mask(
    line: LineString,
    mask: np.ndarray,
    height: int,
    width: int,
) -> bool:
    samples = max(2, math.ceil(line.length) + 1)
    for fraction in np.linspace(0.0, 1.0, samples):
        point = line.interpolate(float(fraction), normalized=True)
        if _point_in_mask(point, mask, height, width):
            return True
    return False


def _area_violation(
    before: VectorPlan,
    after: VectorPlan,
    room_mapping: dict[int, int],
    wall_mapping: dict[int, int],
) -> float:
    area = 0.0
    total = 0.0
    for left, right in room_mapping.items():
        left_room = before.rooms[left]
        right_room = after.rooms[right]
        total += max(left_room.polygon.area, 1.0)
        area += left_room.polygon.symmetric_difference(right_room.polygon).area
    for left, right in wall_mapping.items():
        left_wall = before.walls[left]
        right_wall = after.walls[right]
        width = max(left_wall.thickness, right_wall.thickness, 1.0)
        left_buffer = left_wall.line.buffer(width / 2.0)
        right_buffer = right_wall.line.buffer(width / 2.0)
        total += max(left_buffer.area, 1.0)
        area += left_buffer.symmetric_difference(right_buffer).area
    return float(area / total) if total > 0.0 else 0.0


def _room_edges(
    rooms: tuple[VectorRoom, ...],
    indices: tuple[int, ...],
    mapping: dict[int, int] | None = None,
) -> set[tuple[int, int]]:
    edges = set()
    for left_position, left in enumerate(indices):
        for right in indices[left_position + 1 :]:
            if rooms[left].polygon.boundary.distance(
                rooms[right].polygon.boundary
            ) > 2.0:
                continue
            if mapping is not None and (
                left not in mapping or right not in mapping
            ):
                continue
            left_index = mapping[left] if mapping is not None else left
            right_index = mapping[right] if mapping is not None else right
            edges.add(tuple(sorted((left_index, right_index))))
    return edges


def _wall_support_exact(
    before: tuple[VectorWall, ...],
    before_indices: tuple[int, ...],
    after: tuple[VectorWall, ...],
    after_indices: tuple[int, ...],
) -> bool:
    if not before_indices and not after_indices:
        return True
    if not before_indices or not after_indices:
        return False
    before_geometry = unary_union([before[index].line for index in before_indices])
    after_geometry = unary_union([after[index].line for index in after_indices])
    return bool(
        before_geometry.buffer(WALL_ENDPOINT_TOLERANCE).covers(after_geometry)
        and after_geometry.buffer(WALL_ENDPOINT_TOLERANCE).covers(before_geometry)
    )


def vector_plan_preservation_metrics(
    before: VectorPlan,
    after: VectorPlan,
    edit_region: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    changed_mask: np.ndarray | None = None,
) -> dict:
    region = np.asarray(edit_region, dtype=bool)
    changed = (
        np.zeros(region.shape, dtype=bool)
        if changed_mask is None
        else np.asarray(changed_mask, dtype=bool)
    )
    protected_scope = region | changed
    wall_scope = binary_dilation(protected_scope, iterations=2)
    before_unaffected_rooms = _unaffected_rooms(before.rooms, protected_scope)
    after_unaffected_rooms = _unaffected_rooms(after.rooms, protected_scope)
    room_mapping = _room_mapping(before.rooms, after.rooms)
    wall_mapping = _wall_mapping(before.walls, after.walls)
    opening_mapping = _opening_mapping(before.openings, after.openings)
    before_unaffected_walls = _unaffected_walls(
        before.walls,
        wall_scope,
    )
    after_unaffected_walls = _unaffected_walls(
        after.walls,
        wall_scope,
    )
    before_unaffected_openings = _unaffected_openings(
        before.openings,
        protected_scope,
    )
    after_unaffected_openings = _unaffected_openings(
        after.openings,
        protected_scope,
    )
    room_violations = []
    for index in before_unaffected_rooms:
        right = room_mapping.get(index)
        if right is None:
            room_violations.append(index)
            continue
        if _polygon_hausdorff(
            before.rooms[index].polygon,
            after.rooms[right].polygon,
        ) > POLYGON_EXACT_TOLERANCE:
            room_violations.append(index)
    wall_violations = [
        index
        for index in before_unaffected_walls
        if index not in wall_mapping
    ]
    opening_violations = [
        index
        for index in before_unaffected_openings
        if index not in opening_mapping
    ]
    outside_extra_walls = [
        index
        for index in after_unaffected_walls
        if index not in set(wall_mapping.values())
    ]
    outside_extra_openings = [
        index
        for index in after_unaffected_openings
        if index not in set(opening_mapping.values())
    ]
    outside_room_polygon_exact = float(
        not room_violations
        and len(after_unaffected_rooms) == len(before_unaffected_rooms)
    )
    outside_wall_segment_support_exact = float(
        _wall_support_exact(
            before.walls,
            before_unaffected_walls,
            after.walls,
            after_unaffected_walls,
        )
    )
    outside_wall_exact = float(
        np.array_equal(
            before.wall_mask[~wall_scope],
            after.wall_mask[~wall_scope],
        )
    )
    outside_opening_detection_exact = float(
        not opening_violations
        and not outside_extra_openings
        and len(after_unaffected_openings) == len(before_unaffected_openings)
    )
    opening_scope = binary_dilation(protected_scope, iterations=1)
    outside_opening_exact = float(
        np.array_equal(
            before.opening_mask[~opening_scope],
            after.opening_mask[~opening_scope],
        )
    )
    before_edges = _room_edges(before.rooms, before_unaffected_rooms)
    mapped_before_edges = _room_edges(
        before.rooms,
        before_unaffected_rooms,
        {
            index: room_mapping[index]
            for index in before_unaffected_rooms
            if index in room_mapping
        },
    )
    after_edges = _room_edges(after.rooms, after_unaffected_rooms)
    outside_edge_exact = float(
        len(mapped_before_edges) == len(before_edges)
        and mapped_before_edges == after_edges
    )
    outside_room_label_exact = float(
        all(
            before.rooms[index].label == after.rooms[right].label
            for index, right in room_mapping.items()
            if index in before_unaffected_rooms
        )
        and all(
            index in room_mapping for index in before_unaffected_rooms
        )
    )
    locality_violation_area = _area_violation(
        before,
        after,
        {
            index: room_mapping[index]
            for index in before_unaffected_rooms
            if index in room_mapping
        },
        {
            index: wall_mapping[index]
            for index in before_unaffected_walls
            if index in wall_mapping
        },
    )
    return {
        "outside_room_label_exact": outside_room_label_exact,
        "outside_room_polygon_exact": outside_room_polygon_exact,
        "outside_wall_exact": outside_wall_exact,
        "outside_wall_segment_support_exact": (
            outside_wall_segment_support_exact
        ),
        "outside_opening_exact": outside_opening_exact,
        "outside_opening_detection_exact": (
            outside_opening_detection_exact
        ),
        "outside_edge_exact": outside_edge_exact,
        "outside_room_level_exact": float(
            outside_room_label_exact
            and outside_room_polygon_exact
            and outside_edge_exact
        ),
        "locality_violation_area": locality_violation_area,
        "denominators": {
            "before_rooms": len(before.rooms),
            "after_rooms": len(after.rooms),
            "before_walls": len(before.walls),
            "after_walls": len(after.walls),
            "before_openings": len(before.openings),
            "after_openings": len(after.openings),
            "before_unaffected_rooms": len(before_unaffected_rooms),
            "after_unaffected_rooms": len(after_unaffected_rooms),
            "before_unaffected_walls": len(before_unaffected_walls),
            "after_unaffected_walls": len(after_unaffected_walls),
            "before_unaffected_openings": len(before_unaffected_openings),
            "after_unaffected_openings": len(after_unaffected_openings),
            "room_violations": len(room_violations),
            "wall_violations": len(wall_violations) + len(outside_extra_walls),
            "opening_violations": len(opening_violations)
            + len(outside_extra_openings),
        },
    }


def vector_preservation_metrics(
    before_labels: np.ndarray,
    before_room_labels: np.ndarray,
    after_labels: np.ndarray,
    after_room_labels: np.ndarray,
    edit_region: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict:
    before = extract_vector_plan(before_labels, before_room_labels, taxonomy)
    after = extract_vector_plan(after_labels, after_room_labels, taxonomy)
    return vector_plan_preservation_metrics(
        before,
        after,
        edit_region,
        taxonomy,
        changed_mask=np.asarray(before_labels) != np.asarray(after_labels),
    )


__all__ = [
    "VectorOpening",
    "VectorPlan",
    "VectorRoom",
    "VectorWall",
    "extract_vector_plan",
    "vector_plan_preservation_metrics",
    "vector_preservation_metrics",
]
