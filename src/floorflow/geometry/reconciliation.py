"""Deterministic plan reconciliation and output-level wall fusion."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutOpening,
    ReadoutWall,
    _wall_faces,
)


def _cluster_points(
    walls: list[ReadoutWall],
    tolerance: float,
) -> list[ReadoutWall]:
    endpoints = [point for wall in walls for point in (wall.start, wall.end)]
    clusters: list[list[tuple[float, float]]] = []
    for point in endpoints:
        for cluster in clusters:
            center = (
                sum(item[0] for item in cluster) / len(cluster),
                sum(item[1] for item in cluster) / len(cluster),
            )
            if math.dist(point, center) <= tolerance:
                cluster.append(point)
                break
        else:
            clusters.append([point])
    mapping = {}
    for cluster in clusters:
        center = (
            sum(item[0] for item in cluster) / len(cluster),
            sum(item[1] for item in cluster) / len(cluster),
        )
        for point in cluster:
            mapping[point] = center
    return [
        ReadoutWall(
            start=mapping[wall.start],
            end=mapping[wall.end],
            thickness=wall.thickness,
            wall_type=wall.wall_type,
        )
        for wall in walls
        if math.dist(mapping[wall.start], mapping[wall.end]) > tolerance * 0.25
    ]


def _duplicate(
    left: ReadoutWall,
    right: ReadoutWall,
    tolerance: float,
) -> bool:
    direct = max(
        math.dist(left.start, right.start),
        math.dist(left.end, right.end),
    )
    reverse = max(
        math.dist(left.start, right.end),
        math.dist(left.end, right.start),
    )
    return min(direct, reverse) <= tolerance


def _deduplicate_walls(
    walls: list[ReadoutWall],
    tolerance: float,
) -> list[ReadoutWall]:
    unique = []
    for wall in walls:
        if not any(_duplicate(wall, kept, tolerance) for kept in unique):
            unique.append(wall)
    return unique


def _noded_segments(
    walls: list[ReadoutWall],
    tolerance: float,
) -> list[ReadoutWall]:
    if not walls:
        return []
    lines = [
        LineString([wall.start, wall.end])
        for wall in walls
        if math.dist(wall.start, wall.end) > tolerance * 0.25
    ]
    merged = unary_union(lines)
    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    segments = []
    for part in parts:
        if part.geom_type != "LineString":
            continue
        coordinates = list(part.coords)
        for start, end in pairwise(coordinates):
            if math.dist(start, end) > tolerance * 0.25:
                segments.append(
                    ReadoutWall(
                        start=(float(start[0]), float(start[1])),
                        end=(float(end[0]), float(end[1])),
                        thickness=3.0,
                    )
                )
    return _deduplicate_walls(segments, tolerance)


def _unit_direction(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    length = math.dist(start, end)
    if length <= 1.0e-9:
        return (1.0, 0.0)
    return ((end[0] - start[0]) / length, (end[1] - start[1]) / length)


def _direction_angle(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    cosine = abs(left[0] * right[0] + left[1] * right[1])
    return math.degrees(math.acos(max(0.0, min(1.0, cosine))))


def _plausible_bridge(
    bridge: tuple[float, float],
    left: ReadoutWall,
    right: ReadoutWall,
) -> bool:
    left_dir = _unit_direction(left.start, left.end)
    right_dir = _unit_direction(right.start, right.end)
    left_angle = _direction_angle(bridge, left_dir)
    right_angle = _direction_angle(bridge, right_dir)
    if left_angle < 30.0 and right_angle < 30.0:
        return True
    return bool(
        (left_angle > 60.0 and right_angle < 30.0)
        or (right_angle > 60.0 and left_angle < 30.0)
    )


def _close_gaps(
    walls: list[ReadoutWall],
    tolerance: float,
    max_gap: float,
) -> list[ReadoutWall]:
    """Bridge small endpoint gaps without splitting the retained walls."""

    if not walls:
        return []
    snapped = _cluster_points(walls, tolerance)
    bridges: list[ReadoutWall] = []
    endpoints = [
        (point, index)
        for index, wall in enumerate(snapped)
        for point in (wall.start, wall.end)
    ]
    for left_index in range(len(endpoints)):
        left_point, left_wall = endpoints[left_index]
        for right_index in range(left_index + 1, len(endpoints)):
            right_point, right_wall = endpoints[right_index]
            if left_wall == right_wall:
                continue
            gap = math.dist(left_point, right_point)
            if gap <= tolerance * 0.25 or gap > max_gap:
                continue
            if not _plausible_bridge(
                _unit_direction(left_point, right_point),
                snapped[left_wall],
                snapped[right_wall],
            ):
                continue
            bridges.append(
                ReadoutWall(
                    start=left_point,
                    end=right_point,
                    thickness=min(
                        snapped[left_wall].thickness,
                        snapped[right_wall].thickness,
                    ),
                    wall_type="interior",
                    continuity=0.5,
                )
            )
    return _deduplicate_walls([*snapped, *bridges], tolerance)


def _open_endpoint_indices(
    walls: list[ReadoutWall],
    tolerance: float,
) -> set[int]:
    endpoints = [
        point
        for wall in walls
        for point in (wall.start, wall.end)
    ]
    open_indices = set()
    for index, point in enumerate(endpoints):
        if not any(
            other_index != index and math.dist(point, other) <= tolerance
            for other_index, other in enumerate(endpoints)
        ):
            open_indices.add(index)
    return open_indices


def _repair_open_endpoints(
    walls: list[ReadoutWall],
    frame: float,
    tolerance: float,
    spur_length_fraction: float,
) -> list[ReadoutWall]:
    """Prune short open spurs and close open endpoints on the frame boundary."""

    if not walls:
        return []
    open_indices = _open_endpoint_indices(walls, tolerance)
    spur_limit = max(tolerance, float(spur_length_fraction) * frame)
    repaired = []
    for index, wall in enumerate(walls):
        start_open = 2 * index in open_indices
        end_open = 2 * index + 1 in open_indices
        length = math.dist(wall.start, wall.end)
        if length <= spur_limit and start_open != end_open:
            continue
        repaired.append(wall)
    open_indices = _open_endpoint_indices(repaired, tolerance)
    sides: dict[str, list[tuple[float, float]]] = {
        "left": [],
        "right": [],
        "top": [],
        "bottom": [],
    }
    for index in sorted(open_indices):
        wall = repaired[index // 2]
        point = wall.start if index % 2 == 0 else wall.end
        if abs(point[0]) <= tolerance:
            sides["left"].append(point)
        elif abs(point[0] - frame) <= tolerance:
            sides["right"].append(point)
        if abs(point[1]) <= tolerance:
            sides["top"].append(point)
        elif abs(point[1] - frame) <= tolerance:
            sides["bottom"].append(point)
    connectors = []
    for side, points in sides.items():
        if len(points) < 2:
            continue
        axis = 1 if side in {"left", "right"} else 0
        ordered = sorted(points, key=lambda point: point[axis])
        for left, right in pairwise(ordered):
            if math.dist(left, right) <= tolerance:
                continue
            connectors.append(
                ReadoutWall(
                    start=left,
                    end=right,
                    thickness=3.0,
                    wall_type="exterior",
                    continuity=0.5,
                )
            )
    return _deduplicate_walls([*repaired, *connectors], tolerance)


def _project_openings(
    openings: list[ReadoutOpening],
    walls: list[ReadoutWall],
) -> list[ReadoutOpening]:
    projected = []
    for opening in openings:
        if not walls:
            continue
        best_index = min(
            range(len(walls)),
            key=lambda index: LineString(
                [walls[index].start, walls[index].end]
            ).distance(Point(opening.center)),
        )
        wall = walls[best_index]
        line = LineString([wall.start, wall.end])
        point = line.interpolate(line.project(Point(opening.center)))
        projected.append(
            ReadoutOpening(
                center=(float(point.x), float(point.y)),
                width=min(float(opening.width), float(line.length)),
                kind=opening.kind,
                wall_index=best_index,
            )
        )
    return projected


def reconcile_plan(
    plan: RasterWallPlan,
    room_labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    tolerance_fraction: float = 0.012,
    gap_tolerance_fraction: float = 0.02,
    use_ip: bool = False,
    repair_open_endpoints: bool = False,
    open_spur_length_fraction: float = 0.02,
) -> RasterWallPlan:
    """Snap, node and re-face a raster plan.

    ``use_ip`` delegates to the R2V-style integer-programming reconciliation
    in :mod:`floorflow.geometry.ip_reconcile`, which additionally consolidates
    collinear wall fragments across shared junctions.
    """

    if use_ip:
        from floorflow.geometry.ip_reconcile import ip_reconcile_plan

        return ip_reconcile_plan(
            plan,
            room_labels,
            taxonomy,
            tolerance_fraction=tolerance_fraction,
        )
    tolerance = max(1.0, float(tolerance_fraction) * plan.frame)
    gap_tolerance = max(tolerance, float(gap_tolerance_fraction) * plan.frame)
    walls = _close_gaps(plan.walls, tolerance, gap_tolerance)
    if repair_open_endpoints:
        walls = _repair_open_endpoints(
            walls,
            plan.frame,
            tolerance,
            open_spur_length_fraction,
        )
    rooms = _wall_faces(walls, room_labels, taxonomy)
    if not rooms:
        rooms = plan.rooms
    openings = _project_openings(plan.openings, walls)
    return RasterWallPlan(
        frame=plan.frame,
        walls=walls,
        rooms=rooms,
        openings=openings,
        room_source="reconciled_faces" if rooms is not plan.rooms else plan.room_source,
    )


def _ink_support(
    wall: ReadoutWall,
    support_mask: np.ndarray | None,
    threshold: float,
) -> bool:
    if support_mask is None:
        return True
    height, width = support_mask.shape
    hits = 0
    samples = 9
    for step in range(samples):
        fraction = step / (samples - 1)
        x = wall.start[0] + fraction * (wall.end[0] - wall.start[0])
        y = wall.start[1] + fraction * (wall.end[1] - wall.start[1])
        radius = 2
        left = max(0, round(x) - radius)
        right = min(width, round(x) + radius + 1)
        top = max(0, round(y) - radius)
        bottom = min(height, round(y) + radius + 1)
        hits += int(support_mask[top:bottom, left:right].any())
    return hits / samples >= float(threshold)


def fuse_wall_plans(
    primary: RasterWallPlan,
    donor: RasterWallPlan,
    *,
    support_mask: np.ndarray | None = None,
    support_threshold: float = 0.5,
    tolerance_fraction: float = 0.012,
) -> RasterWallPlan:
    """Keep primary rooms/openings and add only supported donor walls."""

    tolerance = max(1.0, float(tolerance_fraction) * primary.frame)
    primary_walls = _deduplicate_walls(primary.walls, tolerance)
    fused = list(primary_walls)
    for wall in donor.walls:
        if not _ink_support(wall, support_mask, support_threshold):
            continue
        if any(_duplicate(wall, kept, tolerance) for kept in fused):
            continue
        fused.append(wall)
    return RasterWallPlan(
        frame=primary.frame,
        walls=fused,
        rooms=primary.rooms,
        openings=_project_openings(primary.openings, fused),
        room_source=primary.room_source,
    )
