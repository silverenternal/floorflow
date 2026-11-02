"""Conservative near-duplicate wall pruning that preserves plan topology."""

from __future__ import annotations

import math

import numpy as np

from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutOpening,
    ReadoutWall,
)


def _direction(wall: ReadoutWall) -> tuple[np.ndarray, float]:
    delta = np.asarray(wall.end) - np.asarray(wall.start)
    length = float(np.linalg.norm(delta))
    return (delta / length if length > 1.0e-9 else np.zeros(2)), length


def _angle(left: np.ndarray, right: np.ndarray) -> float:
    cosine = abs(float(left @ right))
    return math.degrees(math.acos(max(0.0, min(1.0, cosine))))


def _line_distance(point: np.ndarray, wall: ReadoutWall) -> float:
    start = np.asarray(wall.start)
    direction, _length = _direction(wall)
    delta = point - start
    return float(np.linalg.norm(delta - (delta @ direction) * direction))


def _projection_span(
    wall: ReadoutWall,
    axis: np.ndarray,
) -> tuple[float, float]:
    start = float(np.asarray(wall.start) @ axis)
    end = float(np.asarray(wall.end) @ axis)
    return tuple(sorted((start, end)))


def is_near_duplicate_wall(
    left: ReadoutWall,
    right: ReadoutWall,
    *,
    distance_tolerance: float,
    angle_tolerance: float,
    minimum_overlap_fraction: float,
) -> bool:
    left_direction, left_length = _direction(left)
    right_direction, right_length = _direction(right)
    if left_length <= 1.0e-9 or right_length <= 1.0e-9:
        return False
    if _angle(left_direction, right_direction) > angle_tolerance:
        return False
    left_points = (np.asarray(left.start), np.asarray(left.end))
    right_points = (np.asarray(right.start), np.asarray(right.end))
    if max(_line_distance(point, right) for point in left_points) > distance_tolerance:
        return False
    if max(_line_distance(point, left) for point in right_points) > distance_tolerance:
        return False
    axis = left_direction
    left_low, left_high = _projection_span(left, axis)
    right_low, right_high = _projection_span(right, axis)
    overlap = max(0.0, min(left_high, right_high) - max(left_low, right_low))
    return overlap / max(1.0e-9, min(left_length, right_length)) >= (
        minimum_overlap_fraction
    )


def prune_near_duplicate_walls(
    plan: RasterWallPlan,
    *,
    distance_tolerance_fraction: float = 0.006,
    angle_tolerance_degrees: float = 4.0,
    minimum_overlap_fraction: float = 0.65,
) -> RasterWallPlan:
    """Keep the longest wall in each conservative near-duplicate cluster."""

    tolerance = max(1.0, float(distance_tolerance_fraction) * plan.frame)
    ranked = sorted(
        plan.walls,
        key=lambda wall: (
            -math.dist(wall.start, wall.end),
            -float(wall.continuity),
            wall.start,
            wall.end,
        ),
    )
    kept: list[ReadoutWall] = []
    for wall in ranked:
        if any(
            is_near_duplicate_wall(
                wall,
                existing,
                distance_tolerance=tolerance,
                angle_tolerance=angle_tolerance_degrees,
                minimum_overlap_fraction=minimum_overlap_fraction,
            )
            for existing in kept
        ):
            continue
        kept.append(wall)
    projected = []
    for opening in plan.openings:
        if not kept:
            continue
        best_index = min(
            range(len(kept)),
            key=lambda index: _point_distance(opening, kept[index]),
        )
        projected.append(
            ReadoutOpening(
                center=opening.center,
                width=opening.width,
                kind=opening.kind,
                wall_index=best_index,
            )
        )
    return RasterWallPlan(
        frame=plan.frame,
        walls=kept,
        rooms=plan.rooms,
        openings=projected,
        room_source=plan.room_source,
        furniture=plan.furniture,
        junctions=plan.junctions,
    )


def _point_distance(opening: ReadoutOpening, wall: ReadoutWall) -> float:
    point = np.asarray(opening.center)
    start = np.asarray(wall.start)
    end = np.asarray(wall.end)
    delta = end - start
    denominator = float(delta @ delta)
    fraction = (
        0.0
        if denominator <= 1.0e-9
        else float(np.clip((point - start) @ delta / denominator, 0.0, 1.0))
    )
    return float(np.linalg.norm(point - (start + fraction * delta)))


__all__ = [
    "is_near_duplicate_wall",
    "prune_near_duplicate_walls",
]
