"""CAD-native wall candidates derived from room partition boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import cv2
import numpy as np

from floorflow.geometry.wall_graph_readout import ReadoutWall


def _round_point(point: np.ndarray) -> tuple[float, float]:
    return (round(float(point[0]), 3), round(float(point[1]), 3))


def _segment_key(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (start, end) if start <= end else (end, start)


def _split_segment(
    start: np.ndarray,
    end: np.ndarray,
    vertices: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    delta = end - start
    length_sq = float(delta @ delta)
    if length_sq <= 1.0e-9:
        return []
    relative = vertices - start
    fractions = relative @ delta / length_sq
    projected = start + fractions[:, None] * delta
    distance = np.linalg.norm(vertices - projected, axis=1)
    selected = fractions[
        (distance <= 1.0e-5)
        & (fractions > 1.0e-6)
        & (fractions < 1.0 - 1.0e-6)
    ]
    fractions = np.unique(
        np.concatenate([[0.0], selected, [1.0]])
    )
    points = start + fractions[:, None] * delta
    return list(pairwise(points))


def room_boundary_candidates(
    rooms: list[dict],
    *,
    minimum_length: float = 2.5,
    simplify_epsilon: float = 1.0,
    split_points: Sequence[Sequence[float]] | None = None,
) -> list[ReadoutWall]:
    """Return unique junction-split room boundary segments."""

    rings: list[np.ndarray] = []
    for room in rooms:
        polygon = np.asarray(room.get("polygon") or [], dtype=np.float64)
        if len(polygon) >= 3:
            simplified = cv2.approxPolyDP(
                polygon.astype(np.float32).reshape(-1, 1, 2),
                float(simplify_epsilon),
                True,
            ).reshape(-1, 2)
            rings.append(polygon if len(simplified) < 3 else simplified.astype(np.float64))
        for hole in room.get("holes") or []:
            values = np.asarray(hole, dtype=np.float64)
            if len(values) >= 3:
                simplified = cv2.approxPolyDP(
                    values.astype(np.float32).reshape(-1, 1, 2),
                    float(simplify_epsilon),
                    True,
                ).reshape(-1, 2)
                rings.append(values if len(simplified) < 3 else simplified.astype(np.float64))
    if not rings:
        return []
    vertices = np.concatenate(rings, axis=0)
    unique_points = {}
    for point in vertices:
        unique_points[_round_point(point)] = point
    if split_points is not None:
        for point in np.asarray(split_points, dtype=np.float64).reshape(-1, 2):
            unique_points[_round_point(point)] = point
    vertex_array = np.asarray(list(unique_points.values()), dtype=np.float64)
    segments = {}
    for ring in rings:
        for start, end in pairwise(np.concatenate([ring, ring[:1]], axis=0)):
            for part_start, part_end in _split_segment(
                start,
                end,
                vertex_array,
            ):
                key = _segment_key(
                    _round_point(part_start),
                    _round_point(part_end),
                )
                segments[key] = (part_start, part_end)
    candidates = []
    for start, end in segments.values():
        length = float(np.linalg.norm(end - start))
        if length < float(minimum_length):
            continue
        candidates.append(
            ReadoutWall(
                start=(float(start[0]), float(start[1])),
                end=(float(end[0]), float(end[1])),
                thickness=3.0,
            )
        )
    return sorted(
        candidates,
        key=lambda wall: (
            -np.linalg.norm(np.asarray(wall.end) - np.asarray(wall.start)),
            wall.start,
            wall.end,
        ),
    )


def footprint_boundary_candidates(
    footprint: np.ndarray,
    *,
    epsilon: float = 1.5,
    minimum_length: float = 2.5,
) -> list[ReadoutWall]:
    """Approximate the exterior envelope as wall candidates."""

    contours, _hierarchy = cv2.findContours(
        (np.asarray(footprint) > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    walls = []
    for contour in contours:
        if cv2.contourArea(contour) < 16.0:
            continue
        polygon = cv2.approxPolyDP(
            contour,
            float(epsilon),
            True,
        ).reshape(-1, 2)
        for start, end in pairwise(np.concatenate([polygon, polygon[:1]], axis=0)):
            if np.linalg.norm(end - start) < float(minimum_length):
                continue
            walls.append(
                ReadoutWall(
                    start=(float(start[0]), float(start[1])),
                    end=(float(end[0]), float(end[1])),
                    thickness=3.0,
                    wall_type="exterior",
                )
            )
    return walls


def wall_candidate_evidence(
    walls: list[ReadoutWall],
    wall_probability: np.ndarray,
    *,
    samples: int = 9,
    radius: int = 1,
) -> np.ndarray:
    """Average wall-layer probability along each candidate centerline."""

    probability = np.asarray(wall_probability, dtype=np.float32)
    height, width = probability.shape
    evidence = np.zeros(len(walls), dtype=np.float32)
    for index, wall in enumerate(walls):
        values = []
        for fraction in np.linspace(0.0, 1.0, samples):
            x = wall.start[0] + fraction * (wall.end[0] - wall.start[0])
            y = wall.start[1] + fraction * (wall.end[1] - wall.start[1])
            left = max(0, round(x) - radius)
            right = min(width, round(x) + radius + 1)
            top = max(0, round(y) - radius)
            bottom = min(height, round(y) + radius + 1)
            values.append(
                float(
                    probability[top:bottom, left:right].max()
                    if right > left and bottom > top
                    else 0.0
                )
            )
        evidence[index] = float(np.mean(values))
    return evidence


__all__ = [
    "footprint_boundary_candidates",
    "room_boundary_candidates",
    "wall_candidate_evidence",
]
