"""Bounded endpoint fusion between owner-network and learned walls."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from shapely.geometry import Point, Polygon

from floorflow.evaluation.canonical_floorplan import CanonicalWall


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1.0e-9 else np.zeros_like(vector)


def fuse_learned_wall_endpoints(
    walls: Sequence[CanonicalWall],
    learned_endpoints: np.ndarray,
    learned_scores: np.ndarray,
    footprint: Polygon,
    *,
    weight: float = 1.0,
    score_threshold: float = 0.35,
    cluster_tolerance: float = 2.0,
    match_tolerance: float = 3.0,
    maximum_residual: float = 1.0,
    minimum_wall_length: float = 2.01,
) -> tuple[tuple[CanonicalWall, ...], int]:
    """Move shared wall junctions toward matched learned endpoints."""

    endpoints = np.asarray(learned_endpoints, dtype=np.float64)
    scores = np.asarray(learned_scores, dtype=np.float64)
    if endpoints.ndim != 3 or endpoints.shape[1:] != (2, 2):
        raise ValueError("learned endpoints must be [N,2,2]")
    if scores.ndim != 1 or scores.shape[0] != endpoints.shape[0]:
        raise ValueError("learned scores must match endpoint count")
    if not walls or float(weight) <= 0.0:
        return tuple(walls), 0
    learned = []
    for segment, score in zip(endpoints, scores, strict=True):
        if float(score) < float(score_threshold):
            continue
        direction = _unit(segment[1] - segment[0])
        if np.linalg.norm(segment[1] - segment[0]) < 2.01:
            continue
        learned.append(
            (
                segment[0],
                segment[1],
                direction,
                float(score),
            )
        )
    if not learned:
        return tuple(walls), 0

    endpoint_rows = [
        (wall.id, endpoint_index, wall, index)
        for index, wall in enumerate(walls)
        for endpoint_index in (0, 1)
    ]
    clusters: list[list[tuple[str, int, CanonicalWall, int]]] = []
    for row in endpoint_rows:
        point = row[2].start if row[1] == 0 else row[2].end
        for cluster in clusters:
            existing = cluster[0][2]
            existing_point = (
                existing.start if cluster[0][1] == 0 else existing.end
            )
            if math.dist(point, existing_point) <= float(cluster_tolerance):
                cluster.append(row)
                break
        else:
            clusters.append([row])

    endpoint_updates = {}
    edited_clusters = 0
    for cluster in clusters:
        displacements = []
        for _wall_id, endpoint_index, wall, _wall_index in cluster:
            point = (
                np.asarray(wall.start, dtype=np.float64)
                if endpoint_index == 0
                else np.asarray(wall.end, dtype=np.float64)
            )
            other = (
                np.asarray(wall.end, dtype=np.float64)
                if endpoint_index == 0
                else np.asarray(wall.start, dtype=np.float64)
            )
            owner_direction = _unit(other - point)
            candidates = []
            for start, end, direction, score in learned:
                for learned_point in (start, end):
                    distance = float(np.linalg.norm(learned_point - point))
                    if distance > float(match_tolerance):
                        continue
                    alignment = abs(float(owner_direction @ direction))
                    cost = distance + float(match_tolerance) * (
                        1.0 - alignment
                    )
                    candidates.append((cost, learned_point - point, score))
            if candidates:
                _cost, displacement, _score = min(
                    candidates,
                    key=lambda candidate: candidate[0],
                )
                displacements.append(displacement)
        if not displacements:
            continue
        residual = float(weight) * np.mean(
            np.stack(displacements),
            axis=0,
        )
        residual_length = float(np.linalg.norm(residual))
        if residual_length > float(maximum_residual):
            residual *= float(maximum_residual) / residual_length
        proposed = []
        valid = True
        for _wall_id, endpoint_index, wall, wall_index in cluster:
            point = (
                np.asarray(wall.start, dtype=np.float64)
                if endpoint_index == 0
                else np.asarray(wall.end, dtype=np.float64)
            )
            moved = point + residual
            if footprint.distance(Point(moved)) > 1.0e-6:
                valid = False
                break
            proposed.append((wall_index, endpoint_index, moved))
        if not valid:
            continue
        proposed_by_key = {
            (wall_index, endpoint_index): moved
            for wall_index, endpoint_index, moved in proposed
        }
        for wall_index, endpoint_index, moved in proposed:
            wall = walls[wall_index]
            other = (
                np.asarray(wall.start, dtype=np.float64)
                if endpoint_index == 1
                else np.asarray(wall.end, dtype=np.float64)
            )
            other = proposed_by_key.get(
                (wall_index, 1 - endpoint_index),
                other,
            )
            if float(np.linalg.norm(moved - other)) < float(
                minimum_wall_length
            ):
                valid = False
                break
        if not valid:
            continue
        edited_clusters += 1
        for wall_index, endpoint_index, moved in proposed:
            endpoint_updates[(wall_index, endpoint_index)] = (
                float(moved[0]),
                float(moved[1]),
            )

    fused = []
    for wall_index, wall in enumerate(walls):
        start = endpoint_updates.get((wall_index, 0), wall.start)
        end = endpoint_updates.get((wall_index, 1), wall.end)
        fused.append(
            CanonicalWall(
                id=wall.id,
                start=start,
                end=end,
                thickness=wall.thickness,
                wall_type=wall.wall_type,
            )
        )
    if any(
        math.dist(wall.start, wall.end) < float(minimum_wall_length)
        for wall in fused
    ):
        return tuple(walls), 0
    return tuple(fused), edited_clusters


__all__ = ["fuse_learned_wall_endpoints"]
