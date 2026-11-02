"""Integer-programming wall reconciliation adapted from Raster-to-Vector.

The upstream R2V solver (`third_party/FloorplanTransformation/pytorch/IP.py`,
MIT) selects a consistent subset of candidate wall primitives by maximizing
detection coverage subject to junction consistency.  We reuse that
formulation for the subset of the problem the project actually has: snap
clustered endpoints onto shared junctions, consolidate collinear fragments,
and let a small MILP choose between a merged wall and its fragments.

The MILP is solved with SciPy's HiGHS backend, so no extra dependency is
introduced.  Oversized candidate sets or solver failures fall back to the
existing shapely reconciliation in :mod:`floorflow.geometry.reconciliation`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutOpening,
    ReadoutWall,
    _wall_faces,
)

_ANGLE_TOLERANCE = math.radians(5.0)


@dataclass(frozen=True)
class _Candidate:
    junctions: tuple[int, int]
    length: float
    thickness: float
    support: int
    constituents: tuple[int, ...] = ()


def _cluster_endpoints(
    walls: list[ReadoutWall],
    tolerance: float,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """Union-find clustering of wall endpoints; returns junctions and ids."""

    points = [point for wall in walls for point in (wall.start, wall.end)]
    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            if math.dist(points[left], points[right]) <= tolerance:
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    parent[root_right] = root_left

    root_to_id: dict[int, int] = {}
    members: list[list[tuple[float, float]]] = []
    point_ids: list[int] = []
    for index, point in enumerate(points):
        root = find(index)
        if root not in root_to_id:
            root_to_id[root] = len(members)
            members.append([])
        junction_id = root_to_id[root]
        members[junction_id].append(point)
        point_ids.append(junction_id)
    junctions = [
        (
            float(sum(item[0] for item in group) / len(group)),
            float(sum(item[1] for item in group) / len(group)),
        )
        for group in members
    ]
    wall_junctions = [
        (point_ids[2 * index], point_ids[2 * index + 1])
        for index in range(len(walls))
    ]
    return junctions, wall_junctions


def _base_candidates(
    walls: list[ReadoutWall],
    junctions: list[tuple[float, float]],
    wall_junctions: list[tuple[int, int]],
    tolerance: float,
) -> list[_Candidate]:
    deduped: dict[tuple[int, int], _Candidate] = {}
    thickness: dict[tuple[int, int], list[float]] = {}
    for wall, (left, right) in zip(walls, wall_junctions):
        if left == right:
            continue
        length = math.dist(junctions[left], junctions[right])
        if length <= tolerance * 0.25:
            continue
        key = (min(left, right), max(left, right))
        thickness.setdefault(key, []).append(float(wall.thickness))
        previous = deduped.get(key)
        support = 1 if previous is None else previous.support + 1
        deduped[key] = _Candidate(
            junctions=key,
            length=length,
            thickness=float(np.median(thickness[key])),
            support=support,
        )
    return [deduped[key] for key in sorted(deduped)]


def _angle(left: tuple[float, float], right: tuple[float, float]) -> float:
    norm = math.hypot(*left) * math.hypot(*right)
    if norm <= 1.0e-9:
        return math.pi
    cosine = (left[0] * right[0] + left[1] * right[1]) / norm
    return math.acos(max(-1.0, min(1.0, cosine)))


def _direction(candidate: _Candidate, junctions):
    start = junctions[candidate.junctions[0]]
    end = junctions[candidate.junctions[1]]
    return (end[0] - start[0], end[1] - start[1])


def _collinear_gap(
    left: _Candidate,
    right: _Candidate,
    junctions,
    tolerance: float,
) -> float | None:
    left_dir = _direction(left, junctions)
    right_dir = _direction(right, junctions)
    angle = _angle(left_dir, right_dir)
    if min(angle, math.pi - angle) > _ANGLE_TOLERANCE:
        return None
    norm = math.hypot(*left_dir)
    axis = (left_dir[0] / norm, left_dir[1] / norm)
    projections = []
    for junction_id in right.junctions:
        point = junctions[junction_id]
        projections.append(point[0] * axis[0] + point[1] * axis[1])
    origin = junctions[left.junctions[0]]
    span_start = origin[0] * axis[0] + origin[1] * axis[1]
    span_end = span_start + left.length
    low, high = sorted((span_start, span_end))
    gap = max(0.0, min(abs(min(projections) - high), abs(max(projections) - low)))
    if max(low - max(projections), min(projections) - high, 0.0) > tolerance:
        return None
    return gap


def _merge_groups(
    candidates: list[_Candidate],
    junctions,
    tolerance: float,
    merge_gap: float,
) -> list[list[int]]:
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            shared = set(candidates[left].junctions) & set(candidates[right].junctions)
            gap = _collinear_gap(candidates[left], candidates[right], junctions, tolerance)
            if gap is None:
                continue
            if not shared and gap > merge_gap:
                continue
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left
    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(find(index), []).append(index)
    return [sorted(group) for group in groups.values() if len(group) > 1]


def _merged_candidate(
    group: list[int],
    candidates: list[_Candidate],
    junctions,
) -> _Candidate | None:
    direction = _direction(candidates[group[0]], junctions)
    norm = math.hypot(*direction)
    if norm <= 1.0e-9:
        return None
    axis = (direction[0] / norm, direction[1] / norm)
    ranked = []
    for index in group:
        for junction_id in candidates[index].junctions:
            point = junctions[junction_id]
            ranked.append((point[0] * axis[0] + point[1] * axis[1], junction_id))
    low = min(ranked)
    high = max(ranked)
    if low[1] == high[1]:
        return None
    length = float(high[0] - low[0])
    if length <= norm * 0.5:
        return None
    thickness = float(np.median([candidates[index].thickness for index in group]))
    support = sum(candidates[index].support for index in group)
    return _Candidate(
        junctions=(low[1], high[1]),
        length=length,
        thickness=thickness,
        support=support,
        constituents=tuple(group),
    )


def _select(
    candidates: list[_Candidate],
    junction_count: int,
    max_variables: int,
) -> list[int] | None:
    count = len(candidates)
    variables = count + junction_count
    if variables > max_variables:
        return None
    weights = np.asarray(
        [
            candidate.length * (1.0 + 0.1 * (candidate.support - 1))
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    cost = np.concatenate(
        [
            -weights,
            np.full(junction_count, -1.0e-3, dtype=np.float64),
        ]
    )
    rows = []
    lowers = []
    uppers = []

    def add(row: np.ndarray, lower: float, upper: float) -> None:
        rows.append(row)
        lowers.append(lower)
        uppers.append(upper)

    for index, candidate in enumerate(candidates):
        for junction_id in candidate.junctions:
            row = np.zeros(variables)
            row[index] = 1.0
            row[count + junction_id] = -1.0
            add(row, -np.inf, 0.0)

    incident = [0] * junction_count
    for candidate in candidates:
        for junction_id in candidate.junctions:
            incident[junction_id] += 1
    for junction_id in range(junction_count):
        if incident[junction_id] == 0:
            continue
        row = np.zeros(variables)
        row[count + junction_id] = 1.0
        for index, candidate in enumerate(candidates):
            if junction_id in candidate.junctions:
                row[index] = -1.0
        add(row, -np.inf, 0.0)

    for merged_index, candidate in enumerate(candidates):
        for constituent in candidate.constituents:
            row = np.zeros(variables)
            row[merged_index] = 1.0
            row[constituent] = 1.0
            add(row, -np.inf, 1.0)

    constraints = LinearConstraint(
        np.vstack(rows),
        np.asarray(lowers),
        np.asarray(uppers),
    )
    result = milp(
        c=cost,
        constraints=[constraints],
        integrality=np.ones(variables),
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
    )
    if result.status != 0 or result.x is None:
        return None
    return [index for index in range(count) if result.x[index] > 0.5]


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
            key=lambda index: _point_segment_distance(
                opening.center,
                walls[index].start,
                walls[index].end,
            ),
        )
        wall = walls[best_index]
        start = np.asarray(wall.start, dtype=np.float64)
        end = np.asarray(wall.end, dtype=np.float64)
        delta = end - start
        length_sq = float(delta @ delta)
        if length_sq <= 1.0e-9:
            continue
        fraction = float(
            np.clip(
                (np.asarray(opening.center) - start) @ delta / length_sq,
                0.0,
                1.0,
            )
        )
        center = start + fraction * delta
        projected.append(
            ReadoutOpening(
                center=(float(center[0]), float(center[1])),
                width=min(float(opening.width), math.sqrt(length_sq)),
                kind=opening.kind,
                wall_index=best_index,
            )
        )
    return projected


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    start_array = np.asarray(start, dtype=np.float64)
    delta = np.asarray(end, dtype=np.float64) - start_array
    length_sq = float(delta @ delta)
    if length_sq <= 1.0e-9:
        return float(np.linalg.norm(np.asarray(point) - start_array))
    fraction = float(
        np.clip((np.asarray(point) - start_array) @ delta / length_sq, 0.0, 1.0)
    )
    return float(np.linalg.norm(np.asarray(point) - (start_array + fraction * delta)))


def _fallback(
    plan: RasterWallPlan,
    room_labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    tolerance_fraction: float,
) -> RasterWallPlan:
    from floorflow.geometry.reconciliation import reconcile_plan

    return reconcile_plan(
        plan,
        room_labels,
        taxonomy,
        tolerance_fraction=tolerance_fraction,
    )


def ip_reconcile_plan(
    plan: RasterWallPlan,
    room_labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    tolerance_fraction: float = 0.006,
    merge_gap_fraction: float = 0.02,
    max_variables: int = 800,
    use_milp: bool = True,
) -> RasterWallPlan:
    """Reconcile walls through junction clustering and a small MILP selection."""

    if not plan.walls:
        return plan
    tolerance = max(1.0, float(tolerance_fraction) * plan.frame)
    merge_gap = max(tolerance, float(merge_gap_fraction) * plan.frame)
    junctions, wall_junctions = _cluster_endpoints(plan.walls, tolerance)
    candidates = _base_candidates(plan.walls, junctions, wall_junctions, tolerance)
    if not candidates:
        return plan
    if use_milp:
        for group in _merge_groups(candidates, junctions, tolerance, merge_gap):
            merged = _merged_candidate(group, candidates, junctions)
            if merged is not None:
                candidates.append(merged)
        selected = _select(candidates, len(junctions), max_variables)
        if selected is None:
            return _fallback(plan, room_labels, taxonomy, tolerance_fraction)
    else:
        selected = list(range(len(candidates)))
    walls = [
        ReadoutWall(
            start=junctions[candidates[index].junctions[0]],
            end=junctions[candidates[index].junctions[1]],
            thickness=float(candidates[index].thickness),
        )
        for index in selected
    ]
    if not walls:
        return plan
    rooms = _wall_faces(walls, room_labels, taxonomy)
    if not rooms:
        rooms = plan.rooms
    openings = _project_openings(plan.openings, walls)
    return RasterWallPlan(
        frame=plan.frame,
        walls=walls,
        rooms=rooms,
        openings=openings,
        room_source="ip_reconciled_faces" if rooms is not plan.rooms else plan.room_source,
    )


__all__ = ["ip_reconcile_plan"]
