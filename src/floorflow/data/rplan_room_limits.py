"""Deterministic room subset used by every RPLAN target branch."""

from __future__ import annotations

from dataclasses import replace

from shapely.geometry import Polygon

from floorflow.evaluation.canonical_floorplan import CanonicalFloorplan


def limit_canonical_plan_rooms(
    plan: CanonicalFloorplan,
    max_rooms: int,
) -> CanonicalFloorplan:
    """Keep one deterministic largest-room subset across all target branches."""

    limit = max(0, int(max_rooms))
    retained = tuple(
        sorted(
            plan.rooms,
            key=lambda room: (-Polygon(room.polygon).area, room.id),
        )[:limit]
    )
    retained_ids = {room.id for room in retained}
    adjacency = tuple(
        edge
        for edge in plan.adjacency
        if edge[0] in retained_ids and edge[1] in retained_ids
    )
    return replace(plan, rooms=retained, adjacency=adjacency)


__all__ = ["limit_canonical_plan_rooms"]
