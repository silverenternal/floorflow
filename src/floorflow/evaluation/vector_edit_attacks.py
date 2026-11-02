"""Deterministic attacks for FEVP-2 vector evaluator sensitivity."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from shapely.geometry import Polygon

from floorflow.evaluation.vector_edit_protocol import (
    VectorOpening,
    VectorPlan,
    VectorRoom,
)


def _point_in_mask(
    center: tuple[float, float],
    mask: np.ndarray,
) -> bool:
    x = min(mask.shape[1] - 1, max(0, round(center[0])))
    y = min(mask.shape[0] - 1, max(0, round(center[1])))
    return bool(mask[y, x])


def perturb_unaffected_room_vertex(
    plan: VectorPlan,
    edit_region: np.ndarray,
) -> VectorPlan:
    rooms = list(plan.rooms)
    for index, room in enumerate(rooms):
        if bool((room.mask & edit_region).any()):
            continue
        coordinates = list(room.polygon.exterior.coords)
        if len(coordinates) < 4:
            continue
        x, y = coordinates[1]
        coordinates[1] = (x + 1.0, y)
        rooms[index] = VectorRoom(
            label=room.label,
            polygon=Polygon(coordinates),
            mask=room.mask,
        )
        break
    return VectorPlan(
        rooms=tuple(rooms),
        walls=plan.walls,
        openings=plan.openings,
        wall_mask=plan.wall_mask,
        opening_mask=plan.opening_mask,
        source=plan.source,
    )


def perturb_unaffected_wall_mask(
    plan: VectorPlan,
    edit_region: np.ndarray,
) -> VectorPlan:
    scope = binary_dilation(np.asarray(edit_region, dtype=bool), iterations=2)
    wall_mask = plan.wall_mask.copy()
    candidates = np.argwhere(wall_mask & ~scope)
    if candidates.size == 0:
        return plan
    y, x = candidates[len(candidates) // 2]
    wall_mask[y, x] = False
    return VectorPlan(
        rooms=plan.rooms,
        walls=plan.walls,
        openings=plan.openings,
        wall_mask=wall_mask,
        opening_mask=plan.opening_mask,
        source=plan.source,
    )


def perturb_unaffected_opening(
    plan: VectorPlan,
    edit_region: np.ndarray,
) -> VectorPlan:
    openings = list(plan.openings)
    opening_mask = plan.opening_mask.copy()
    for index, opening in enumerate(openings):
        if _point_in_mask(opening.center, edit_region):
            continue
        openings[index] = VectorOpening(
            center=(opening.center[0] + 2.0, opening.center[1]),
            width=opening.width,
            kind=opening.kind,
        )
        candidates = np.argwhere(opening_mask)
        if candidates.size:
            y, x = candidates[len(candidates) // 2]
            opening_mask[y, x] = False
        break
    return VectorPlan(
        rooms=plan.rooms,
        walls=plan.walls,
        openings=tuple(openings),
        wall_mask=plan.wall_mask,
        opening_mask=opening_mask,
        source=plan.source,
    )


def perturb_unaffected_room_adjacency(
    plan: VectorPlan,
    edit_region: np.ndarray,
) -> VectorPlan:
    index = next(
        (
            item
            for item, room in enumerate(plan.rooms)
            if not bool((room.mask & edit_region).any())
        ),
        None,
    )
    if index is None:
        return plan
    rooms = list(plan.rooms)
    room = rooms[index]
    rooms[index] = VectorRoom(
        label=room.label,
        polygon=Polygon(
            [(x + 3.0, y) for x, y in room.polygon.exterior.coords]
        ),
        mask=room.mask,
    )
    return VectorPlan(
        rooms=tuple(rooms),
        walls=plan.walls,
        openings=plan.openings,
        wall_mask=plan.wall_mask,
        opening_mask=plan.opening_mask,
        source=plan.source,
    )


__all__ = [
    "perturb_unaffected_opening",
    "perturb_unaffected_room_adjacency",
    "perturb_unaffected_room_vertex",
    "perturb_unaffected_wall_mask",
]
