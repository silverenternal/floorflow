"""Inference adapter for edit-local vector residual refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from shapely.geometry import Polygon

from floorflow.evaluation.vector_edit_protocol import (
    VectorOpening,
    VectorPlan,
    VectorRoom,
    VectorWall,
)
from floorflow.heads.vector_proposal_refiner import VectorProposalRefiner


@dataclass(frozen=True)
class VectorRefinementResult:
    plan: VectorPlan
    affected_rooms: int
    affected_walls: int
    affected_openings: int
    room_mask: tuple[bool, ...]
    wall_mask: tuple[bool, ...]
    opening_mask: tuple[bool, ...]


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    closed = np.concatenate([values, values[:1]], axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    result = np.empty((count, 2), dtype=np.float32)
    result[:, 0] = np.interp(samples, cumulative, closed[:, 0])
    result[:, 1] = np.interp(samples, cumulative, closed[:, 1])
    return result


def _point_touches(center: tuple[float, float], radius: float, region: np.ndarray) -> bool:
    height, width = region.shape
    x, y = center
    left = max(0, int(np.floor(x - radius)))
    right = min(width - 1, int(np.ceil(x + radius)))
    top = max(0, int(np.floor(y - radius)))
    bottom = min(height - 1, int(np.ceil(y + radius)))
    yy, xx = np.mgrid[top : bottom + 1, left : right + 1]
    circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    return bool((region[top : bottom + 1, left : right + 1] & circle).any())


def _wall_touches(wall: VectorWall, region: np.ndarray) -> bool:
    for fraction in np.linspace(0.0, 1.0, max(2, int(wall.length) + 1)):
        point = wall.line.interpolate(float(fraction), normalized=True)
        if _point_touches((point.x, point.y), 0.5, region):
            return True
    return False


@torch.no_grad()
def refine_vector_plan(
    plan: VectorPlan,
    edit_region: np.ndarray,
    features: torch.Tensor,
    refiner: VectorProposalRefiner,
    *,
    polygon_points: int = 16,
    max_rooms: int = 32,
    max_walls: int = 64,
    max_openings: int = 48,
    graph_token: torch.Tensor | None = None,
    refine_rooms: bool = True,
) -> VectorRefinementResult:
    """Refine only primitives intersecting the declared edit region."""

    region = np.asarray(edit_region, dtype=bool)
    scale = float(region.shape[-1])
    room_mask = np.asarray(
        [bool((room.mask & region).any()) for room in plan.rooms[:max_rooms]]
    )
    if not refine_rooms:
        room_mask[:] = False
    wall_mask = np.asarray(
        [_wall_touches(wall, region) for wall in plan.walls[:max_walls]]
    )
    opening_mask = np.asarray(
        [
            _point_touches(
                opening.center,
                max(1.0, opening.width / 2.0),
                region,
            )
            for opening in plan.openings[:max_openings]
        ]
    )
    room_polygons = torch.zeros(
        1,
        max_rooms,
        polygon_points,
        2,
        device=features.device,
    )
    for index, room in enumerate(plan.rooms[:max_rooms]):
        room_polygons[0, index] = torch.from_numpy(
            _resample(
                np.asarray(room.polygon.exterior.coords)[:-1] / scale,
                polygon_points,
            )
        )
    walls = torch.zeros(1, max_walls, 2, 2, device=features.device)
    for index, wall in enumerate(plan.walls[:max_walls]):
        walls[0, index] = torch.tensor(
            [
                [wall.start[0] / scale, wall.start[1] / scale],
                [wall.end[0] / scale, wall.end[1] / scale],
            ],
            device=features.device,
        )
    openings = torch.zeros(1, max_openings, 2, device=features.device)
    for index, opening in enumerate(plan.openings[:max_openings]):
        openings[0, index] = torch.tensor(
            [opening.center[0] / scale, opening.center[1] / scale],
            device=features.device,
        )
    room_valid = torch.zeros(1, max_rooms, dtype=torch.bool, device=features.device)
    room_valid[0, : room_mask.size] = torch.from_numpy(room_mask)
    wall_valid = torch.zeros(1, max_walls, dtype=torch.bool, device=features.device)
    wall_valid[0, : wall_mask.size] = torch.from_numpy(wall_mask)
    opening_valid = torch.zeros(
        1,
        max_openings,
        dtype=torch.bool,
        device=features.device,
    )
    opening_valid[0, : opening_mask.size] = torch.from_numpy(opening_mask)
    output = refiner(
        features,
        room_polygons,
        room_valid,
        walls,
        wall_valid,
        openings,
        opening_valid,
        graph_token,
    )
    rooms = list(plan.rooms)
    for index in np.flatnonzero(room_mask):
        polygon = Polygon(
            output["room_polygons"][0, index].cpu().numpy() * scale
        ).buffer(0)
        if (
            isinstance(polygon, Polygon)
            and polygon.is_valid
            and polygon.area > 0.0
        ):
            rooms[index] = VectorRoom(
                label=rooms[index].label,
                polygon=polygon,
                mask=rooms[index].mask,
            )
    wall_rows = list(plan.walls)
    for index in np.flatnonzero(wall_mask):
        coordinates = output["wall_endpoints"][0, index].cpu().numpy() * scale
        wall_rows[index] = VectorWall(
            start=tuple(coordinates[0]),
            end=tuple(coordinates[1]),
            thickness=wall_rows[index].thickness,
            wall_type=wall_rows[index].wall_type,
        )
    opening_rows = list(plan.openings)
    for index in np.flatnonzero(opening_mask):
        center = output["opening_centers"][0, index].cpu().numpy() * scale
        opening_rows[index] = VectorOpening(
            center=tuple(center),
            width=opening_rows[index].width,
            kind=opening_rows[index].kind,
        )
    refined = VectorPlan(
        rooms=tuple(rooms),
        walls=tuple(wall_rows),
        openings=tuple(opening_rows),
        wall_mask=plan.wall_mask,
        opening_mask=plan.opening_mask,
        source=plan.source,
    )
    return VectorRefinementResult(
        plan=refined,
        affected_rooms=int(room_mask.sum()),
        affected_walls=int(wall_mask.sum()),
        affected_openings=int(opening_mask.sum()),
        room_mask=tuple(bool(value) for value in room_mask),
        wall_mask=tuple(bool(value) for value in wall_mask),
        opening_mask=tuple(bool(value) for value in opening_mask),
    )


__all__ = ["VectorRefinementResult", "refine_vector_plan"]
