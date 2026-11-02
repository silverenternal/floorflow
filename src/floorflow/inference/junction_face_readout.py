"""Graph-first room-face candidate from token junction and wall incidence."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

from floorflow.data.rplan_boundary_adapter import resample_ring
from floorflow.inference.token_graph_projection import (
    project_wall_endpoints_to_token_junctions,
)


@torch.no_grad()
def junction_face_candidate(
    output: dict[str, torch.Tensor],
    *,
    minimum_area: float = 1.0e-4,
) -> dict[str, torch.Tensor] | None:
    if output["wall_endpoints"].shape[0] != 1:
        return None
    projected = project_wall_endpoints_to_token_junctions(output)
    endpoints = projected["wall_endpoints"][0].cpu().numpy()
    active = projected["wall_exists"][0].sigmoid().cpu().numpy() >= 0.5
    lines = [
        LineString([endpoints[index, 0], endpoints[index, 1]])
        for index, valid in enumerate(active)
        if valid
    ]
    if len(lines) < 3:
        return None
    faces = [
        polygon
        for polygon in polygonize(unary_union(lines))
        if polygon.area >= minimum_area
    ]
    if not faces:
        return None
    faces.sort(key=lambda polygon: polygon.area, reverse=True)
    reference_polygons = projected["room_polygons"][0].cpu().numpy()
    room_count = min(len(faces), reference_polygons.shape[0])
    face_centers = np.asarray(
        [polygon.centroid.coords[0] for polygon in faces[:room_count]]
    )
    room_centers = reference_polygons[:room_count].mean(axis=1)
    rows, columns = linear_sum_assignment(
        np.linalg.norm(room_centers[:, None] - face_centers[None], axis=-1)
    )
    polygons = projected["room_polygons"].clone()
    point_count = polygons.shape[2]
    for room, face in zip(rows, columns, strict=True):
        ring = np.asarray(faces[face].exterior.coords[:-1], dtype=np.float32)
        polygons[0, room] = torch.from_numpy(
            resample_ring(ring, point_count)
        ).to(polygons.device)
    candidate = dict(projected)
    candidate["room_polygons"] = polygons
    candidate["junction_face_count"] = polygons.new_tensor(float(len(faces)))
    return candidate


__all__ = ["junction_face_candidate"]
