"""Compose learned vector primitives with a frozen room layer."""

from __future__ import annotations

import math

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.partition_polygons import partition_polygon_rows
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutOpening,
    ReadoutWall,
)


def _nearest_wall(
    center: tuple[float, float],
    walls: list[ReadoutWall],
) -> int:
    best = 0
    distance = float("inf")
    point = np.asarray(center, dtype=np.float64)
    for index, wall in enumerate(walls):
        start = np.asarray(wall.start, dtype=np.float64)
        end = np.asarray(wall.end, dtype=np.float64)
        delta = end - start
        scale = float(delta @ delta)
        fraction = (
            0.0
            if scale <= 1.0e-9
            else float(np.clip((point - start) @ delta / scale, 0.0, 1.0))
        )
        distance_to_wall = float(
            np.linalg.norm(point - (start + fraction * delta))
        )
        if distance_to_wall < distance:
            best = index
            distance = distance_to_wall
    return best


def vector_scaffold_plan(
    room_logits: torch.Tensor,
    footprint: torch.Tensor,
    wall_endpoints: torch.Tensor,
    wall_exists: torch.Tensor,
    wall_thickness: torch.Tensor,
    opening_centers: torch.Tensor,
    opening_exists: torch.Tensor,
    opening_widths: torch.Tensor,
    opening_classes: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float = 0.5,
    opening_host_logits: torch.Tensor | None = None,
    host_threshold: float = 0.5,
) -> RasterWallPlan:
    """Build one vector plan from scaffold outputs and a room layer."""

    if room_logits.ndim != 3:
        raise ValueError("room_logits must be one [C,H,W] map")
    if footprint.ndim != 2:
        raise ValueError("footprint must be one [H,W] map")
    room_ids = torch.tensor(taxonomy.room_ids, device=room_logits.device)
    room_labels = room_ids[room_logits.argmax(dim=0)]
    room_labels = room_labels.cpu().numpy() * footprint.cpu().numpy()
    rooms = partition_polygon_rows(
        room_labels,
        footprint.cpu().numpy(),
        taxonomy,
        minimum_area=16,
    )
    scale = float(room_logits.shape[-1])
    walls = []
    for index in range(wall_endpoints.shape[0]):
        if float(wall_exists[index].sigmoid().cpu()) < float(threshold):
            continue
        points = wall_endpoints[index].detach().float().cpu().numpy().copy()
        points = np.clip(points, 0.0, 1.0) * scale
        if math.dist(points[0], points[1]) < 2.0:
            continue
        thickness = max(
            1.0,
            float(wall_thickness[index].sigmoid().detach().cpu()) * scale,
        )
        walls.append(
            ReadoutWall(
                start=(float(points[0, 0]), float(points[0, 1])),
                end=(float(points[1, 0]), float(points[1, 1])),
                thickness=thickness,
            )
        )
    openings = []
    if walls:
        for index in range(opening_centers.shape[0]):
            if float(opening_exists[index].sigmoid().cpu()) < float(threshold):
                continue
            center = (
                opening_centers[index].detach().float().cpu().numpy()
                * scale
            )
            class_index = int(opening_classes[index].argmax().cpu())
            class_id = int(taxonomy.opening_ids[class_index])
            host = _nearest_wall((float(center[0]), float(center[1])), walls)
            if (
                opening_host_logits is not None
                and index < opening_host_logits.shape[0]
            ):
                probability = opening_host_logits[index].softmax(dim=-1)
                predicted_host = int(probability.argmax().cpu())
                if (
                    predicted_host < len(walls)
                    and float(probability[predicted_host].cpu())
                    >= float(host_threshold)
                ):
                    host = predicted_host
            width = max(
                1.0,
                float(opening_widths[index].sigmoid().cpu()) * scale,
            )
            host_wall = walls[host]
            host_length = math.dist(host_wall.start, host_wall.end)
            openings.append(
                ReadoutOpening(
                    center=(float(center[0]), float(center[1])),
                    width=min(width, host_length),
                    kind=(
                        "window"
                        if taxonomy.class_names[class_id].startswith("window:")
                        else "door"
                    ),
                    wall_index=host,
                )
            )
    return RasterWallPlan(
        frame=scale,
        walls=walls,
        rooms=rooms,
        openings=openings,
        room_source="vector_scaffold_rooms",
    )


__all__ = ["vector_scaffold_plan"]
