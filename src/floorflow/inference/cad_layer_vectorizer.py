"""Vectorize independent CAD layers into a plan-first representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as component_label
from shapely.geometry import LineString, Point

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.cad_wall_candidates import (
    footprint_boundary_candidates,
    room_boundary_candidates,
    wall_candidate_evidence,
)
from floorflow.geometry.ip_reconcile import ip_reconcile_plan
from floorflow.geometry.opening_readout import (
    read_wall_interval_detections,
)
from floorflow.geometry.partition_polygons import partition_polygon_rows
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutFurniture,
    ReadoutOpening,
    ReadoutWall,
    extract_wall_segments,
)
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.inference.furniture_vector_readout import (
    clip_furniture_to_rooms,
    read_furniture_queries,
    read_furniture_raster,
)
from floorflow.inference.opening_heatmap_decoder import (
    learned_opening_detections,
)
from floorflow.model.cad_layers import CadLayeredOutput


@dataclass(frozen=True)
class CadVectorLayerOutput:
    """Vector-first wall/opening/room output from a layer prediction."""

    frame: float
    walls: tuple[ReadoutWall, ...]
    openings: tuple[ReadoutOpening, ...]
    furniture: tuple[ReadoutFurniture, ...]
    rooms: tuple[dict, ...]
    labels: np.ndarray

    def to_plan(self) -> RasterWallPlan:
        return RasterWallPlan(
            frame=self.frame,
            walls=list(self.walls),
            rooms=list(self.rooms),
            openings=list(self.openings),
            room_source="layer_partition",
            furniture=list(self.furniture),
        )


def vectorize_layers(
    layers: CadLayeredOutput,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    wall_threshold: float = 0.5,
    opening_threshold: float = 0.5,
    furniture_threshold: float = 0.5,
    minimum_room_area: int = 16,
    use_room_boundary_scaffold: bool = True,
    use_room_boundary_candidates: bool = False,
    candidate_evidence_threshold: float = 0.2,
    reconcile_wall_candidates: bool = False,
    opening_family_probability: np.ndarray | None = None,
    wall_probability: np.ndarray | None = None,
    opening_wall_probability: np.ndarray | None = None,
    largest_room_component_only: bool = False,
    furniture_queries: dict[str, torch.Tensor] | None = None,
) -> CadVectorLayerOutput:
    """Convert each semantic layer into its native CAD representation."""

    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    if footprint.ndim != 3 or footprint.shape[0] != 1:
        raise ValueError("vectorize_layers expects one sample")
    valid = footprint[0].detach().cpu().numpy().astype(bool)
    room_ids = torch.tensor(
        taxonomy.room_ids,
        device=layers.room_logits.device,
    )
    room_labels = room_ids[layers.room_logits.argmax(dim=1)]
    room_labels = room_labels[0].detach().cpu().numpy() * valid
    room_footprint = valid
    if largest_room_component_only and valid.any():
        components, count = component_label(valid)
        if count > 1:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            room_footprint = components == int(sizes.argmax())
            room_labels = room_labels * room_footprint
    rooms = partition_polygon_rows(
        room_labels,
        room_footprint,
        taxonomy,
        minimum_area=int(minimum_room_area),
    )
    if wall_probability is None:
        wall_probability = (
            layers.wall_logits[0, 0].sigmoid().detach().cpu().numpy()
        )
    wall_probability = np.asarray(wall_probability, dtype=np.float32)
    if wall_probability.shape != valid.shape:
        raise ValueError("wall probability and footprint must share one shape")
    wall_mask = (wall_probability >= float(wall_threshold)).astype(np.uint8)
    if use_room_boundary_candidates:
        room_tensor = torch.from_numpy(room_labels)[None]
        boundary = partition_wall_mask(room_tensor, radius=1)[0].numpy()
        detection_probability = (
            wall_probability
            if opening_wall_probability is None
            else np.asarray(opening_wall_probability, dtype=np.float32)
        )
        opening_detection_walls = extract_wall_segments(
            np.maximum(
                (detection_probability >= float(wall_threshold)).astype(np.uint8),
                boundary.astype(np.uint8),
            ),
            taxonomy,
        )
        walls_by_key = {}
        for wall in [
            *room_boundary_candidates(rooms),
            *footprint_boundary_candidates(room_footprint),
        ]:
            start = tuple(round(value, 2) for value in wall.start)
            end = tuple(round(value, 2) for value in wall.end)
            key = (start, end) if start <= end else (end, start)
            walls_by_key[key] = wall
        walls = list(walls_by_key.values())
        evidence = wall_candidate_evidence(
            walls,
            wall_probability,
            radius=2,
        )
        walls = [
            wall
            for wall, score in zip(walls, evidence, strict=True)
            if float(score) >= float(candidate_evidence_threshold)
        ]
        if reconcile_wall_candidates:
            reconciled = ip_reconcile_plan(
                RasterWallPlan(
                    frame=float(room_labels.shape[0]),
                    walls=walls,
                    rooms=rooms,
                    openings=[],
                ),
                room_labels,
                taxonomy,
            )
            walls = reconciled.walls
    else:
        if use_room_boundary_scaffold:
            room_tensor = torch.from_numpy(room_labels)[None]
            boundary = partition_wall_mask(room_tensor, radius=1)[0].numpy()
            wall_mask = np.maximum(wall_mask, boundary.astype(np.uint8))
        wall_mask *= valid
        walls = extract_wall_segments(wall_mask, taxonomy)
        opening_detection_walls = walls
    plan = RasterWallPlan(
        frame=float(room_labels.shape[0]),
        walls=walls,
        rooms=rooms,
        openings=[],
        room_source="layer_vector",
    )
    opening_probability = layers.opening_logits[0].softmax(dim=0)
    opening_probability = opening_probability.detach().cpu().numpy()
    foreground = opening_probability.max(axis=0)
    logits = np.log(np.clip(opening_probability, 1.0e-6, 1.0))
    opening_detection_plan = RasterWallPlan(
        frame=plan.frame,
        walls=opening_detection_walls,
        rooms=rooms,
        openings=[],
    )
    detections = (
        learned_opening_detections(
            opening_detection_plan,
            opening_family_probability,
            taxonomy,
            threshold=float(opening_threshold),
            min_area=7,
        )
        if opening_family_probability is not None
        else read_wall_interval_detections(
            foreground,
            logits,
            taxonomy.opening_ids,
            taxonomy.class_names,
            opening_detection_plan,
            threshold=float(opening_threshold),
            radius=1,
            min_width=7.0,
            max_per_wall=8,
            score_smoothing=3,
            max_gap_samples=2,
        )
    )
    openings = project_detections_to_walls(detections, plan.walls, plan.frame)
    plan.openings = openings
    if furniture_queries is None:
        furniture = read_furniture_raster(
            layers.furniture_logits[0],
            room_labels,
            taxonomy,
            threshold=float(furniture_threshold),
        )
    else:
        class_logits = furniture_queries["class_logits"]
        mask_logits = furniture_queries["mask_logits"]
        if class_logits.ndim == 3:
            class_logits = class_logits[0]
        if mask_logits.ndim == 4:
            mask_logits = mask_logits[0]
        if mask_logits.shape[-2:] != room_labels.shape:
            mask_logits = F.interpolate(
                mask_logits[None].float(),
                size=room_labels.shape,
                mode="bilinear",
                align_corners=False,
            )[0]
        query_budget = furniture_queries.get("query_budget")
        if query_budget is not None and query_budget.numel() > 0:
            query_budget = query_budget.reshape(-1)[0]
        active_queries = furniture_queries.get("active_queries")
        if active_queries is not None:
            active_queries = active_queries.reshape(-1)
        room_assignment_logits = furniture_queries.get(
            "room_assignment_logits"
        )
        if (
            room_assignment_logits is not None
            and room_assignment_logits.ndim == 3
        ):
            room_assignment_logits = room_assignment_logits[0]
        furniture = read_furniture_queries(
            class_logits,
            mask_logits,
            room_labels,
            taxonomy,
            query_budget=query_budget,
            active_queries=active_queries,
            room_assignment_logits=room_assignment_logits,
            presence_threshold=float(furniture_threshold),
            mask_threshold=float(furniture_threshold),
        )
    furniture = clip_furniture_to_rooms(
        furniture,
        rooms,
        taxonomy=taxonomy,
    )
    plan.furniture = list(furniture)
    return CadVectorLayerOutput(
        frame=plan.frame,
        walls=tuple(walls),
        openings=tuple(openings),
        furniture=tuple(furniture),
        rooms=tuple(rooms),
        labels=room_labels,
    )


def project_detections_to_walls(
    detections,
    walls,
    frame: float,
) -> list[ReadoutOpening]:
    openings = []
    if not walls:
        return openings
    lines = [
        LineString([wall.start, wall.end])
        for wall in walls
    ]
    for detection in detections:
        point = Point(detection.center)
        host = min(
            range(len(lines)),
            key=lambda index: lines[index].distance(point),
        )
        line = lines[host]
        projected = line.interpolate(line.project(point))
        center = detection.center
        if line.distance(point) > 0.012 * float(frame):
            center = (float(projected.x), float(projected.y))
        openings.append(
            ReadoutOpening(
                center=center,
                width=min(
                    max(1.0, float(detection.width)),
                    max(float(line.length), 1.0e-6),
                ),
                kind=detection.family,
                wall_index=host,
            )
        )
    return openings


__all__ = [
    "CadVectorLayerOutput",
    "project_detections_to_walls",
    "vectorize_layers",
]
