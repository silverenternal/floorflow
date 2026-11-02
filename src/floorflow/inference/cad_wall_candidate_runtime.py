"""Batchnative room-boundary candidates for wall scoring and composition."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label as component_label

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.cad_wall_candidates import (
    footprint_boundary_candidates,
    room_boundary_candidates,
    wall_candidate_evidence,
)
from floorflow.geometry.partition_polygons import partition_polygon_rows
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutOpening,
    ReadoutWall,
)
from floorflow.heads.cad_topology_repair import (
    apply_bounded_topology_repair,
)
from floorflow.inference.cad_layer_vectorizer import project_detections_to_walls
from floorflow.inference.opening_heatmap_decoder import (
    learned_opening_detections,
)
from floorflow.inference.typed_junction_stage import typed_junction_rows


def canonical_room_rows(
    room_logits: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    minimum_area: int = 64,
    largest_component_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Convert one room-logit map into a legal explicit partition."""

    valid = footprint.detach().cpu().numpy().astype(bool)
    room_ids = torch.tensor(taxonomy.room_ids, device=room_logits.device)
    labels = room_ids[room_logits.argmax(dim=0)].detach().cpu().numpy()
    labels = labels * valid
    room_footprint = valid
    if largest_component_only and valid.any():
        components, count = component_label(valid)
        if count > 1:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            room_footprint = components == int(sizes.argmax())
            labels = labels * room_footprint
    rooms = partition_polygon_rows(
        labels,
        room_footprint,
        taxonomy,
        minimum_area=int(minimum_area),
    )
    return labels, room_footprint, rooms


def candidate_walls(
    rooms: list[dict],
    footprint: np.ndarray,
    wall_probability: np.ndarray,
    extra_walls=(),
) -> tuple[list[ReadoutWall], np.ndarray]:
    """Generate and evidence-score unique room/footprint wall candidates."""

    unique = {}
    for wall in [
        *extra_walls,
        *room_boundary_candidates(rooms),
        *footprint_boundary_candidates(footprint),
    ]:
        start = tuple(round(value, 2) for value in wall.start)
        end = tuple(round(value, 2) for value in wall.end)
        key = (start, end) if start <= end else (end, start)
        unique[key] = wall
    walls = list(unique.values())
    evidence = wall_candidate_evidence(
        walls,
        wall_probability,
        radius=2,
    )
    order = np.argsort(-evidence, kind="stable")
    return [walls[index] for index in order], evidence[order]


def candidate_walls_with_sources(
    rooms: list[dict],
    footprint: np.ndarray,
    wall_probability: np.ndarray,
    *,
    extra_walls_by_source: dict[int, list[ReadoutWall]] | None = None,
) -> tuple[list[ReadoutWall], np.ndarray, np.ndarray]:
    """Generate candidates while preserving their proposal-source identity."""

    proposed: dict[tuple, tuple[ReadoutWall, int]] = {}

    def add(walls: list[ReadoutWall], source: int) -> None:
        for wall in walls:
            start = tuple(round(value, 2) for value in wall.start)
            end = tuple(round(value, 2) for value in wall.end)
            key = (start, end) if start <= end else (end, start)
            previous = proposed.get(key)
            if previous is None or int(source) < previous[1]:
                proposed[key] = (wall, int(source))

    add(room_boundary_candidates(rooms), 0)
    add(footprint_boundary_candidates(footprint), 1)
    for source, walls in (extra_walls_by_source or {}).items():
        add(walls, int(source))
    walls = [item[0] for item in proposed.values()]
    sources = np.asarray([item[1] for item in proposed.values()], dtype=np.int64)
    evidence = wall_candidate_evidence(walls, wall_probability, radius=2)
    order = np.argsort(-evidence, kind="stable")
    return (
        [walls[index] for index in order],
        evidence[order],
        sources[order],
    )


def pad_candidate_tensors(
    walls: list[ReadoutWall],
    evidence: np.ndarray,
    *,
    frame: float,
    max_walls: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad one candidate set into scorer tensors."""

    count = min(len(walls), int(max_walls))
    endpoints = np.zeros((int(max_walls), 2, 2), dtype=np.float32)
    valid = np.zeros(int(max_walls), dtype=bool)
    scores = np.zeros(int(max_walls), dtype=np.float32)
    for index, wall in enumerate(walls[:count]):
        endpoints[index] = np.asarray([wall.start, wall.end]) / float(frame)
        valid[index] = True
        scores[index] = float(evidence[index])
    return (
        torch.from_numpy(endpoints).to(device),
        torch.from_numpy(valid).to(device),
        torch.from_numpy(scores).to(device),
    )


def pad_candidate_sources(
    sources: np.ndarray,
    *,
    max_walls: int,
    device: torch.device,
) -> torch.Tensor:
    output = np.zeros(int(max_walls), dtype=np.int64)
    count = min(len(sources), int(max_walls))
    output[:count] = sources[:count]
    return torch.from_numpy(output).to(device)


def compose_scored_candidate_plan(
    endpoints: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    rooms: list[dict],
    frame: float,
    opening_detection_walls: list[ReadoutWall],
    opening_probability: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    score_threshold: float = 0.5,
    wall_budget: int | None = None,
    exterior: torch.Tensor | None = None,
    exterior_budget: int | None = None,
    interior_budget: int | None = None,
    wall_type_logits: torch.Tensor | None = None,
    routed_openings: list[ReadoutOpening] | None = None,
    opening_threshold: float = 0.3,
    repair_keep_logits: torch.Tensor | None = None,
    repair_endpoint_delta: torch.Tensor | None = None,
    repair_max_edits: int = 8,
    repair_keep_threshold: float = 0.5,
) -> RasterWallPlan:
    """Compose selected candidate walls with the fixed room/opening layers."""

    if (repair_keep_logits is None) != (repair_endpoint_delta is None):
        raise ValueError(
            "repair keep logits and endpoint deltas must be provided together"
        )
    if repair_keep_logits is not None:
        repaired = apply_bounded_topology_repair(
            endpoints[None],
            valid[None],
            repair_keep_logits[None],
            repair_endpoint_delta[None],
            keep_threshold=float(repair_keep_threshold),
            max_edits=int(repair_max_edits),
        )
        endpoints = repaired["endpoints"][0]
        valid = repaired["valid"][0]
    walls = []
    active = valid & (scores >= float(score_threshold))
    if (
        exterior is not None
        and exterior_budget is not None
        and interior_budget is not None
    ):
        active = torch.zeros_like(valid)
        for phase, budget in (
            (exterior.bool(), exterior_budget),
            (~exterior.bool(), interior_budget),
        ):
            candidates = valid & phase
            count = min(int(budget), int(candidates.sum()))
            if count <= 0:
                continue
            indices = scores.masked_fill(~candidates, float("-inf")).topk(
                count
            ).indices
            active[indices] = True
    elif wall_budget is not None:
        active = torch.zeros_like(valid)
        count = min(int(wall_budget), int(valid.sum()))
        if count > 0:
            indices = scores.masked_fill(~valid, float("-inf")).topk(count).indices
            active[indices] = True
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    for index in active_indices.detach().cpu().tolist():
        points = endpoints[index].detach().cpu().numpy() * float(frame)
        walls.append(
            ReadoutWall(
                start=(float(points[0, 0]), float(points[0, 1])),
                end=(float(points[1, 0]), float(points[1, 1])),
                thickness=3.0,
                wall_type=(
                    "exterior"
                    if (
                        wall_type_logits is not None
                        and bool(wall_type_logits[index].sigmoid().cpu() >= 0.5)
                    )
                    or (
                        wall_type_logits is None
                        and exterior is not None
                        and bool(exterior[index].cpu())
                    )
                    else "interior"
                ),
            )
        )
    if routed_openings is None:
        host_plan = RasterWallPlan(
            frame=float(frame),
            walls=opening_detection_walls,
            rooms=rooms,
            openings=[],
        )
        detections = learned_opening_detections(
            host_plan,
            opening_probability,
            taxonomy,
            threshold=float(opening_threshold),
        )
        openings = project_detections_to_walls(
            detections,
            walls,
            float(frame),
        )
    else:
        openings = routed_openings
    junction_rows = typed_junction_rows(
        endpoints[active_indices],
        torch.ones(len(active_indices), dtype=torch.bool),
        frame=float(frame),
        wall_indices=torch.arange(len(active_indices)),
    )
    return RasterWallPlan(
        frame=float(frame),
        walls=walls,
        rooms=rooms,
        openings=openings,
        room_source="candidate_score_partition",
        junctions=junction_rows,
    )


def opening_plan_signature(plan: RasterWallPlan) -> tuple:
    """Return the target-free opening readout used by the safe wall gate."""

    return tuple(
        sorted(
            (
                opening.kind,
                round(float(opening.center[0]), 4),
                round(float(opening.center[1]), 4),
                round(float(opening.width), 4),
            )
            for opening in plan.openings
        )
    )


def select_opening_safe_plan(
    control: RasterWallPlan,
    candidate: RasterWallPlan,
) -> RasterWallPlan:
    """Accept candidate walls only when their opening readout is unchanged."""

    if opening_plan_signature(control) == opening_plan_signature(candidate):
        return candidate
    return control


__all__ = [
    "candidate_walls",
    "candidate_walls_with_sources",
    "canonical_room_rows",
    "compose_scored_candidate_plan",
    "opening_plan_signature",
    "pad_candidate_sources",
    "pad_candidate_tensors",
    "select_opening_safe_plan",
]
