"""Owner-network wall candidates for learned keep/residual actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from shapely.geometry import Point, Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.owner_cell_complex import OwnerCellComplex
from floorflow.geometry.wall_graph_readout import ReadoutWall
from floorflow.inference.mask_partition_readout import decode_mask_partition
from floorflow.inference.rplan_query_quality_arbitration import (
    select_quality_queries,
)

_LEARNED_WALL_EVIDENCE_TOLERANCE = 5.0


@dataclass(frozen=True)
class OwnerWallActionBatch:
    endpoints: torch.Tensor
    valid: torch.Tensor
    evidence: torch.Tensor
    exterior: torch.Tensor
    cell_complexes: tuple[OwnerCellComplex | None, ...]
    walls: tuple[tuple[ReadoutWall, ...], ...]


def _selected_queries(
    predictions: dict[str, torch.Tensor],
    index: int,
) -> list[int]:
    exists = predictions["room_exists"][index].detach().sigmoid()
    count = int(exists.shape[0])
    if "room_count_logits" in predictions:
        count = int(
            predictions["room_count_logits"][index].argmax().item()
        )
    count = max(1, min(count, int(exists.shape[0])))
    if "query_quality_logits" in predictions:
        selected = select_quality_queries(
            exists[None],
            predictions["query_quality_logits"][index : index + 1].detach(),
            torch.tensor([count], device=exists.device),
        )[0]
        return torch.nonzero(selected, as_tuple=False).flatten().tolist()
    return torch.topk(exists, k=count).indices.tolist()


def _learned_wall_evidence(
    wall: ReadoutWall,
    learned_endpoints: np.ndarray,
    learned_scores: np.ndarray,
    *,
    scale: float,
    tolerance: float = _LEARNED_WALL_EVIDENCE_TOLERANCE,
) -> float:
    if learned_endpoints.size == 0:
        return 0.0
    owner_start = np.asarray(wall.start, dtype=np.float64)
    owner_end = np.asarray(wall.end, dtype=np.float64)
    endpoints = learned_endpoints * scale
    direct = 0.5 * (
        np.linalg.norm(endpoints[:, 0] - owner_start, axis=-1)
        + np.linalg.norm(endpoints[:, 1] - owner_end, axis=-1)
    )
    reverse = 0.5 * (
        np.linalg.norm(endpoints[:, 1] - owner_start, axis=-1)
        + np.linalg.norm(endpoints[:, 0] - owner_end, axis=-1)
    )
    cost = np.minimum(direct, reverse)
    index = int(np.argmin(cost))
    if float(cost[index]) > float(tolerance):
        return 0.0
    confidence = float(learned_scores[index])
    proximity = 1.0 - float(cost[index]) / float(tolerance)
    return confidence * max(0.0, proximity)


def select_connected_wall_actions(
    endpoints: np.ndarray,
    scores: np.ndarray,
    valid: np.ndarray,
    exterior: np.ndarray,
    *,
    threshold: float,
    coordinate_scale: float,
    cluster_tolerance: float = 2.0,
) -> np.ndarray:
    """Select a score-preferred forest before adding high-confidence edges."""

    endpoints = np.asarray(endpoints, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    exterior = np.asarray(exterior, dtype=bool)
    selected = valid & (exterior | (scores >= float(threshold)))
    count = min(len(scores), len(valid))
    if count == 0:
        return selected
    pixels = endpoints[:count] * float(coordinate_scale)
    parent = list(range(count * 2))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> bool:
        left_root = root(left)
        right_root = root(right)
        if left_root == right_root:
            return False
        parent[right_root] = left_root
        return True

    endpoint_rows = [
        (index * 2 + endpoint_index, point)
        for index in range(count)
        if valid[index]
        for endpoint_index, point in enumerate(pixels[index])
    ]
    for left in range(len(endpoint_rows)):
        for right in range(left + 1, len(endpoint_rows)):
            if np.linalg.norm(
                endpoint_rows[left][1] - endpoint_rows[right][1]
            ) <= float(cluster_tolerance):
                union(endpoint_rows[left][0], endpoint_rows[right][0])

    order = sorted(
        (
            index
            for index in range(count)
            if valid[index]
        ),
        key=lambda index: (
            -float(scores[index]),
            -float(exterior[index]),
        ),
    )
    for index in order:
        left_root = root(index * 2)
        right_root = root(index * 2 + 1)
        if left_root != right_root:
            union(left_root, right_root)
            selected[index] = True
    return selected


def build_owner_wall_action_batch(
    predictions: dict[str, torch.Tensor],
    boundary: torch.Tensor | np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    canvas_size: int,
    max_walls: int,
    confidence_threshold: float = 0.5,
    minimum_area: float = 16.0,
    gap_policy: str = "adaptive_fill",
) -> OwnerWallActionBatch:
    """Pool merged owner walls into fixed-size scorer tensors."""

    points = (
        boundary.detach().cpu().numpy()
        if torch.is_tensor(boundary)
        else np.asarray(boundary, dtype=np.float32)
    )
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("boundary must be [B,P,2]")
    batch = int(points.shape[0])
    device = predictions["room_exists"].device
    dtype = predictions["room_exists"].dtype
    endpoints = torch.zeros(
        (batch, int(max_walls), 2, 2),
        dtype=dtype,
        device=device,
    )
    valid = torch.zeros(
        (batch, int(max_walls)),
        dtype=torch.bool,
        device=device,
    )
    evidence = torch.zeros(
        (batch, int(max_walls)),
        dtype=dtype,
        device=device,
    )
    exterior = torch.zeros_like(valid)
    complexes = []
    wall_rows = []
    scale = float(max(1, canvas_size - 1))
    for index in range(batch):
        boundary_points = points[index] * scale
        selected = _selected_queries(predictions, index)
        classes = predictions["room_class"][index].argmax(dim=-1).cpu().numpy()
        sample = {
            key: value[index : index + 1]
            for key, value in predictions.items()
            if torch.is_tensor(value)
        }
        readout = decode_mask_partition(
            sample["room_instance_mask_logits"],
            boundary_points,
            selected,
            classes,
            sample["room_exists"][0].sigmoid(),
            taxonomy,
            canvas_size=canvas_size,
            confidence_threshold=confidence_threshold,
            minimum_area=minimum_area,
            gap_policy=gap_policy,
            mask_probabilities=sample.get(
                "room_instance_partition_probabilities"
            ),
            partition_support="independent_confidence",
        )
        complex_ = readout.cell_complex
        complexes.append(complex_)
        if complex_ is None:
            wall_rows.append(())
            continue
        learned_endpoints = (
            sample["wall_endpoints"][0].detach().cpu().numpy()
            if "wall_endpoints" in sample
            else np.zeros((0, 2, 2), dtype=np.float32)
        )
        learned_scores = (
            sample["wall_exists"][0].detach().cpu().sigmoid().numpy()
            if "wall_exists" in sample
            else np.zeros(0, dtype=np.float32)
        )
        walls = tuple(
            sorted(
                complex_.merged_wall_candidates(),
                key=lambda wall: -float(
                    np.linalg.norm(
                        np.asarray(wall.end) - np.asarray(wall.start)
                    )
                ),
            )[: int(max_walls)]
        )
        footprint = Polygon(boundary_points)
        for wall_index, wall in enumerate(walls):
            endpoints[index, wall_index, 0] = torch.tensor(
                wall.start,
                dtype=dtype,
                device=device,
            ) / scale
            endpoints[index, wall_index, 1] = torch.tensor(
                wall.end,
                dtype=dtype,
                device=device,
            ) / scale
            valid[index, wall_index] = True
            evidence[index, wall_index] = (
                0.5
                + 0.5
                * _learned_wall_evidence(
                    wall,
                    learned_endpoints,
                    learned_scores,
                    scale=scale,
                )
            )
            midpoint = Point(
                0.5 * (wall.start[0] + wall.end[0]),
                0.5 * (wall.start[1] + wall.end[1]),
            )
            exterior[index, wall_index] = (
                midpoint.distance(footprint.boundary) <= 1.5
            )
        wall_rows.append(walls)
    return OwnerWallActionBatch(
        endpoints=endpoints,
        valid=valid,
        evidence=evidence,
        exterior=exterior,
        cell_complexes=tuple(complexes),
        walls=tuple(wall_rows),
    )


__all__ = [
    "OwnerWallActionBatch",
    "build_owner_wall_action_batch",
    "select_connected_wall_actions",
]
