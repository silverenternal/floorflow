"""Deterministic generated-prediction to canonical-floorplan readout."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.validation import make_valid

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.cad_stage_audit import (
    audit_cad_stages,
    stage_exit_operations,
)
from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    CanonicalOperation,
    CanonicalWall,
)
from floorflow.inference.mask_partition_readout import (
    decode_mask_partition,
    fixed_point_safe_polygon,
)
from floorflow.inference.owner_wall_action import (
    select_connected_wall_actions,
)
from floorflow.inference.rplan_query_quality_arbitration import (
    select_quality_queries,
)
from floorflow.representation.polygon_targets import (
    collapse_invalid_polygon_points,
)
from floorflow.representation.vector_floorplan_assembly import (
    assemble_vector_floorplan,
)


def _valid_polygon(points: np.ndarray) -> tuple[Polygon | None, list[str]]:
    operations = []
    shape = Polygon(points)
    if not shape.is_valid:
        shape = make_valid(shape)
        operations.append("polygon_make_valid")
    if shape.geom_type == "MultiPolygon":
        shape = max(shape.geoms, key=lambda item: item.area)
        operations.append("polygon_keep_largest")
    if shape.geom_type != "Polygon" or shape.is_empty or shape.area <= 1.0:
        return None, operations
    return shape, operations


def decode_generated_floorplan(
    predictions: dict[str, torch.Tensor],
    boundary: torch.Tensor | np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    case_id: str,
    canvas_size: int = 256,
    confidence_threshold: float = 0.25,
    minimum_area: float = 16.0,
    use_predicted_count: bool = True,
    geometry_source: str = "polygon",
    mask_gap_policy: str = "threshold",
    partition_support: str = "independent_confidence",
    wall_source: str = "room_polygons",
    wall_action_endpoints: torch.Tensor | np.ndarray | None = None,
    wall_action_scores: torch.Tensor | np.ndarray | None = None,
    wall_action_valid: torch.Tensor | np.ndarray | None = None,
    wall_action_exterior: torch.Tensor | np.ndarray | None = None,
    wall_action_threshold: float = 0.4,
    wall_fusion_weight: float = 0.0,
    wall_fusion_threshold: float = 0.35,
    entrance_points: torch.Tensor | np.ndarray | None = None,
) -> CanonicalFloorplan:
    """Deterministically assemble one generated plan with explicit repair logs."""

    values = predictions["room_exists"].detach().cpu()
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError("generated decoder expects one sample")
    exists = values.sigmoid()[0]
    polygons_tensor = predictions["room_polygons"].detach().cpu()
    if "room_point_valid_logits" in predictions:
        polygons_tensor = collapse_invalid_polygon_points(
            polygons_tensor,
            predictions["room_point_valid_logits"].detach().cpu() >= 0.0,
        )
    polygons = polygons_tensor.numpy()[0]
    classes = (
        predictions["room_class"]
        .detach()
        .argmax(dim=-1)
        .cpu()
        .numpy()[0]
    )
    count = int(exists.shape[0])
    if use_predicted_count:
        if "room_count_logits" not in predictions:
            raise ValueError("predicted room count logits are unavailable")
        count = int(predictions["room_count_logits"].argmax(dim=-1).item())
    count = max(1, min(count, exists.shape[0]))
    if "query_quality_logits" in predictions:
        selected_mask = select_quality_queries(
            values,
            predictions["query_quality_logits"].detach().cpu(),
            torch.tensor([count]),
        )[0]
        selected = torch.nonzero(selected_mask, as_tuple=False).flatten().tolist()
    else:
        selected = torch.topk(exists, k=count).indices.tolist()
    boundary_points = (
        boundary[0].detach().cpu().numpy()
        if torch.is_tensor(boundary)
        else np.asarray(boundary, dtype=np.float32)
    )
    if boundary_points.ndim != 2 or boundary_points.shape[1] != 2:
        raise ValueError("boundary must be [P,2]")
    boundary_points = boundary_points * max(1, canvas_size - 1)
    entrance_interval = None
    if entrance_points is not None:
        entrance_interval = (
            entrance_points.detach().cpu().numpy()
            if torch.is_tensor(entrance_points)
            else np.asarray(entrance_points, dtype=np.float32)
        )
        if entrance_interval.ndim == 3:
            entrance_interval = entrance_interval[0]
        if entrance_interval.shape != (2, 2):
            raise ValueError("entrance points must be [2,2]")
        entrance_interval = entrance_interval * max(1, canvas_size - 1)
    footprint = Polygon(boundary_points)
    if not footprint.is_valid:
        raise ValueError("boundary condition is self-intersecting")
    if geometry_source == "mask_partition":
        footprint = fixed_point_safe_polygon(footprint)
        if footprint.is_empty:
            raise ValueError("boundary condition collapses on the canonical grid")
        boundary_points = np.asarray(footprint.exterior.coords[:-1])
    accepted = []
    accepted_union = None
    operations = []
    mask_adjacency = None
    mask_cell_complex = None
    if geometry_source == "mask_partition":
        if "room_instance_mask_logits" not in predictions:
            raise ValueError("mask partition requires room instance mask logits")
        if mask_gap_policy not in {
            "threshold",
            "hole_fill",
            "nearest_fill",
            "partition_fill",
            "crack_fill",
            "bridge_fill",
            "adaptive_fill",
        }:
            raise ValueError(
                "mask_gap_policy must be threshold, hole_fill, nearest_fill, "
                "partition_fill, bridge_fill, or adaptive_fill"
            )
        mask_readout = decode_mask_partition(
            predictions["room_instance_mask_logits"],
            boundary_points,
            selected,
            classes,
            exists,
            taxonomy,
            canvas_size=canvas_size,
            confidence_threshold=confidence_threshold,
            minimum_area=minimum_area,
            gap_policy=mask_gap_policy,
            mask_probabilities=predictions.get(
                "room_instance_partition_probabilities"
            ),
            partition_support=partition_support,
        )
        accepted = mask_readout.rooms
        operations.extend(mask_readout.operations)
        mask_adjacency = mask_readout.adjacency
        mask_cell_complex = mask_readout.cell_complex
        if entrance_interval is not None:
            operations.append(
                CanonicalOperation(
                    "entrance_conditioned_mask_readout",
                    "opening_0001",
                )
            )
    elif geometry_source != "polygon":
        raise ValueError("geometry_source must be polygon or mask_partition")
    for query in selected if geometry_source == "polygon" else ():
        if float(exists[query]) < float(confidence_threshold):
            continue
        points = polygons[query] * max(1, canvas_size - 1)
        shape, local_operations = _valid_polygon(points)
        operations.extend(
            CanonicalOperation(operation, f"query_{query:02d}")
            for operation in local_operations
        )
        if shape is None:
            operations.append(
                CanonicalOperation("room_drop_invalid", f"query_{query:02d}")
            )
            continue
        clipped = shape.intersection(footprint)
        if not shape.equals(clipped):
            operations.append(
                CanonicalOperation("room_clip_footprint", f"query_{query:02d}")
            )
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda item: item.area)
            operations.append(
                CanonicalOperation(
                    "room_keep_largest_clipped",
                    f"query_{query:02d}",
                )
            )
        if clipped.geom_type != "Polygon" or clipped.area < minimum_area:
            operations.append(
                CanonicalOperation("room_drop_small", f"query_{query:02d}")
            )
            continue
        if accepted_union is not None and clipped.intersection(accepted_union).area > 1.0:
            clipped = clipped.difference(accepted_union)
            operations.append(
                CanonicalOperation("room_resolve_overlap", f"query_{query:02d}")
            )
            if clipped.geom_type == "MultiPolygon":
                clipped = max(clipped.geoms, key=lambda item: item.area)
            if clipped.geom_type != "Polygon" or clipped.area < minimum_area:
                operations.append(
                    CanonicalOperation(
                        "room_drop_after_overlap",
                        f"query_{query:02d}",
                    )
                )
                continue
        label = taxonomy.class_names[
            taxonomy.room_ids[int(classes[query])]
        ]
        accepted.append(
            {
                "id": f"room_{len(accepted) + 1:02d}",
                "label": label,
                "polygon": list(clipped.exterior.coords[:-1]),
                "holes": [
                    list(interior.coords[:-1])
                    for interior in clipped.interiors
                ],
                "confidence": float(exists[query]),
            }
        )
        accepted_union = (
            clipped
            if accepted_union is None
            else accepted_union.union(clipped)
        )
    if not accepted:
        raise ValueError("all generated rooms were rejected by the CAD decoder")
    wall_override = None
    if wall_action_endpoints is not None and wall_action_scores is not None:
        action_endpoints = (
            wall_action_endpoints.detach().cpu().numpy()
            if torch.is_tensor(wall_action_endpoints)
            else np.asarray(wall_action_endpoints, dtype=np.float64)
        )
        action_scores = (
            wall_action_scores.detach().cpu().numpy()
            if torch.is_tensor(wall_action_scores)
            else np.asarray(wall_action_scores, dtype=np.float64)
        )
        action_valid = (
            wall_action_valid.detach().cpu().numpy().astype(bool)
            if torch.is_tensor(wall_action_valid)
            else np.asarray(wall_action_valid, dtype=bool)
            if wall_action_valid is not None
            else np.ones(len(action_scores), dtype=bool)
        )
        action_exterior = (
            wall_action_exterior.detach().cpu().numpy()
            if torch.is_tensor(wall_action_exterior)
            else np.asarray(wall_action_exterior, dtype=bool)
            if wall_action_exterior is not None
            else np.zeros(len(action_scores), dtype=bool)
        )
        wall_override = []
        selected_indices = select_connected_wall_actions(
            action_endpoints,
            action_scores,
            action_valid,
            action_exterior,
            threshold=float(wall_action_threshold),
            coordinate_scale=float(max(1, canvas_size - 1)),
        )
        for index in np.flatnonzero(selected_indices):
            start = tuple(action_endpoints[index, 0] * max(1, canvas_size - 1))
            end = tuple(action_endpoints[index, 1] * max(1, canvas_size - 1))
            if float(np.linalg.norm(np.subtract(end, start))) < 2.01:
                continue
            wall_override.append(
                CanonicalWall(
                    id=f"wall_action_{len(wall_override) + 1:04d}",
                    start=start,
                    end=end,
                    thickness=3.0,
                    wall_type=(
                        "exterior"
                        if index < len(action_exterior)
                        and bool(action_exterior[index])
                        else "interior"
                    ),
                )
            )
    learned_wall_endpoints = None
    learned_wall_scores = None
    if (
        float(wall_fusion_weight) > 0.0
        and "wall_exists" in predictions
        and "wall_endpoints" in predictions
    ):
        learned_wall_scores = (
            predictions["wall_exists"]
            .detach()
            .sigmoid()[0]
            .cpu()
            .numpy()
        )
        learned_wall_endpoints = (
            predictions["wall_endpoints"]
            .detach()[0]
            .cpu()
            .numpy()
            * max(1, canvas_size - 1)
        )
    hasher = hashlib.sha256()
    hasher.update(case_id.encode())
    hasher.update(np.asarray(polygons).tobytes())
    plan = assemble_vector_floorplan(
        case_id=case_id,
        track="rplan_generation",
        width=canvas_size,
        height=canvas_size,
        footprint=list(boundary_points),
        rooms=accepted,
        adjacency=mask_adjacency,
        owner_cell_complex=mask_cell_complex,
        wall_source=wall_source,
        wall_candidates_override=wall_override,
        learned_wall_endpoints=learned_wall_endpoints,
        learned_wall_scores=learned_wall_scores,
        learned_wall_fusion_weight=float(wall_fusion_weight),
        source_hash=hasher.hexdigest(),
        taxonomy=taxonomy,
        operations=tuple(operations),
        include_entrance=(
            geometry_source == "polygon" or entrance_interval is not None
        ),
        entrance_interval=entrance_interval,
    )
    stage_gates = {}
    action_gates = predictions.get("action_gates")
    if action_gates:
        stage_gates = {
            str(name): float(value.detach().reshape(-1)[0].cpu())
            for name, value in action_gates.items()
            if torch.is_tensor(value) and value.numel() > 0
        }
    stage_states = audit_cad_stages(plan, gates=stage_gates)
    operations = tuple(
        sorted(
            plan.operations + stage_exit_operations(stage_states),
            key=lambda item: (item.kind, item.target_id, item.detail),
        )
    )
    return replace(
        plan,
        operations=operations,
    )


__all__ = ["decode_generated_floorplan"]
