"""Validation helpers for the typed predicted-control layer model."""

from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.dynamic_object_instance import render_object_queries
from floorflow.inference.canonical_layer_composition import (
    canonical_union_labels,
)
from floorflow.representation.layered_floorplan import LayeredFloorplan

GATE_TARGETS = {
    "room_macro_iou": (0.45, True),
    "instance_f1": (0.50, True),
    "instance_type_accuracy": (0.75, True),
    "wall_f1": (0.58, True),
    "wall_precision": (0.52, True),
    "wall_recall": (0.65, True),
    "opening_f1": (0.35, True),
    "opening_vector_f1_proxy": (0.35, True),
    "furniture_f1": (0.40, True),
    "instance_count_abs_error": (2.80, False),
    "room_adjacency_f1": (0.50, True),
    "node_f1": (0.82, True),
    "edge_overlap": (0.50, True),
    "ged": (6.0, False),
}


def selection_score(
    metrics: dict[str, float],
    *,
    furniture_weight: float = 0.05,
) -> float:
    weighted = 0.0
    total_weight = 0.0
    for name, (target, higher_is_better) in GATE_TARGETS.items():
        if name not in metrics:
            continue
        value = float(metrics[name])
        progress = (
            value / target
            if higher_is_better
            else 1.0 - value / target
        )
        weight = (
            max(0.0, float(furniture_weight))
            if name == "furniture_f1"
            else 1.0
        )
        if weight == 0.0:
            continue
        weighted += weight * min(1.25, max(-1.25, progress))
        total_weight += weight
    if total_weight == 0.0:
        raise ValueError("selection metrics do not match any gate target")
    return weighted / total_weight


def family_metrics(
    prediction: LayeredFloorplan,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    visible_labels: torch.Tensor | None = None,
) -> dict[str, float]:
    visible = (
        prediction.visible_labels()
        if visible_labels is None
        else visible_labels
    )
    groups = {
        "room_f1": (
            torch.isin(
                prediction.room_labels,
                torch.tensor(taxonomy.room_ids, device=target.device),
            ),
            torch.isin(
                target,
                torch.tensor(taxonomy.room_ids, device=target.device),
            ),
        ),
        "wall_f1": (
            visible == taxonomy.class_id("wall"),
            target == taxonomy.class_id("wall"),
        ),
        "opening_f1": (
            torch.isin(
                visible,
                torch.tensor(taxonomy.opening_ids, device=target.device),
            ),
            torch.isin(
                target,
                torch.tensor(taxonomy.opening_ids, device=target.device),
            ),
        ),
        "furniture_f1": (
            torch.isin(
                visible,
                torch.tensor(taxonomy.furniture_ids, device=target.device),
            ),
            torch.isin(
                target,
                torch.tensor(taxonomy.furniture_ids, device=target.device),
            ),
        ),
        "structure_f1": (
            torch.zeros_like(footprint, dtype=torch.bool),
            target == taxonomy.class_id("structure"),
        ),
    }
    metrics = {}
    for key, (predicted, expected) in groups.items():
        family = key.removesuffix("_f1")
        predicted = predicted.bool() & footprint.bool()
        expected = expected.bool() & footprint.bool()
        true_positive = int((predicted & expected).sum())
        precision = true_positive / max(1, int(predicted.sum()))
        recall = true_positive / max(1, int(expected.sum()))
        metrics[f"{family}_precision"] = precision
        metrics[f"{family}_recall"] = recall
        metrics[key] = 2.0 * precision * recall / max(
            1.0e-8,
            precision + recall,
        )
    return metrics


def compose_stage_labels(
    output: dict[str, torch.Tensor],
    target_size: tuple[int, int],
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    wall_threshold: float = 0.5,
    opening_threshold: float = 0.5,
    furniture_threshold: float = 0.5,
    furniture_query_threshold: float | None = None,
    canonical_boundary_width: int = 7,
) -> tuple[LayeredFloorplan, torch.Tensor]:
    """Compose non-overlapping CAD layers from first-class stage outputs."""

    layers = output["cad_layers"].resized(target_size)
    rendered = layers.render(
        footprint,
        taxonomy,
        wall_threshold=wall_threshold,
        opening_threshold=opening_threshold,
        furniture_threshold=furniture_threshold,
    )
    if "furniture_query_class_logits" in output:
        query_threshold = (
            furniture_threshold
            if furniture_query_threshold is None
            else float(furniture_query_threshold)
        )
        rendered.furniture = render_object_queries(
            output["furniture_query_class_logits"],
            output["furniture_query_mask_logits"],
            taxonomy.furniture_ids,
            footprint,
            presence_threshold=query_threshold,
            mask_threshold=query_threshold,
            active_queries=output.get("furniture_query_active"),
        )
    visible = canonical_union_labels(
        rendered,
        taxonomy,
        boundary_width=int(canonical_boundary_width),
    )
    return rendered, visible


def empty_train_component_totals(
    device: torch.device | str | None = None,
) -> dict[str, float | torch.Tensor]:
    """Return the stable component vector reported by training."""

    values = {
        "ce": 0.0,
        "room_iou": 0.0,
        "tversky": 0.0,
        "boundary": 0.0,
        "family_iou": 0.0,
        "stage_family": 0.0,
        "stage_wall": 0.0,
        "stage_wall_precision": 0.0,
        "stage_opening": 0.0,
        "stage_opening_precision": 0.0,
        "stage_opening_host": 0.0,
        "stage_opening_type": 0.0,
        "stage_furniture": 0.0,
        "routing": 0.0,
        "furniture_query": 0.0,
        "furniture_query_class": 0.0,
        "furniture_query_mask": 0.0,
        "furniture_query_unmatched": 0.0,
        "furniture_query_tversky": 0.0,
        "furniture_query_containment": 0.0,
        "furniture_room_assignment": 0.0,
        "furniture_room_assignment_accuracy": 0.0,
        "instance_room_class": 0.0,
        "instance_room_mask": 0.0,
        "instance_room_unmatched": 0.0,
        "instance_room_tversky": 0.0,
        "instance_room_matched_dice": 0.0,
        "instance_room_matched_iou": 0.0,
        "instance_room_predicted_area": 0.0,
        "instance_room_target_area": 0.0,
        "instance_room_unmatched_area": 0.0,
        "instance_room_centroid_error": 0.0,
        "instance_room_reference_anchor": 0.0,
        "instance_room_mask_centroid": 0.0,
        "instance_room_query_separation": 0.0,
        "instance_room_partition": 0.0,
        "room_query_set_total": 0.0,
        "room_query_class_cardinality": 0.0,
        "room_query_mask_only_assignment_disagreement": 0.0,
        "room_query_count_selected_target_recall": 0.0,
        "room_query_count_selected_background_rate": 0.0,
        "room_query_count_selection_available": 0.0,
        "room_query_selection_alignment": 0.0,
        "room_query_presence_rank": 0.0,
        "room_query_presence_count_regression": 0.0,
        "room_shared_match_count": 0.0,
        "room_valid_target_count": 0.0,
        "room_query_gate_output_present": 0.0,
        "room_adjacency": 0.0,
        "room_adjacency_total": 0.0,
        "room_adjacency_consistency_loss": 0.0,
        "room_adjacency_valid_pairs": 0.0,
        "room_adjacency_assigned_queries": 0.0,
        "room_type_adjacency": 0.0,
        "room_type_adjacency_total": 0.0,
        "room_type_adjacency_loss": 0.0,
        "room_type_adjacency_valid_pairs": 0.0,
        "room_type_adjacency_target_mean": 0.0,
        "room_type_node": 0.0,
        "room_type_node_total": 0.0,
        "room_type_node_loss": 0.0,
        "dense_type_adjacency": 0.0,
        "dense_type_node": 0.0,
        "dense_type_total": 0.0,
        "dense_type_edge_target_mean": 0.0,
    }
    if device is None:
        return values
    return {
        key: torch.zeros((), device=device, dtype=torch.float32)
        for key in values
    }


def accumulate_train_component_totals(
    totals: dict[str, float | torch.Tensor],
    components: dict[str, object],
    batch_size: int,
) -> None:
    """Accumulate scalar diagnostics on the accelerator without host sync."""

    for key, value in components.items():
        if not torch.is_tensor(value) or value.numel() != 1:
            continue
        scalar = value.detach().float().reshape(())
        current = totals.get(key)
        if not torch.is_tensor(current):
            current = scalar.new_zeros(())
            totals[key] = current
        current.add_(scalar, alpha=int(batch_size))


def train_component_values(components: dict[str, object]) -> dict[str, object]:
    """Select every independently tracked training scalar."""

    return {
        "ce": components["pixel_distribution_ce"],
        "room_iou": components["room_soft_iou_loss"],
        "tversky": components["room_tversky_loss"],
        "boundary": components["room_boundary_loss"],
        "family_iou": components["family_iou_term"],
        "stage_family": components["stage_family_total"],
        "stage_wall": components["stage_wall_loss"],
        "stage_wall_precision": components["stage_wall_precision_loss"],
        "stage_opening": components["stage_opening_loss"],
        "stage_opening_precision": components[
            "stage_opening_precision_loss"
        ],
        "stage_opening_host": components["stage_opening_host_loss"],
        "stage_opening_type": components["stage_opening_type_loss"],
        "stage_furniture": components["stage_furniture_loss"],
        "routing": components["routing_regularization_loss"],
        "furniture_query": components["furniture_query_loss"],
        "furniture_query_class": components["room_mask_class_loss"],
        "furniture_query_mask": components["room_mask_dice_bce_loss"],
        "furniture_query_unmatched": components["room_mask_unmatched_loss"],
        "furniture_query_tversky": components["room_mask_tversky_loss"],
        "furniture_query_containment": components["furniture_query_containment"],
        "furniture_room_assignment": components["furniture_room_assignment_loss"],
        "furniture_room_assignment_accuracy": components[
            "furniture_room_assignment_accuracy"
        ],
        "instance_room_class": components["instance_room_class"],
        "instance_room_mask": components["instance_room_mask"],
        "instance_room_unmatched": components.get(
            "instance_room_unmatched",
            0.0,
        ),
        "instance_room_tversky": components.get(
            "instance_room_tversky",
            0.0,
        ),
        "instance_room_matched_dice": components.get(
            "instance_room_matched_dice",
            0.0,
        ),
        "instance_room_matched_iou": components.get(
            "instance_room_matched_iou",
            0.0,
        ),
        "instance_room_predicted_area": components.get(
            "instance_room_predicted_area",
            0.0,
        ),
        "instance_room_target_area": components.get(
            "instance_room_target_area",
            0.0,
        ),
        "instance_room_unmatched_area": components.get(
            "instance_room_unmatched_area",
            0.0,
        ),
        "instance_room_centroid_error": components.get(
            "instance_room_centroid_error",
            0.0,
        ),
        "instance_room_reference_anchor": components.get(
            "instance_room_reference_anchor",
            0.0,
        ),
        "instance_room_mask_centroid": components.get(
            "instance_room_mask_centroid",
            0.0,
        ),
        "instance_room_query_separation": components.get(
            "instance_room_query_separation",
            0.0,
        ),
        "instance_room_partition": components["instance_room_partition"],
        "room_query_class_cardinality": components.get(
            "room_query_class_cardinality_consistency",
            0.0,
        ),
        "room_query_mask_only_assignment_disagreement": components.get(
            "room_query_mask_only_assignment_disagreement",
            0.0,
        ),
        "room_query_count_selected_target_recall": components.get(
            "room_query_count_selected_target_recall",
            0.0,
        ),
        "room_query_count_selected_background_rate": components.get(
            "room_query_count_selected_background_rate",
            0.0,
        ),
        "room_query_count_selection_available": components.get(
            "room_query_count_selection_available",
            0.0,
        ),
        "room_query_selection_alignment": components.get(
            "room_query_selection_alignment",
            0.0,
        ),
        "room_query_presence_rank": components.get(
            "room_query_presence_rank",
            0.0,
        ),
        "room_query_presence_count_regression": components.get(
            "room_query_presence_count_regression",
            0.0,
        ),
        "room_shared_match_count": components.get(
            "room_shared_match_count",
            0.0,
        ),
        "room_valid_target_count": components.get(
            "room_valid_target_count",
            0.0,
        ),
        "room_query_gate_output_present": components.get(
            "room_query_gate_output_present",
            0.0,
        ),
        "room_query_set_total": components.get(
            "room_query_set_total",
            components.get("room_query_set_total_loss", 0.0),
        ),
        "room_adjacency": components.get("room_adjacency_total", 0.0),
        "room_adjacency_total": components.get("room_adjacency_total", 0.0),
        "room_adjacency_consistency_loss": components.get(
            "room_adjacency_consistency_loss",
            0.0,
        ),
        "room_adjacency_valid_pairs": components.get(
            "room_adjacency_valid_pairs",
            0.0,
        ),
        "room_adjacency_assigned_queries": components.get(
            "room_adjacency_assigned_queries",
            0.0,
        ),
        "room_type_adjacency": components.get(
            "room_type_adjacency_total",
            0.0,
        ),
        "room_type_adjacency_total": components.get(
            "room_type_adjacency_total",
            0.0,
        ),
        "room_type_adjacency_loss": components.get(
            "room_type_adjacency_loss",
            0.0,
        ),
        "room_type_adjacency_valid_pairs": components.get(
            "room_type_adjacency_valid_pairs",
            0.0,
        ),
        "room_type_adjacency_target_mean": components.get(
            "room_type_adjacency_target_mean",
            0.0,
        ),
        "room_type_node": components.get("room_type_node_total", 0.0),
        "room_type_node_total": components.get(
            "room_type_node_total",
            0.0,
        ),
        "room_type_node_loss": components.get(
            "room_type_node_loss",
            0.0,
        ),
        "dense_type_adjacency": components.get("dense_type_edge_loss", 0.0),
        "dense_type_node": components.get("dense_type_node_loss", 0.0),
        "dense_type_total": components.get("dense_type_total", 0.0),
        "dense_type_edge_target_mean": components.get(
            "dense_type_edge_target_mean",
            0.0,
        ),
    }


def materialize_train_component_totals(
    totals: dict[str, float | torch.Tensor],
    samples: int,
) -> dict[str, float]:
    """Transfer the complete scalar vector to the host once per epoch."""

    keys = sorted(totals)
    values = [
        value.detach().float().reshape(())
        if torch.is_tensor(value)
        else torch.tensor(float(value))
        for value in (totals[key] for key in keys)
    ]
    host = torch.stack(values).cpu().tolist() if values else []
    return {
        key: float(value) / max(1, int(samples))
        for key, value in zip(keys, host, strict=True)
    }


__all__ = [
    "GATE_TARGETS",
    "accumulate_train_component_totals",
    "compose_stage_labels",
    "empty_train_component_totals",
    "family_metrics",
    "materialize_train_component_totals",
    "selection_score",
    "train_component_values",
]
