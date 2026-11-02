"""Validation metrics and checkpoint-selection scores for condition models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.flow.core import sample_floorplans
from floorflow.heads.room_mask_decoder import render_room_masks
from floorflow.heads.room_partition import (
    batch_room_partition_target,
    room_partition_metrics,
)
from floorflow.heads.room_program import room_presence_target
from floorflow.heads.structured_objects import (
    compose_structured_objects,
    constrain_structured_objects,
    object_group_metrics,
    object_group_predictions,
)
from floorflow.objectives.condition_objective import (
    ConditionObjectiveWeights,
    condition_layout_loss,
    partition_room_labels,
    room_area_coverage_from_labels,
    room_macro_iou_from_labels,
)
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.training.runtime_acceleration import RuntimeConfig


def selection_score(metrics: dict[str, float]) -> float:
    partition_iou = metrics.get("partition_room_macro_iou", metrics["room_macro_iou"])
    partition_coverage = metrics.get(
        "partition_room_area_coverage",
        metrics["room_area_coverage"],
    )
    coverage_shortfall = max(0.0, 0.95 - partition_coverage)
    return (
        partition_iou
        + 0.10 * metrics["opening_f1"]
        + 0.10 * metrics["furniture_iou"]
        + 0.05 * metrics["wall_iou"]
        - 0.25 * coverage_shortfall
    )


def case_metrics(
    aux: dict,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    room_labels = partition_room_labels(aux["room_partition_logits"], taxonomy) * footprint.long()
    partition_metrics = {
        "partition_room_macro_iou": room_macro_iou_from_labels(room_labels, target, taxonomy),
        "partition_room_area_coverage": room_area_coverage_from_labels(
            room_labels,
            footprint,
            taxonomy,
        ),
    }
    objects = object_group_predictions(
        aux["structured_objects"],
        taxonomy,
        opening_threshold=0.5,
        furniture_threshold=0.5,
        score_scale="probability",
    )
    objects = constrain_structured_objects(objects, room_labels, footprint, taxonomy)
    prediction = compose_structured_objects(room_labels, objects, footprint, taxonomy)
    metrics = floorplan_metrics(prediction, target, footprint, taxonomy)
    metrics["room_area_coverage_footprint"] = room_area_coverage_from_labels(
        prediction,
        footprint,
        taxonomy,
    )
    metrics.update(partition_metrics)
    hierarchy = aux.get("hierarchical_rooms")
    if isinstance(hierarchy, dict):
        dense_target = batch_room_partition_target(
            target,
            footprint,
            taxonomy,
        )
        for name in ("coarse_room_logits", "refined_room_logits"):
            logits = hierarchy.get(name)
            if logits is None:
                continue
            full_resolution_logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            room_ids = torch.tensor(
                taxonomy.room_ids,
                device=target.device,
            )
            full_resolution_labels = (
                room_ids[full_resolution_logits.argmax(dim=1)]
                * footprint.long()
            )
            metrics[f"{name.removesuffix('_logits')}_iou"] = (
                room_macro_iou_from_labels(
                    full_resolution_labels,
                    target,
                    taxonomy,
                )
            )
            coarse_target = F.interpolate(
                dense_target.float()[:, None],
                size=logits.shape[-2:],
                mode="nearest",
            )[:, 0].long()
            hierarchy_metrics = room_partition_metrics(
                logits,
                coarse_target,
                taxonomy,
            )
            metrics[f"{name.removesuffix('_logits')}_grid_iou"] = (
                hierarchy_metrics["room_partition_iou"]
            )
    metrics.update(object_group_metrics(objects, target, taxonomy))
    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
    room_weights: torch.Tensor,
    positive_weights: torch.Tensor,
    objective_weights: ConditionObjectiveWeights,
    retrieval_memory: ContourPartitionMemory | None = None,
    retrieval_weight: float = 0.0,
    retrieval_top_k: int = 1,
    runtime: RuntimeConfig | None = None,
    flow_steps: int = 0,
    flow_noise_scale: float = 0.75,
    program_conditioning: bool = False,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    model.eval()
    weighted_loss = 0.0
    weighted_components: dict[str, float] = {}
    seen = 0
    rows: list[dict[str, float | str]] = []
    for batch in loader:
        if len(batch) == 6:
            target, footprint, floor_ids, names, room_support, occlusion_mask = batch
            room_support = room_support.to(device)
        else:
            target, footprint, floor_ids, names = batch
            room_support = None
            occlusion_mask = None
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        if flow_steps > 0:
            model_state, _trajectory = sample_floorplans(
                model,
                footprint,
                steps=flow_steps,
                noise_scale=flow_noise_scale,
                floor_ids=floor_ids,
                objective_mode="velocity",
                semantic_decode_weight=0.0,
            )
            model_time = torch.ones(target.shape[0], device=device)
        else:
            model_state = torch.zeros(
                target.shape[0],
                model.target_channels,
                *target.shape[-2:],
                device=device,
            )
            model_time = torch.zeros(target.shape[0], device=device)
        room_prior_logits = None
        if retrieval_memory is not None and retrieval_weight > 0:
            room_prior_logits = retrieval_memory.retrieve_logits(
                footprint.detach().cpu().numpy(),
                taxonomy,
                device,
                top_k=retrieval_top_k,
            )
        with (runtime or RuntimeConfig()).autocast(device):
            room_program = (
                room_presence_target(target, taxonomy)
                if program_conditioning
                else None
            )
            _, aux = model.forward_with_aux(
                model_state,
                model_time,
                footprint,
                floor_ids=floor_ids,
                layout_prior_logits=room_prior_logits,
                room_program=room_program,
            )
        with (runtime or RuntimeConfig()).autocast(device):
            loss, components, _terms = condition_layout_loss(
                aux,
                target,
                footprint,
                taxonomy,
                room_class_weights=room_weights,
                positive_class_weights=positive_weights,
                weights=objective_weights,
                room_prior_logits=room_prior_logits,
                room_prior_weight=retrieval_weight,
                room_support=room_support,
                occlusion_mask=occlusion_mask,
            )
        weighted_loss += float(loss.cpu()) * target.shape[0]
        for key, value in components.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                weighted_components[key] = weighted_components.get(key, 0.0) + (
                    float(value.detach().cpu()) * target.shape[0]
                )
        seen += target.shape[0]
        room_mask_labels = None
        if aux.get("room_mask_predictions") is not None:
            room_mask_labels = render_room_masks(
                aux["room_mask_predictions"],
                taxonomy,
                footprint,
            )
        for index, name in enumerate(names):
            adjusted_logits = aux["room_partition_logits"][index : index + 1]
            if room_prior_logits is not None:
                adjusted_logits = (
                    adjusted_logits
                    + retrieval_weight * room_prior_logits[index : index + 1]
                )
            case_aux = {
                "room_partition_logits": adjusted_logits,
                "structured_objects": {
                    key: value[index : index + 1]
                    for key, value in aux["structured_objects"].items()
                },
            }
            hierarchy = aux.get("hierarchical_rooms")
            if isinstance(hierarchy, dict):
                case_aux["hierarchical_rooms"] = {
                    key: (
                        value[index : index + 1]
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in hierarchy.items()
                }
            metrics = case_metrics(
                case_aux,
                target[index : index + 1],
                footprint[index : index + 1],
                taxonomy,
            )
            if room_mask_labels is not None:
                metrics["room_mask_room_iou"] = room_macro_iou_from_labels(
                    room_mask_labels[index : index + 1],
                    target[index : index + 1],
                    taxonomy,
                )
            metrics["name"] = name
            metrics["selection_score"] = selection_score(metrics)
            rows.append(metrics)
    summary = {"loss": weighted_loss / max(1, seen)}
    for key, value in weighted_components.items():
        summary[key] = value / max(1, seen)
    for key in (
        "room_macro_iou",
        "room_group_macro_iou",
        "room_area_coverage",
        "room_area_coverage_footprint",
        "present_class_macro_iou",
        "partition_room_macro_iou",
        "partition_room_area_coverage",
        "coarse_room_iou",
        "refined_room_iou",
        "coarse_room_grid_iou",
        "refined_room_grid_iou",
        "room_mask_room_iou",
        "opening_f1",
        "furniture_iou",
        "wall_iou",
        "interior_wall_iou",
        "room_component_gap",
        "selection_score",
    ):
        if not any(key in row for row in rows):
            continue
        values = [float(row[key]) for row in rows if key in row]
        summary[key] = sum(values) / max(1, len(values))
    return summary, rows
