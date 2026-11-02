"""CAD-facing metrics for composed distribution overlay labels."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion

from floorflow.data.taxonomy import FURNITURE_RARE, ROOM_RARE
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.room_groups import room_group_macro_iou
from floorflow.representation.core import room_area_distribution


def room_metrics(
    labels: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    room_labels: torch.Tensor,
    taxonomy,
) -> dict[str, float]:
    """Measure visible semantics, support and Hungarian room instances."""

    room_ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    predicted_support = torch.isin(labels, room_ids).cpu().numpy()
    target_support = torch.isin(target, room_ids).cpu().numpy()
    predicted_boundary = predicted_support & ~binary_erosion(
        predicted_support,
        structure=np.ones((3, 3), dtype=bool),
    )
    target_boundary = target_support & ~binary_erosion(
        target_support,
        structure=np.ones((3, 3), dtype=bool),
    )
    boundary_denominator = predicted_boundary.sum() + target_boundary.sum()
    boundary_f1 = (
        float(
            2.0
            * (predicted_boundary & target_boundary).sum()
            / boundary_denominator
        )
        if boundary_denominator
        else 1.0
    )
    suite = room_metric_suite(
        labels.cpu().numpy(),
        target.cpu().numpy(),
        footprint.cpu().numpy(),
        taxonomy,
        predicted_room_support=torch.isin(room_labels, room_ids).cpu().numpy(),
        predicted_instance_support=torch.isin(
            room_labels,
            room_ids,
        ).cpu().numpy(),
        target_room_support=torch.isin(target, room_ids).cpu().numpy(),
    )
    instance = suite["matched_instance"]
    rare_ids = np.asarray(
        [
            taxonomy.class_id(name)
            for name in (ROOM_RARE, FURNITURE_RARE)
            if taxonomy.class_id(name) > 0
        ],
        dtype=np.int64,
    )
    rare_present = np.isin(target.cpu().numpy(), rare_ids)
    rare_predicted = np.isin(labels.cpu().numpy(), rare_ids)
    rare_recall = (
        float((rare_present & rare_predicted).sum() / rare_present.sum())
        if rare_present.any()
        else 1.0
    )
    return {
        "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
        "instance_geometric_iou": instance["matched_geometric_iou"],
        "instance_room_f1": instance["instance_f1_at_threshold"],
        "instance_type_accuracy": instance["matched_type_accuracy"],
        "room_group_macro_iou": room_group_macro_iou(
            labels[None],
            target[None],
            taxonomy,
        ),
        "room_component_gap": abs(
            float(instance["predicted_instances"])
            - float(instance["target_instances"])
        ),
        "class_constrained_iou": instance["matched_class_constrained_iou"],
        "room_support_iou": suite["occlusion_aware_support"][
            "room_support_iou"
        ],
        "boundary_f1": boundary_f1,
        "rare_class_recall": rare_recall,
    }


def pixel_objects(
    labels: torch.Tensor,
    taxonomy,
) -> dict[str, torch.Tensor]:
    """Split wall, opening and furniture channels for object metrics."""

    opening = torch.isin(
        labels,
        torch.tensor(taxonomy.opening_ids, device=labels.device),
    )
    furniture = torch.isin(
        labels,
        torch.tensor(taxonomy.furniture_ids, device=labels.device),
    )
    return {
        "wall": (labels == taxonomy.class_id("wall")).long(),
        "opening": torch.where(opening, labels, torch.zeros_like(labels)),
        "furniture": torch.where(furniture, labels, torch.zeros_like(labels)),
    }


def constraint_metrics(
    labels: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    wall: torch.Tensor,
    opening: torch.Tensor,
    furniture: torch.Tensor,
    taxonomy,
    predicted_room_support: torch.Tensor | None = None,
) -> dict[str, float]:
    """Measure footprint, containment and host-wall constraints."""

    room_ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    room_support = torch.isin(labels, room_ids)
    if predicted_room_support is not None:
        room_support = predicted_room_support.bool()
    target_room_support = torch.isin(target, room_ids)
    footprint = footprint.bool()
    outside = int(((labels != 0) & ~footprint).sum())
    coverage = float(
        (room_support & footprint).sum() / footprint.sum().clamp_min(1)
    )
    target_recall = float(
        (room_support & target_room_support).sum()
        / target_room_support.sum().clamp_min(1)
    )
    wall_band = F.max_pool2d(
        (wall > 0).float()[None, None],
        kernel_size=5,
        stride=1,
        padding=2,
    )[0, 0] > 0
    opening = opening > 0
    furniture = furniture > 0
    alignment = float(
        (opening & wall_band).sum() / opening.sum().clamp_min(1)
    )
    containment = float(
        (furniture & room_support).sum() / furniture.sum().clamp_min(1)
    )
    area_mae = float(
        np.abs(
            room_area_distribution(labels.cpu().numpy(), taxonomy)
            - room_area_distribution(target.cpu().numpy(), taxonomy)
        ).mean()
    )
    return {
        "outside_footprint_pixels": outside,
        "room_coverage": coverage,
        "room_target_recall": target_recall,
        "room_area_distribution_mae": area_mae,
        "opening_wall_alignment": alignment,
        "furniture_room_containment": containment,
    }


__all__ = ["constraint_metrics", "pixel_objects", "room_metrics"]
