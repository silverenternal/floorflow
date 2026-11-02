"""Floorplan reconstruction and design metrics."""

from __future__ import annotations

import torch

from floorflow.data.classes import CLASS_NAMES, FURNITURE_CLASS_IDS, ROOM_CLASS_IDS
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.structural_metrics import structural_metrics
from floorflow.evaluation.wall_metrics import wall_fields
from floorflow.geometry.core import footprint_band, interior_mask
from floorflow.heads.room_groups import room_group_macro_iou


def _binary_iou(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.bool()
    target = target.bool()
    intersection = (prediction & target).sum().float()
    union = (prediction | target).sum().float()
    return intersection / union.clamp_min(1.0)


def floorplan_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy | None = None,
    room_support: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute stable, paper-friendly metrics for the initial raster baseline."""

    prediction = prediction.long()
    target = target.long()
    footprint_bool = footprint.bool()
    values: dict[str, float] = {}
    interior = interior_mask(footprint, boundary_width=7).bool()
    values["interior_pixel_accuracy"] = float(
        ((prediction == target) & interior).sum().detach().cpu()
    ) / max(1, int(interior.sum().detach().cpu()))

    class_names = taxonomy.class_names if taxonomy else CLASS_NAMES
    wall_id = taxonomy.class_id("wall") if taxonomy else 1
    room_ids = taxonomy.room_ids if taxonomy else ROOM_CLASS_IDS
    furniture_ids = taxonomy.furniture_ids if taxonomy else FURNITURE_CLASS_IDS
    opening_ids = taxonomy.opening_ids if taxonomy else (2, 3)
    values["interior_wall_iou"] = float(
        _binary_iou((prediction == wall_id) & interior, (target == wall_id) & interior).detach().cpu()
    )
    furniture_prediction = torch.isin(prediction, torch.tensor(furniture_ids, device=prediction.device))
    furniture_target = torch.isin(target, torch.tensor(furniture_ids, device=target.device))
    values["furniture_iou"] = float(_binary_iou(furniture_prediction, furniture_target).detach().cpu())

    num_classes = len(class_names)
    flat_prediction = prediction.flatten().long()
    flat_target = target.flatten().long()
    encoded = flat_prediction * num_classes + flat_target
    confusion = torch.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes,
        num_classes,
    )
    predicted_counts = confusion.sum(dim=1)
    target_counts = confusion.sum(dim=0)
    class_intersection = torch.diagonal(confusion)
    class_iou_values = class_intersection / (
        predicted_counts + target_counts - class_intersection
    ).clamp_min(1.0)
    room_indices = torch.tensor(room_ids, device=prediction.device)
    present_rooms = room_indices[target_counts[room_indices] > 0]
    values["room_macro_iou"] = (
        float(class_iou_values[present_rooms].mean().detach().cpu()) if len(present_rooms) else 0.0
    )
    values["room_group_macro_iou"] = room_group_macro_iou(prediction, target, taxonomy) if taxonomy else 0.0
    room_prediction = torch.isin(prediction, torch.tensor(room_ids, device=prediction.device))
    room_target = torch.isin(target, torch.tensor(room_ids, device=target.device))
    target_room_mass = (room_target & interior).sum().float()
    values["room_area_coverage"] = float(
        (room_prediction & room_target & interior).sum().float()
        / target_room_mass.clamp_min(1.0).float()
    )
    values["room_pixel_precision"] = float(
        (room_prediction & room_target & interior).sum().float()
        / (room_prediction & interior).sum().clamp_min(1.0).float()
    )
    values["room_generated_occupancy"] = float(
        (room_prediction & interior).sum().float() / interior.sum().clamp_min(1).float()
    )
    values["target_room_occupancy"] = float(
        target_room_mass / interior.sum().clamp_min(1).float()
    )

    opening_pred = torch.isin(prediction, torch.tensor(opening_ids, device=prediction.device))
    opening_target = torch.isin(target, torch.tensor(opening_ids, device=target.device))
    intersection = (opening_pred & opening_target).sum().float()
    precision = intersection / opening_pred.sum().float().clamp_min(1.0)
    recall = intersection / opening_target.sum().float().clamp_min(1.0)
    values["opening_f1"] = float((2.0 * precision * recall / (precision + recall).clamp_min(1.0e-6)).cpu())

    outside = ~footprint_bool
    values["outside_footprint_fraction"] = (
        float((prediction[outside] != 0).float().mean().cpu())
        if outside.any()
        else 0.0
    )
    band = footprint_band(footprint, kernel_size=5)[:, 0].bool()
    opening_ids_tensor = torch.tensor(opening_ids, device=prediction.device)
    envelope_valid = (prediction == wall_id) | torch.isin(prediction, opening_ids_tensor)
    values["exterior_band_is_known_wall"] = (
        float(envelope_valid[band].float().mean().cpu()) if band.any() else 1.0
    )
    values["exterior_band_opening_fraction"] = (
        float(torch.isin(prediction[band], opening_ids_tensor).float().mean().cpu()) if band.any() else 0.0
    )
    class_iou = []
    for class_id, name in enumerate(class_names):
        if class_id == 0:
            continue
        if int(target_counts[class_id].detach().cpu()) == 0:
            continue
        class_iou.append((class_iou_values[class_id], name))
    if class_iou:
        for value, name in class_iou:
            values[f"iou_{name}"] = float(value.detach().cpu())
        values["present_class_macro_iou"] = float(torch.stack([value for value, _ in class_iou]).mean().cpu())
    else:
        values["present_class_macro_iou"] = 0.0
    if taxonomy is not None:
        values.update(
            structural_metrics(
                prediction,
                target,
                footprint,
                taxonomy,
                room_support=room_support,
            )
        )
        values.update(wall_fields(prediction, target, taxonomy))
    return values
