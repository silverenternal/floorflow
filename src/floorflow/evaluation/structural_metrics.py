"""Structural validity metrics for generated floorplans."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as component_label

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.core import footprint_band


def connected_component_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    _components, count = component_label(mask, structure=structure)
    return int(count)


def structural_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    room_support: torch.Tensor | None = None,
) -> dict[str, float]:
    device = prediction.device
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=device)
    wall_id = taxonomy.class_id("wall")
    wall = (prediction == wall_id).float()
    rooms = torch.isin(prediction, room_ids).float()
    furniture = torch.isin(prediction, furniture_ids).float()
    openings = torch.isin(prediction, opening_ids).float()
    wall_near = F.max_pool2d(wall[:, None], 7, stride=1, padding=3)[:, 0]
    envelope = footprint_band(footprint[:, None] if footprint.ndim == 3 else footprint, 7)[:, 0]
    wall_near = torch.maximum(wall_near, envelope)
    if room_support is None:
        room_support = torch.stack(
            [
                torch.from_numpy(binary_fill_holes(item.cpu().numpy())).to(device)
                for item in rooms.bool()
            ]
        )
    room_support = room_support.float()
    if room_support.ndim == 4:
        room_support = room_support[:, 0]
    room_support = F.max_pool2d(
        (room_support > 0.5).float()[:, None],
        25,
        stride=1,
        padding=12,
    )[:, 0] > 0.5
    opening_alignment = (openings * wall_near).sum(dim=(1, 2)) / openings.sum(dim=(1, 2)).clamp_min(1.0)
    furniture_support = (furniture * room_support.float()).sum(dim=(1, 2))
    furniture_mass = furniture.sum(dim=(1, 2))
    containment = torch.where(
        furniture_mass > 0,
        furniture_support / furniture_mass.clamp_min(1.0),
        torch.ones_like(furniture_mass),
    )

    predicted_areas = torch.stack(
        [(prediction == class_id).float().sum(dim=(1, 2)) for class_id in taxonomy.room_ids],
        dim=1,
    )
    target_areas = torch.stack(
        [(target == class_id).float().sum(dim=(1, 2)) for class_id in taxonomy.room_ids],
        dim=1,
    )
    predicted_fractions = predicted_areas / predicted_areas.sum(dim=1, keepdim=True).clamp_min(1.0)
    target_fractions = target_areas / target_areas.sum(dim=1, keepdim=True).clamp_min(1.0)
    area_mae = (predicted_fractions - target_fractions).abs().mean(dim=1)

    pred_components = []
    target_components = []
    room_union_pred = rooms.bool().cpu().numpy()
    room_union_target = torch.isin(target, room_ids).cpu().numpy()
    for index in range(prediction.shape[0]):
        pred_components.append(connected_component_count(room_union_pred[index]))
        target_components.append(connected_component_count(room_union_target[index]))
    component_gap = np.abs(np.asarray(pred_components) - np.asarray(target_components))
    return {
        "opening_wall_alignment": float(opening_alignment.mean().cpu()),
        "furniture_room_containment": float(containment.mean().cpu()),
        "room_area_distribution_mae": float(area_mae.mean().cpu()),
        "room_component_gap": float(component_gap.mean()),
    }
