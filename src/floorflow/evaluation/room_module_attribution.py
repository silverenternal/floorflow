"""Metrics for counterfactual comparison of room-generation modules."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.structured_objects import constrain_structured_objects
from floorflow.objectives.condition_objective import room_macro_iou_from_labels


def _room_component_count(mask: np.ndarray) -> int:
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    _components, count = label(np.asarray(mask, dtype=bool), structure=structure)
    return int(count)


def room_support_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    labels = prediction.long() * footprint.long()
    room_ids = torch.tensor(taxonomy.room_ids, device=prediction.device)
    target_room = torch.isin(target, room_ids)
    predicted_room = torch.isin(labels, room_ids)
    coverage = (predicted_room & target_room).sum().float() / target_room.sum().clamp_min(1).float()
    component_gap = abs(
        _room_component_count(predicted_room[0].cpu().numpy())
        - _room_component_count(target_room[0].cpu().numpy())
    )
    return {
        "room_macro_iou": room_macro_iou_from_labels(labels, target, taxonomy),
        "room_coverage": float(coverage.cpu()),
        "component_gap": float(component_gap),
    }


def labels_from_room_logits(
    logits: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    room_ids = torch.tensor(taxonomy.room_ids, device=logits.device)
    return room_ids[logits.argmax(dim=1)] * footprint.long()


def room_objects(
    aux: dict,
    taxonomy: FloorplanTaxonomy,
    room_labels: torch.Tensor,
    footprint: torch.Tensor,
) -> dict:
    from floorflow.heads.structured_objects import object_group_predictions

    objects = object_group_predictions(
        aux["structured_objects"],
        taxonomy,
        opening_threshold=0.7,
        furniture_threshold=0.7,
        score_scale="probability",
    )
    return constrain_structured_objects(
        objects,
        room_labels,
        footprint,
        taxonomy,
    )
