"""Reusable Hungarian room-query assignments for identity and polygon heads."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from floorflow.objectives.gpu_assignment import exact_assignment_batch
from floorflow.objectives.room_assignment_cost import pairwise_mask_cost
from floorflow.objectives.room_matching_types import RoomQueryMatch


def match_room_queries(
    class_logits: torch.Tensor,
    polygons: torch.Tensor,
    target_classes: torch.Tensor,
    target_polygons: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    class_cost_weight: float = 1.0,
    polygon_cost_weight: float = 2.0,
    target_counts: Sequence[int] | None = None,
    predicted_masks: torch.Tensor | None = None,
    target_masks: torch.Tensor | None = None,
    mask_cost_weight: float = 0.0,
) -> list[RoomQueryMatch]:
    """Match room queries with the exact cost used by polygon supervision."""

    if class_logits.ndim != 3 or polygons.ndim != 4:
        raise ValueError("predictions must be class [B,Q,C] and polygon [B,Q,P,2]")
    if target_classes.shape != target_valid.shape:
        raise ValueError("target classes and validity must share shape")
    if target_polygons.ndim != 4 or target_polygons.shape[0] != class_logits.shape[0]:
        raise ValueError("target polygons must be [B,T,P,2]")
    if (predicted_masks is None) != (target_masks is None):
        raise ValueError("predicted and target masks must be provided together")
    if predicted_masks is not None and target_masks is not None:
        if predicted_masks.ndim != 4 or target_masks.ndim != 4:
            raise ValueError("room masks must be [B,N,H,W]")
        if predicted_masks.shape[0] != class_logits.shape[0]:
            raise ValueError("predicted masks must match the prediction batch")
        if target_masks.shape[0] != class_logits.shape[0]:
            raise ValueError("target masks must match the prediction batch")
    probability = class_logits.softmax(dim=-1)
    counts = (
        [int(value) for value in target_counts]
        if target_counts is not None
        else [int(value) for value in target_valid.sum(dim=1).tolist()]
    )
    if len(counts) != class_logits.shape[0]:
        raise ValueError("target_counts must match the prediction batch")
    costs = []
    populated_batches = []
    for batch, count in enumerate(counts):
        if count == 0:
            continue
        class_cost = -probability[batch][:, target_classes[batch, :count]]
        polygon_cost = torch.cdist(
            polygons[batch].flatten(1),
            target_polygons[batch, :count].flatten(1),
            p=1,
        )
        cost = (
            float(class_cost_weight) * class_cost
            + float(polygon_cost_weight) * polygon_cost
        )
        if predicted_masks is not None and target_masks is not None:
            cost = cost + float(mask_cost_weight) * pairwise_mask_cost(
                predicted_masks[batch].sigmoid(),
                target_masks[batch, :count],
            )
        costs.append(cost)
        populated_batches.append(batch)
    solved = exact_assignment_batch(costs)
    solved_by_batch = dict(zip(populated_batches, solved))
    matches: list[RoomQueryMatch] = []
    for batch in range(class_logits.shape[0]):
        assignment = solved_by_batch.get(batch)
        if assignment is None:
            matches.append(
                RoomQueryMatch(
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64),
                )
            )
            continue
        matches.append(
            RoomQueryMatch(
                assignment[0],
                assignment[1],
            )
        )
    return matches


def match_room_mask_queries(
    class_logits: torch.Tensor,
    predicted_mask_logits: torch.Tensor,
    target_classes: torch.Tensor,
    target_masks: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    class_cost_weight: float = 1.0,
    mask_cost_weight: float = 2.0,
    target_counts: Sequence[int] | None = None,
) -> list[RoomQueryMatch]:
    """Match room queries with the exact class-plus-mask identity cost."""

    if class_logits.ndim != 3:
        raise ValueError("class logits must be [B,Q,C]")
    if predicted_mask_logits.ndim != 4:
        raise ValueError("predicted masks must be [B,Q,H,W]")
    if class_logits.shape[:2] != predicted_mask_logits.shape[:2]:
        raise ValueError("class and mask queries must share [B,Q]")
    if target_classes.ndim != 2 or target_valid.ndim != 2:
        raise ValueError("target classes and validity must be rank two")
    if target_classes.shape[0] != target_valid.shape[0]:
        raise ValueError("target classes and validity must share a batch")
    if target_masks.ndim != 4:
        raise ValueError("target masks must be [B,T,H,W]")
    if target_masks.shape[:2] != target_valid.shape:
        raise ValueError("target masks and validity must share [B,T]")
    if (
        target_classes.shape[0] != class_logits.shape[0]
        or target_masks.shape[0] != class_logits.shape[0]
    ):
        raise ValueError("targets and predictions must share a batch")
    if target_masks.shape[-2:] != predicted_mask_logits.shape[-2:]:
        target_masks = torch.nn.functional.interpolate(
            target_masks.float(),
            size=predicted_mask_logits.shape[-2:],
            mode="nearest",
        )
    counts = (
        [int(value) for value in target_counts]
        if target_counts is not None
        else [int(value) for value in target_valid.sum(dim=1).tolist()]
    )
    if len(counts) != class_logits.shape[0]:
        raise ValueError("target_counts must match the prediction batch")
    if counts and max(counts) > target_classes.shape[1]:
        raise ValueError("target class tensor is shorter than the valid count")
    class_probability = class_logits.float().softmax(dim=-1)
    predicted_probability = predicted_mask_logits.float().sigmoid()
    costs = []
    populated_batches = []
    for batch, count in enumerate(counts):
        count = min(max(0, count), target_valid.shape[1])
        if count == 0:
            continue
        class_cost = -class_probability[batch][
            :, target_classes[batch, :count].long()
        ]
        mask_cost = pairwise_mask_cost(
            predicted_probability[batch],
            target_masks[batch, :count],
        )
        costs.append(
            float(class_cost_weight) * class_cost
            + float(mask_cost_weight) * mask_cost
        )
        populated_batches.append(batch)
    solved_by_batch = dict(
        zip(populated_batches, exact_assignment_batch(costs))
    )
    empty = (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
    )
    matches = [
        RoomQueryMatch(*solved_by_batch.get(batch, empty))
        for batch in range(class_logits.shape[0])
    ]
    for batch, count in enumerate(counts):
        if count > 0 and not len(matches[batch].predicted_indices):
            raise RuntimeError(
                "room shared matching dropped all valid targets"
            )
    return matches


__all__ = [
    "RoomQueryMatch",
    "match_room_mask_queries",
    "match_room_queries",
]
