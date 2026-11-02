"""Shared RPLAN mask-partition labels and validation metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_metric_suite import matched_instance_metrics


def mask_partition_labels(
    probabilities: torch.Tensor,
    class_ids: torch.Tensor,
    footprint: torch.Tensor,
    *,
    confidence_threshold: float = 0.5,
) -> torch.Tensor:
    """Assign each in-footprint pixel to the strongest selected room query."""

    if probabilities.ndim != 3:
        raise ValueError("probabilities must be [R,H,W]")
    if class_ids.ndim != 1 or class_ids.shape[0] != probabilities.shape[0]:
        raise ValueError("class_ids must match the room dimension")
    if footprint.shape != probabilities.shape[-2:]:
        raise ValueError("footprint must match the probability resolution")
    owner = probabilities.argmax(dim=0)
    confidence = probabilities.max(dim=0).values
    support = footprint.bool() & (confidence >= float(confidence_threshold))
    labels = torch.zeros_like(owner, dtype=torch.long)
    for index, class_id in enumerate(class_ids):
        labels[(owner == index) & support] = int(class_id)
    return labels


@torch.no_grad()
def mask_partition_query_metrics(
    predictions: dict[str, torch.Tensor],
    target_labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    confidence_threshold: float = 0.5,
    minimum_area: int = 4,
) -> dict[str, float]:
    """Score the mask-partition readout with the room instance protocol."""

    partition_probabilities = predictions.get(
        "room_instance_partition_probabilities"
    )
    probabilities = (
        partition_probabilities
        if partition_probabilities is not None
        else predictions["room_instance_mask_logits"].sigmoid()
    )
    probabilities = F.interpolate(
        probabilities,
        size=target_labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    footprint = F.interpolate(
        footprint.float(),
        size=target_labels.shape[-2:],
        mode="nearest",
    )[:, 0]
    predicted_count = predictions["room_count_logits"].argmax(dim=-1).clamp(
        1,
        predictions["room_exists"].shape[1],
    )
    room_ids = torch.tensor(
        taxonomy.room_ids,
        device=target_labels.device,
    )
    rows = []
    for index, count in enumerate(predicted_count):
        selected = torch.topk(
            predictions["room_exists"][index],
            k=int(count),
        ).indices
        class_ids = room_ids[
            predictions["room_class"][index, selected].argmax(dim=-1)
        ]
        labels = mask_partition_labels(
            probabilities[index, selected],
            class_ids,
            footprint[index],
            confidence_threshold=(
                0.0
                if partition_probabilities is not None
                else confidence_threshold
            ),
        )
        rows.append(
            matched_instance_metrics(
                labels.cpu().numpy(),
                target_labels[index].cpu().numpy(),
                taxonomy,
                min_area=minimum_area,
            )
        )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "count_abs_error",
            "matched_geometric_iou",
            "instance_f1_at_threshold",
            "matched_type_accuracy",
        )
    }


__all__ = ["mask_partition_labels", "mask_partition_query_metrics"]
