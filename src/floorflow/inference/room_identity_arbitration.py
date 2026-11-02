"""Target-free arbitration between dense and instance-query room predictions."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label

from floorflow.recursion.trust_gate import (
    RoomGraphTrustGate,
    load_room_graph_trust_gate,
)

ROOM_IDENTITY_FEATURE_NAMES = (
    "dense_confidence",
    "query_confidence",
    "routed_confidence",
    "confidence_gain",
    "dense_query_agreement",
    "dense_routed_agreement",
    "dense_entropy",
    "routed_entropy",
    "room_support_disagreement",
    "boundary_room_fraction",
    "query_count_delta",
    "routed_count_delta",
    "query_tiny_component_rate",
    "routed_tiny_component_rate",
)


def _component_stats(
    labels: np.ndarray,
    room_ids: tuple[int, ...],
    min_area: int,
) -> tuple[int, float]:
    total = 0
    exportable = 0
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    for class_id in room_ids:
        components, count = label(labels == int(class_id), structure=structure)
        total += int(count)
        if count:
            sizes = np.bincount(components.ravel())[1:]
            exportable += int((sizes >= int(min_area)).sum())
    tiny_rate = 1.0 - exportable / total if total else 0.0
    return exportable, float(tiny_rate)


@torch.no_grad()
def room_identity_arbitration_features(
    dense_logits: torch.Tensor,
    query_logits: torch.Tensor,
    routed_logits: torch.Tensor,
    footprint: torch.Tensor,
    room_ids: tuple[int, ...],
    *,
    min_area: int = 8,
) -> torch.Tensor:
    """Return target-free per-case evidence for selecting the routed branch."""

    if dense_logits.shape != query_logits.shape or dense_logits.shape != routed_logits.shape:
        raise ValueError("dense, query, and routed logits must share one shape")
    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    valid = footprint.bool()
    dense_probability = dense_logits.float().softmax(dim=1)
    query_probability = query_logits.float().softmax(dim=1)
    routed_probability = routed_logits.float().softmax(dim=1)
    dense_confidence = dense_probability.amax(dim=1)
    query_confidence = query_probability.amax(dim=1)
    routed_confidence = routed_probability.amax(dim=1)
    dense_entropy = -(
        dense_probability.clamp_min(1.0e-8)
        * dense_probability.clamp_min(1.0e-8).log()
    ).sum(dim=1) / math.log(max(2, dense_logits.shape[1]))
    routed_entropy = -(
        routed_probability.clamp_min(1.0e-8)
        * routed_probability.clamp_min(1.0e-8).log()
    ).sum(dim=1) / math.log(max(2, routed_logits.shape[1]))
    dense_labels = dense_probability.argmax(dim=1)
    query_labels = query_probability.argmax(dim=1)
    routed_labels = routed_probability.argmax(dim=1)
    denominator = valid.flatten(1).sum(dim=1).clamp_min(1.0)
    boundary = valid & ~(
        -F.max_pool2d(
            -valid.float()[:, None],
            5,
            stride=1,
            padding=2,
        )
    )[:, 0].bool()
    dense_room = (dense_confidence >= 0.5) & valid
    routed_room = (routed_confidence >= 0.5) & valid

    def masked_mean(values: torch.Tensor) -> torch.Tensor:
        return (values * valid).flatten(1).sum(dim=1) / denominator

    batch_features = [
        masked_mean(dense_confidence),
        masked_mean(query_confidence),
        masked_mean(routed_confidence),
        masked_mean(routed_confidence - dense_confidence),
        ((dense_labels == query_labels) & valid).flatten(1).sum(dim=1)
        / denominator,
        ((dense_labels == routed_labels) & valid).flatten(1).sum(dim=1)
        / denominator,
        masked_mean(dense_entropy),
        masked_mean(routed_entropy),
        ((dense_room != routed_room) & valid).flatten(1).sum(dim=1)
        / denominator,
        ((routed_room & boundary).flatten(1).sum(dim=1) / denominator),
    ]
    room_id_tensor = torch.tensor(room_ids, device=dense_labels.device)
    dense_count_rows = []
    for dense_row, query_row, routed_row in zip(
        dense_labels,
        query_labels,
        routed_labels,
        strict=True,
    ):
        dense_labels_np = room_id_tensor[dense_row].cpu().numpy()
        query_labels_np = room_id_tensor[query_row].cpu().numpy()
        routed_labels_np = room_id_tensor[routed_row].cpu().numpy()
        dense_count, _dense_tiny = _component_stats(
            dense_labels_np,
            room_ids,
            min_area,
        )
        query_count, query_tiny = _component_stats(
            query_labels_np,
            room_ids,
            min_area,
        )
        routed_count, routed_tiny = _component_stats(
            routed_labels_np,
            room_ids,
            min_area,
        )
        dense_count_rows.append(
            [
                float(query_count - dense_count),
                float(routed_count - dense_count),
                query_tiny,
                routed_tiny,
            ]
        )
    component_features = torch.tensor(
        dense_count_rows,
        device=dense_logits.device,
        dtype=torch.float32,
    )
    return torch.cat([torch.stack(batch_features, dim=1), component_features], dim=1)


def load_room_identity_arbitration_gate(
    path: str | Path,
    device: torch.device,
) -> RoomGraphTrustGate:
    gate = load_room_graph_trust_gate(path, device)
    if gate.feature_size != len(ROOM_IDENTITY_FEATURE_NAMES):
        raise ValueError(
            f"expected {len(ROOM_IDENTITY_FEATURE_NAMES)} arbitration features, "
            f"got {gate.feature_size}"
        )
    return gate


@torch.no_grad()
def arbitrate_room_logits(
    gate: RoomGraphTrustGate,
    dense_logits: torch.Tensor,
    routed_logits: torch.Tensor,
    features: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = gate.trust(features) >= float(threshold)
    selected_logits = torch.where(
        selected[:, None, None, None],
        routed_logits,
        dense_logits,
    )
    return selected_logits, selected


__all__ = [
    "ROOM_IDENTITY_FEATURE_NAMES",
    "arbitrate_room_logits",
    "load_room_identity_arbitration_gate",
    "room_identity_arbitration_features",
]
