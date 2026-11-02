"""Target-free reliability routing between predicted room-control branches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RoutedRoomControl:
    probabilities: torch.Tensor
    selected_candidate: torch.Tensor
    confidence: torch.Tensor
    agreement: torch.Tensor


def room_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError("room logits must be [B,C,H,W]")
    return logits.float().softmax(dim=1)


def footprint_confidence(
    probabilities: torch.Tensor,
    footprint: torch.Tensor,
) -> torch.Tensor:
    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    valid = footprint.bool()
    confidence = probabilities.amax(dim=1)
    return (confidence * valid).flatten(1).sum(dim=1) / valid.flatten(
        1
    ).sum(dim=1).clamp_min(1)


def label_agreement(
    primary: torch.Tensor,
    candidate: torch.Tensor,
    footprint: torch.Tensor,
) -> torch.Tensor:
    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    valid = footprint.bool()
    equal = (primary == candidate) & valid
    return equal.flatten(1).sum(dim=1) / valid.flatten(1).sum(
        dim=1
    ).clamp_min(1)


@torch.no_grad()
def route_room_control(
    primary_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    footprint: torch.Tensor,
    *,
    minimum_agreement: float = 0.0,
    confidence_margin: float = 0.0,
) -> RoutedRoomControl:
    """Select a whole candidate branch only when agreement and confidence agree."""

    if primary_logits.shape != candidate_logits.shape:
        raise ValueError("primary and candidate logits must share shape")
    primary_probability = room_probabilities(primary_logits)
    candidate_probability = room_probabilities(candidate_logits)
    primary_confidence = footprint_confidence(
        primary_probability,
        footprint,
    )
    candidate_confidence = footprint_confidence(
        candidate_probability,
        footprint,
    )
    agreement = label_agreement(
        primary_probability.argmax(dim=1),
        candidate_probability.argmax(dim=1),
        footprint,
    )
    selected = (agreement >= float(minimum_agreement)) & (
        candidate_confidence + float(confidence_margin)
        >= primary_confidence
    )
    probability = torch.where(
        selected[:, None, None, None],
        candidate_probability,
        primary_probability,
    )
    confidence = torch.where(
        selected,
        candidate_confidence,
        primary_confidence,
    )
    return RoutedRoomControl(
        probabilities=probability,
        selected_candidate=selected,
        confidence=confidence,
        agreement=agreement,
    )


def resize_room_logits(
    logits: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    if logits.shape[-2:] == tuple(size):
        return logits
    return F.interpolate(
        logits,
        size=size,
        mode="bilinear",
        align_corners=False,
    )


__all__ = [
    "RoutedRoomControl",
    "footprint_confidence",
    "label_agreement",
    "resize_room_logits",
    "room_probabilities",
    "route_room_control",
]
