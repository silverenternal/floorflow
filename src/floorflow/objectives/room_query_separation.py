"""Differentiable separation penalties for active room-instance queries."""

from __future__ import annotations

import torch

from floorflow.objectives.gpu_assignment import assignment_index_tensors
from floorflow.objectives.room_matching_types import RoomQueryMatch


def active_query_presence(class_logits: torch.Tensor) -> torch.Tensor:
    """Convert identity logits with a final no-object class into presence scores."""

    if class_logits.ndim != 3 or class_logits.shape[-1] < 2:
        raise ValueError("class_logits must be [B,Q,C+1]")
    foreground = torch.logsumexp(class_logits[..., :-1], dim=-1)
    return (foreground - class_logits[..., -1]).sigmoid()


def gated_active_query_presence(
    class_logits: torch.Tensor,
    existence_logits: torch.Tensor | None = None,
    selection: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use the count-selected query gate when the identity head exposes it."""

    presence = active_query_presence(class_logits)
    if existence_logits is None:
        return presence
    gate = (
        existence_logits.sigmoid()
        if selection is None
        else selection.to(presence.dtype).clamp(0.0, 1.0)
    )
    if gate.shape != presence.shape:
        raise ValueError("query gate must match [B,Q] presence")
    return presence * gate


def active_query_overlap_loss(
    mask_logits: torch.Tensor,
    presence_probability: torch.Tensor,
    *,
    class_probability: torch.Tensor | None = None,
    minimum_mass: float = 1.0e-4,
) -> torch.Tensor:
    """Penalize soft Dice overlap between jointly active room queries."""

    if mask_logits.ndim != 4:
        raise ValueError("mask_logits must be [B,Q,H,W]")
    if presence_probability.shape != mask_logits.shape[:2]:
        raise ValueError("presence_probability must be [B,Q]")
    probability = mask_logits.sigmoid().flatten(2)
    mass = probability.sum(dim=-1)
    intersection = torch.bmm(probability, probability.transpose(1, 2))
    denominator = mass[:, :, None] + mass[:, None, :]
    dice = 2.0 * intersection / denominator.clamp_min(float(minimum_mass))
    eye = torch.eye(
        mask_logits.shape[1],
        device=mask_logits.device,
        dtype=torch.bool,
    )[None]
    pair_weight = (
        presence_probability[:, :, None]
        * presence_probability[:, None, :]
    )
    if class_probability is not None:
        if class_probability.shape != (
            mask_logits.shape[0],
            mask_logits.shape[1],
            class_probability.shape[-1],
        ):
            raise ValueError("class_probability must be [B,Q,C]")
        class_probability = class_probability / class_probability.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-8)
        class_affinity = torch.bmm(
            class_probability,
            class_probability.transpose(1, 2),
        )
        pair_weight = pair_weight * class_affinity
    pair_weight = pair_weight.masked_fill(eye, 0.0)
    weighted = dice.clamp(0.0, 1.0) * pair_weight
    return weighted.sum() / pair_weight.sum().clamp_min(1.0)


def count_gated_query_separation(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    existence_logits: torch.Tensor | None = None,
    selection: torch.Tensor | None = None,
    *,
    class_aware: bool = False,
) -> torch.Tensor:
    """Return the target-free query separation diagnostic used by validation."""

    presence = gated_active_query_presence(
        class_logits,
        existence_logits,
        selection,
    )
    return active_query_overlap_loss(
        mask_logits,
        presence,
        class_probability=(
            class_logits[..., :-1].softmax(dim=-1)
            if class_aware
            else None
        ),
    )


def per_case_query_separation(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    existence_logits: torch.Tensor | None = None,
    selection: torch.Tensor | None = None,
    *,
    class_aware: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Return one target-free separation value per batch case."""

    if class_logits.shape[0] != mask_logits.shape[0]:
        raise ValueError("query separation batch size must match")
    return tuple(
        count_gated_query_separation(
            class_logits[index : index + 1],
            mask_logits[index : index + 1],
            None if existence_logits is None else existence_logits[index : index + 1],
            None if selection is None else selection[index : index + 1],
            class_aware=class_aware,
        )
        for index in range(class_logits.shape[0])
    )


def balanced_room_class_weights(
    target_classes: torch.Tensor,
    target_valid: torch.Tensor,
    num_classes: int,
    *,
    power: float = 0.5,
) -> torch.Tensor:
    """Inverse-frequency identity weights with unit mean over valid rooms."""

    if target_classes.shape != target_valid.shape or target_classes.ndim != 2:
        raise ValueError("target classes and validity must be [B,Q]")
    valid_classes = target_classes[target_valid].long()
    counts = torch.bincount(
        valid_classes,
        minlength=int(num_classes),
    ).to(dtype=torch.float32)
    present = counts > 0
    frequency = counts / counts.sum().clamp_min(1.0)
    safe_frequency = torch.where(
        present,
        frequency,
        torch.ones_like(frequency),
    )
    raw = safe_frequency.rsqrt().pow(float(power))
    present_mean = (
        raw * present.to(raw.dtype)
    ).sum() / present.sum().clamp_min(1.0)
    weights = torch.where(
        present,
        raw / present_mean.clamp_min(1.0e-8),
        torch.ones_like(raw),
    )
    return weights


def supervised_same_class_overlap_loss(
    mask_logits: torch.Tensor,
    matches: list[RoomQueryMatch],
    target_classes: torch.Tensor,
    *,
    minimum_mass: float = 1.0e-4,
) -> torch.Tensor:
    """Separate only Hungarian-matched queries assigned to the same room class."""

    if mask_logits.ndim != 4 or len(matches) != mask_logits.shape[0]:
        raise ValueError("mask logits and Hungarian matches must share a batch")
    if target_classes.shape[0] != mask_logits.shape[0]:
        raise ValueError("target classes must match the mask batch")
    probability = mask_logits.sigmoid().flatten(2)
    mass = probability.sum(dim=-1)
    intersection = torch.bmm(probability, probability.transpose(1, 2))
    dice = 2.0 * intersection / (
        mass[:, :, None] + mass[:, None, :]
    ).clamp_min(float(minimum_mass))
    pair_weight = mask_logits.new_zeros(
        (mask_logits.shape[0], mask_logits.shape[1], mask_logits.shape[1])
    )
    pairs = [
        (match.predicted_indices, match.target_indices)
        for match in matches
    ]
    queries, targets, valid = assignment_index_tensors(
        pairs,
        device=mask_logits.device,
    )
    classes = target_classes.gather(1, targets)
    same_class = (
        classes[:, :, None].eq(classes[:, None, :])
        & valid[:, :, None]
        & valid[:, None, :]
    )
    pair_index = (
        queries[:, :, None] * mask_logits.shape[1]
        + queries[:, None, :]
    )
    batch_offset = (
        torch.arange(mask_logits.shape[0], device=mask_logits.device)
        * mask_logits.shape[1]
        * mask_logits.shape[1]
    )[:, None, None]
    pair_weight.reshape(-1).index_add_(
        0,
        (pair_index + batch_offset).reshape(-1),
        same_class.to(pair_weight.dtype).reshape(-1),
    )
    pair_weight.clamp_(max=1.0)
    eye = torch.eye(
        mask_logits.shape[1],
        device=mask_logits.device,
        dtype=torch.bool,
    )[None]
    pair_weight = pair_weight.masked_fill(eye, 0.0)
    weighted = dice.clamp(0.0, 1.0) * pair_weight
    return weighted.sum() / pair_weight.sum().clamp_min(1.0)


__all__ = [
    "active_query_overlap_loss",
    "active_query_presence",
    "balanced_room_class_weights",
    "count_gated_query_separation",
    "gated_active_query_presence",
    "per_case_query_separation",
    "supervised_same_class_overlap_loss",
]
