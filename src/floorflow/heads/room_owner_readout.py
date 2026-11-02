"""Owner-preserving room readout for raster and wall consumers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.heads.room_mask_fusion import (
    fused_room_mask_logits,
    query_gate_probability,
)


def exclusive_query_owner_probability(
    mask_logits: torch.Tensor,
) -> torch.Tensor:
    """Normalize query masks against one background owner per pixel."""

    if mask_logits.ndim != 4:
        raise ValueError("mask logits must be [B,Q,H,W]")
    background = -mask_logits.amax(dim=1, keepdim=True)
    return torch.cat([mask_logits, background], dim=1).softmax(dim=1)[
        :, :-1
    ]


def _effective_query_mass(
    predictions: dict[str, torch.Tensor],
) -> torch.Tensor:
    probability = predictions["mask_logits"].sigmoid()
    existence = predictions.get("query_existence_logits")
    if existence is not None:
        gate = query_gate_probability(
            existence,
            predictions.get("query_count_selection_mask"),
        )
        probability = probability * gate[..., None, None]
    return probability


def _query_availability(
    predictions: dict[str, torch.Tensor],
    queries: int,
) -> torch.Tensor:
    existence = predictions.get("query_existence_logits")
    availability = (
        torch.ones(
            (predictions["mask_logits"].shape[0], queries),
            device=predictions["mask_logits"].device,
            dtype=predictions["mask_logits"].dtype,
        )
        if existence is None
        else query_gate_probability(
            existence[:, :queries],
            predictions.get("query_count_selection_mask"),
        )
    )
    return availability


def _owner_boundaries(
    owner: torch.Tensor,
    foreground: torch.Tensor,
) -> torch.Tensor:
    boundary = torch.zeros_like(foreground)
    left_right = (
        owner[:, :, 1:] != owner[:, :, :-1]
    ) & foreground[:, :, 1:] & foreground[:, :, :-1]
    top_bottom = (
        owner[:, 1:, :] != owner[:, :-1, :]
    ) & foreground[:, 1:, :] & foreground[:, :-1, :]
    boundary[:, :, 1:] |= left_right
    boundary[:, 1:, :] |= top_bottom
    return boundary


def room_owner_readout(
    predictions: dict[str, torch.Tensor],
    num_room_classes: int,
    *,
    confidence_threshold: float = 0.5,
    footprint: torch.Tensor | None = None,
    fallback_logits: torch.Tensor | None = None,
    soft_foreground: bool = False,
    confidence_mode: str = "absolute",
) -> dict[str, torch.Tensor]:
    """Expose query ownership before semantic class mass is collapsed."""

    if confidence_mode not in {"absolute", "relative"}:
        raise ValueError(
            "owner confidence mode must be absolute or relative"
        )
    query_mass = _effective_query_mass(predictions)
    _, owner = query_mass.max(dim=1)
    raw_mask_confidence = predictions["mask_logits"].sigmoid()
    owner_confidence = torch.gather(
        raw_mask_confidence,
        1,
        owner[:, None, :, :],
    )[:, 0]
    availability = _query_availability(
        predictions,
        raw_mask_confidence.shape[1],
    )
    owner_availability = torch.gather(
        availability[:, :, None, None].expand(
            -1,
            -1,
            owner.shape[-2],
            owner.shape[-1],
        ),
        1,
        owner[:, None, :, :],
    )[:, 0]
    owner_probability = query_mass / query_mass.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    owner_relative_confidence = torch.gather(
        owner_probability,
        1,
        owner[:, None, :, :],
    )[:, 0]
    owner_weight = (
        owner_confidence
        if confidence_mode == "absolute"
        else owner_relative_confidence
    )
    foreground = (
        owner_weight >= float(confidence_threshold)
    ) & (owner_availability >= 0.5)
    if footprint is not None:
        area = footprint[:, 0] if footprint.ndim == 4 else footprint
        if area.shape[-2:] != owner_confidence.shape[-2:]:
            area = F.interpolate(
                area[:, None].float(),
                size=owner_confidence.shape[-2:],
                mode="nearest",
            )[:, 0]
        foreground &= area.bool()
    boundaries = _owner_boundaries(owner, foreground)

    class_probability = predictions["class_logits"][
        ..., :num_room_classes
    ].softmax(dim=-1)
    owner_index = owner[:, None, :, :, None].expand(
        -1,
        1,
        -1,
        -1,
        num_room_classes,
    )
    owner_class = torch.gather(
        class_probability[:, :, None, None, :].expand(
            -1,
            -1,
            owner.shape[-2],
            owner.shape[-1],
            -1,
        ),
        1,
        owner_index,
    )[:, 0].permute(0, 3, 1, 2)
    fallback = (
        fused_room_mask_logits(predictions, num_room_classes)
        if fallback_logits is None
        else fallback_logits
    )
    if fallback.shape[-2:] != owner.shape[-2:]:
        fallback = F.interpolate(
            fallback.float(),
            size=owner.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    owner_logits = owner_class.clamp_min(1.0e-8).log()
    if soft_foreground:
        soft_weight = owner_weight * owner_availability
        owner_preserving = (
            soft_weight[:, None] * owner_logits
            + (1.0 - soft_weight[:, None]) * fallback
        )
    else:
        owner_preserving = torch.where(
            foreground[:, None],
            owner_logits,
            fallback,
        )
    return {
        "query_mass": query_mass,
        "owner_map": owner,
        "owner_confidence": owner_confidence,
        "owner_foreground": foreground,
        "owner_confidence_map": owner_confidence,
        "owner_relative_confidence": owner_relative_confidence,
        "owner_mass_sum": query_mass.sum(dim=1),
        "owner_availability": owner_availability,
        "owner_soft_weight": (
            owner_weight * owner_availability
        ),
        "owner_confidence_mode": confidence_mode,
        "owner_boundary": boundaries,
        "owner_preserving_logits": owner_preserving,
    }


__all__ = ["exclusive_query_owner_probability", "room_owner_readout"]
