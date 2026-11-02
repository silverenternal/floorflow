"""Differentiable FEVP-inspired training proxies, never audit metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _foreground(probability: torch.Tensor) -> torch.Tensor:
    if probability.ndim == 4:
        return probability.sum(dim=1).clamp(0.0, 1.0)
    return probability.clamp(0.0, 1.0)


def soft_room_overlap(
    room_probability: torch.Tensor,
    footprint: torch.Tensor,
) -> torch.Tensor:
    if room_probability.ndim != 4:
        raise ValueError("room_probability must be [B,R,H,W]")
    valid = footprint.float()
    overlap = (room_probability.sum(dim=1) - 1.0).clamp_min(0.0)
    return (overlap * valid).sum() / valid.sum().clamp_min(1.0)


def soft_outside_footprint(
    room_probability: torch.Tensor,
    footprint: torch.Tensor,
) -> torch.Tensor:
    foreground = _foreground(room_probability)
    outside = 1.0 - footprint.float()
    return (foreground * outside).sum() / outside.sum().clamp_min(1.0)


def soft_wall_continuity(
    room_probability: torch.Tensor,
    wall_probability: torch.Tensor,
    footprint: torch.Tensor,
) -> torch.Tensor:
    probability = room_probability.float()
    if probability.ndim == 3:
        probability = probability[:, None]
    seam_x = (
        probability[:, :, :, 1:] - probability[:, :, :, :-1]
    ).abs().sum(dim=1)
    seam_y = (
        probability[:, :, 1:, :] - probability[:, :, :-1, :]
    ).abs().sum(dim=1)
    seam = F.pad(seam_x, (0, 1)) + F.pad(seam_y, (0, 0, 0, 1))
    seam = (seam > 0.05).float()
    near = F.max_pool2d(
        seam[:, None],
        5,
        stride=1,
        padding=2,
    )[:, 0]
    missing = 1.0 - wall_probability.float()
    valid = footprint.float()
    weighted = near * missing * valid
    return weighted.sum() / (near * valid).sum().clamp_min(1.0)


def soft_opening_attachment(
    opening_probability: torch.Tensor,
    wall_probability: torch.Tensor,
    footprint: torch.Tensor,
    *,
    radius: int = 3,
) -> torch.Tensor:
    near_wall = F.max_pool2d(
        wall_probability.float()[:, None],
        2 * int(radius) + 1,
        stride=1,
        padding=int(radius),
    )[:, 0]
    detached = opening_probability.float() * (1.0 - near_wall)
    valid = footprint.float()
    return (detached * valid).sum() / (
        opening_probability.float().sum() + valid.sum() * 1.0e-6
    )


def soft_outside_preservation(
    after_probability: torch.Tensor,
    before_probability: torch.Tensor,
    footprint: torch.Tensor,
    edit_region: torch.Tensor,
) -> torch.Tensor:
    after = _foreground(after_probability)
    before = _foreground(before_probability)
    outside = footprint.float() * (1.0 - edit_region.float())
    return ((after - before).abs() * outside).sum() / outside.sum().clamp_min(1.0)


def soft_boundary_agreement(
    predicted_probability: torch.Tensor,
    target_probability: torch.Tensor,
) -> torch.Tensor:
    predicted = predicted_probability.float()
    target = target_probability.float()
    if predicted.ndim == 3:
        predicted = predicted[:, None]
    if target.ndim == 3:
        target = target[:, None]
    predicted_boundary = (
        (predicted[:, :, :, 1:] - predicted[:, :, :, :-1]).abs().sum(dim=1)
        + (predicted[:, :, 1:, :] - predicted[:, :, :-1, :]).abs().sum(dim=1)
    )
    target_boundary = (
        (target[:, :, :, 1:] - target[:, :, :, :-1]).abs().sum(dim=1)
        + (target[:, :, 1:, :] - target[:, :, :-1, :]).abs().sum(dim=1)
    )
    return F.binary_cross_entropy(
        predicted_boundary.clamp(1.0e-5, 1.0 - 1.0e-5),
        target_boundary.clamp(0.0, 1.0),
    )


def differentiable_fevp_terms(
    after_room_probability: torch.Tensor,
    before_room_probability: torch.Tensor,
    wall_probability: torch.Tensor,
    opening_probability: torch.Tensor,
    footprint: torch.Tensor,
    edit_region: torch.Tensor,
    *,
    target_room_probability: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return training-only relaxations with explicitly soft names."""

    size = tuple(after_room_probability.shape[-2:])

    def resize(value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape[-2:]) == size:
            return value
        if value.ndim == 3:
            return F.interpolate(
                value[:, None],
                size=size,
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        return F.interpolate(
            value,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    wall_probability = resize(wall_probability)
    opening_probability = resize(opening_probability)
    footprint = resize(footprint)
    edit_region = resize(edit_region)
    if target_room_probability is not None:
        target_room_probability = resize(target_room_probability)
    terms = {
        "soft_room_overlap": soft_room_overlap(
            after_room_probability,
            footprint,
        ),
        "soft_outside_footprint": soft_outside_footprint(
            after_room_probability,
            footprint,
        ),
        "soft_wall_continuity": soft_wall_continuity(
            after_room_probability,
            wall_probability,
            footprint,
        ),
        "soft_opening_attachment": soft_opening_attachment(
            opening_probability,
            wall_probability,
            footprint,
        ),
        "soft_outside_preservation": soft_outside_preservation(
            after_room_probability,
            before_room_probability,
            footprint,
            edit_region,
        ),
    }
    if target_room_probability is not None:
        terms["soft_boundary_agreement"] = soft_boundary_agreement(
            after_room_probability,
            target_room_probability,
        )
    total = (
        terms["soft_room_overlap"]
        + terms["soft_outside_footprint"]
        + 0.25 * terms["soft_wall_continuity"]
        + 0.25 * terms["soft_opening_attachment"]
        + terms["soft_outside_preservation"]
    )
    if "soft_boundary_agreement" in terms:
        total = total + 0.25 * terms["soft_boundary_agreement"]
    return total, {key: value.detach() for key, value in terms.items()}


__all__ = [
    "differentiable_fevp_terms",
    "soft_boundary_agreement",
    "soft_opening_attachment",
    "soft_outside_footprint",
    "soft_outside_preservation",
    "soft_room_overlap",
    "soft_wall_continuity",
]
