"""Joint shape, pin-tail and MIB consistency terms for chip endpoints."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ShapePinConsistencyTerms:
    total: torch.Tensor
    mib_shared: torch.Tensor
    pin_tail: torch.Tensor
    rho_trust: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "endpoint_shape_pin": self.total,
            "endpoint_mib_shared": self.mib_shared,
            "endpoint_pin_tail": self.pin_tail,
            "endpoint_rho_trust": self.rho_trust,
        }


def _mib_shared_shape(boxes: torch.Tensor, mib_ids: torch.Tensor) -> torch.Tensor:
    log_width = boxes[..., 2].clamp_min(1.0e-6).log()
    log_height = boxes[..., 3].clamp_min(1.0e-6).log()
    losses = []
    for group in torch.unique(mib_ids[mib_ids > 0]):
        members = mib_ids == group
        count = members.sum(dim=1).clamp_min(1)
        mean_width = (
            log_width.masked_fill(~members, 0.0).sum(dim=1)
            / count
        )
        mean_height = (
            log_height.masked_fill(~members, 0.0).sum(dim=1)
            / count
        )
        error = (
            (log_width - mean_width[:, None]).square()
            + (log_height - mean_height[:, None]).square()
        )
        losses.append(
            torch.where(
                count > 1,
                (error * members).sum(1) / count.clamp_min(1),
                0.0,
            )
        )
    if not losses:
        return boxes.new_zeros(())
    return torch.stack(losses, dim=1).mean()


def _pin_tail(
    boxes: torch.Tensor,
    areas: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
    n_p2b: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("pin tail fraction must be in (0, 1]")
    if p2b.shape[1] == 0:
        return boxes.new_zeros(())
    centers = boxes[..., :2] + boxes[..., 2:] / 2.0
    valid = (
        torch.arange(p2b.shape[1], device=p2b.device)[None]
        < n_p2b.to(p2b.device).long()[:, None]
    )
    pin_id = p2b[..., 0].long().clamp(0, pins.shape[1] - 1)
    block_id = p2b[..., 1].long().clamp(0, boxes.shape[1] - 1)
    pin_xy = pins.gather(1, pin_id[..., None].expand(-1, -1, 2))
    block_xy = centers.gather(1, block_id[..., None].expand(-1, -1, 2))
    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    distance = (pin_xy.float() - block_xy.float()).abs().sum(-1) / scale[:, None]
    distance = distance * p2b[..., 2].float().clamp_min(0.0) * valid
    maximum_count = max(1, math.ceil(distance.shape[1] * float(fraction)))
    top_indices = distance.masked_fill(~valid, float("-inf")).topk(
        maximum_count,
        dim=1,
    ).indices
    top_counts = torch.ceil(
        n_p2b.float() * float(fraction)
    ).long().clamp(min=1, max=distance.shape[1])
    top_rank = torch.full_like(distance, distance.shape[1], dtype=torch.long)
    rank_values = torch.arange(maximum_count, device=distance.device)[None].expand_as(
        top_indices
    )
    top_rank.scatter_(1, top_indices, rank_values)
    selected = valid & top_rank.lt(top_counts[:, None])
    denominator = selected.sum(dim=1).clamp_min(1).to(distance.dtype)
    return (distance * selected).sum(dim=1).div(denominator).mean()


def shape_pin_consistency_terms(
    projected_boxes: torch.Tensor,
    reference_boxes: torch.Tensor,
    areas: torch.Tensor,
    constraints: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
    n_p2b: torch.Tensor,
    *,
    mib_weight: float = 0.1,
    pin_tail_weight: float = 0.1,
    rho_trust_weight: float = 0.1,
    rho_trust_radius: float = 0.1,
    pin_tail_fraction: float = 0.2,
) -> ShapePinConsistencyTerms:
    """Return endpoint terms coupled across shape, pins and MIB groups."""

    weights = (mib_weight, pin_tail_weight, rho_trust_weight)
    if any(value < 0.0 for value in weights) or rho_trust_radius < 0.0:
        raise ValueError("shape-pin weights and trust radius must be non-negative")
    projected_ratio = torch.log(
        projected_boxes[..., 2].clamp_min(1.0e-6)
        / projected_boxes[..., 3].clamp_min(1.0e-6)
    )
    reference_ratio = torch.log(
        reference_boxes[..., 2].clamp_min(1.0e-6)
        / reference_boxes[..., 3].clamp_min(1.0e-6)
    )
    mib_shared = _mib_shared_shape(
        projected_boxes,
        constraints[..., 2].long(),
    )
    pin_tail = _pin_tail(
        projected_boxes,
        areas,
        p2b,
        pins,
        n_p2b,
        pin_tail_fraction,
    )
    rho_trust = (
        (projected_ratio - reference_ratio).abs() - float(rho_trust_radius)
    ).clamp_min(0.0).mean()
    total = (
        float(mib_weight) * mib_shared
        + float(pin_tail_weight) * pin_tail
        + float(rho_trust_weight) * rho_trust
    )
    return ShapePinConsistencyTerms(
        total=total,
        mib_shared=mib_shared,
        pin_tail=pin_tail,
        rho_trust=rho_trust,
    )


__all__ = [
    "ShapePinConsistencyTerms",
    "shape_pin_consistency_terms",
]
