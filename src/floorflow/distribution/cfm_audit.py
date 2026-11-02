"""Representation and cache audits for structured distribution CFM."""

from __future__ import annotations

from collections.abc import Mapping

import torch


@torch.no_grad()
def outside_support_change(
    before: torch.Tensor,
    after: torch.Tensor,
    footprint: torch.Tensor,
) -> float:
    """Return mean absolute change strictly outside the support mask."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    outside = ~footprint.bool().expand_as(before)
    if not outside.any():
        return 0.0
    return float((after - before).abs()[outside].mean().cpu())


@torch.no_grad()
def family_stream_distinctness(
    streams: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Measure normalized separation among family token streams."""

    names = [name for name in ("room", "wall", "opening", "furniture") if name in streams]
    output = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = streams[left_name].float()
            right = streams[right_name].float()
            scale = 0.5 * (left.square().mean().sqrt() + right.square().mean().sqrt())
            output[f"{left_name}_{right_name}"] = float(
                ((left - right).square().mean().sqrt() / scale.clamp_min(1.0e-6))
                .cpu()
            )
    return output


@torch.no_grad()
def condition_cache_reuse_ratio(steps: int) -> float:
    """Return encoder evaluations saved relative to one call per NFE."""

    steps = max(0, int(steps))
    if steps == 0:
        return 0.0
    return float((steps - 1) / steps)


@torch.no_grad()
def group_coupling_residual(
    pixel_probability: torch.Tensor,
    element_probability: torch.Tensor,
    groups: Mapping[str, tuple[int, ...]],
) -> dict[str, float]:
    """Report normalized L1 residual for each parent/child family."""

    output = {}
    for name in ("wall", "opening", "room", "furniture", "support"):
        indices = tuple(int(value) for value in groups.get(name, ()))
        if not indices:
            continue
        pixel = pixel_probability[:, indices].float()
        pixel = pixel / pixel.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        element = element_probability[:, indices].float()
        element = element / element.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        output[name] = float((pixel - element).abs().mean().cpu())
    return output


__all__ = [
    "condition_cache_reuse_ratio",
    "family_stream_distinctness",
    "group_coupling_residual",
    "outside_support_change",
]
