"""Iterative sparse-control sampling inspired by MaskPLAN inference."""

from __future__ import annotations

import torch

from floorflow.model.sparse_control_attention import (
    MaskedControlFlow,
    sample_sparse_guided_floorplans,
)


@torch.no_grad()
def sample_iterative_controlled(
    model: MaskedControlFlow,
    footprint: torch.Tensor,
    initial_control: torch.Tensor,
    initial_mask: torch.Tensor,
    *,
    rounds: int = 3,
    steps_per_round: int = 4,
    reveal_fraction: float = 0.25,
    floor_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    control = initial_control.to(footprint.dtype) * footprint.float()
    control_mask = initial_mask.bool()
    if control_mask.ndim == 3:
        control_mask = control_mask[:, None]
    control_mask = control_mask.expand_as(control) & footprint.bool()
    sampled = control
    for _round in range(max(1, int(rounds))):
        sampled = sample_sparse_guided_floorplans(
            model,
            footprint,
            control,
            control_mask,
            steps=steps_per_round,
            floor_ids=floor_ids,
        )
        masked_region = control_mask.amax(dim=1) & footprint.bool()[:, 0]
        scores = sampled.clamp(0.0, 1.0).amax(dim=1)
        flat_scores = scores.flatten(1)
        flat_region = masked_region.flatten(1)
        new_support = torch.zeros_like(masked_region)
        for index in range(scores.shape[0]):
            valid = flat_region[index]
            count = int(valid.sum() * float(reveal_fraction))
            if count <= 0:
                continue
            eligible = torch.where(
                valid,
                flat_scores[index],
                torch.full_like(flat_scores[index], -1.0),
            )
            threshold = eligible.topk(count).values[-1]
            selected = valid & (flat_scores[index] >= threshold)
            new_support[index] = selected.reshape_as(new_support[index])
        reveal = new_support[:, None].expand_as(control) & control_mask
        control = torch.where(reveal, sampled, control)
        control_mask = control_mask & ~reveal
    return sampled, control, control_mask


__all__ = ["sample_iterative_controlled"]
