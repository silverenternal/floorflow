"""Leak-free construction of sparse controls for guided CFM training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.flow.condition_view import (
    ConditionView,
    contour_only_view,
    layout_conditioned_view,
    sparse_control_view,
)
from floorflow.flow.mask_policies import (
    AdversarialMaskScheduler,
    AttributeMaskPolicy,
    MaskPolicy,
)
from floorflow.flow.masked_control import apply_control_mask, control_channel_groups
from floorflow.model.learned_mask_generator import LearnedMaskGenerator


@dataclass(frozen=True)
class PreparedCondition:
    view: ConditionView
    control: torch.Tensor
    missing_mask: torch.Tensor
    coverage: torch.Tensor
    proposal_probability: torch.Tensor | None = None
    soft_mask: torch.Tensor | None = None
    soft_control: torch.Tensor | None = None
    policy_name: str | None = None


def prepare_masked_condition(
    source: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    missing_token: torch.Tensor,
    *,
    missing_ratio: float,
    floor_ids: torch.Tensor | None = None,
    mask_generator: LearnedMaskGenerator | None = None,
    mask_policy: MaskPolicy | None = None,
    scheduler: AdversarialMaskScheduler | None = None,
) -> PreparedCondition:
    """Build a model input without exposing target values behind a mask."""

    ratio = min(1.0, max(0.0, float(missing_ratio)))
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    if ratio <= 0.0:
        view = layout_conditioned_view(
            source,
            footprint,
            floor_ids=floor_ids,
        )
        missing_mask = torch.zeros_like(view.hidden_mask)
        return PreparedCondition(
            view=view,
            control=view.control,
            missing_mask=missing_mask,
            coverage=torch.zeros(source.shape[0], device=source.device),
        )
    if ratio >= 1.0:
        view = contour_only_view(
            footprint,
            source.shape[1],
            floor_ids=floor_ids,
        )
        missing_mask = view.hidden_mask
        control = apply_control_mask(view.control, missing_mask, missing_token)
        return PreparedCondition(
            view=view,
            control=control,
            missing_mask=missing_mask,
            coverage=torch.ones(source.shape[0], device=source.device),
        )

    proposal_probability = None
    soft_mask = None
    soft_control = None
    policy_name = None
    if mask_generator is not None:
        seed_policy = AttributeMaskPolicy()
        policy_name = None
        seed = seed_policy.sample(
            source,
            footprint,
            taxonomy,
            ratio=min(0.75, ratio * 0.5),
        )
        view = sparse_control_view(
            source,
            footprint,
            seed.value,
            floor_ids=floor_ids,
        )
        groups = control_channel_groups(taxonomy)
        generated = mask_generator(
            view.control,
            view.footprint,
            target_ratio=ratio,
            known_mask=seed.value,
            visibility=view.visibility,
            channel_groups=tuple(
                groups[name].to(source.device)
                for name in ("room", "wall", "opening", "furniture")
            ),
        )
        missing_mask = generated.hard
        proposal_probability = generated.probability
        soft_mask = torch.maximum(proposal_probability, seed.value.float())
        soft_control = (
            view.control * (1.0 - soft_mask)
            + missing_token.to(soft_mask.dtype)[None, :, None, None] * soft_mask
        )
    else:
        selected = scheduler.sample() if scheduler is not None else mask_policy
        if selected is None:
            selected = AttributeMaskPolicy()
        sampled = selected.sample(
            source,
            footprint,
            taxonomy,
            ratio=ratio,
        )
        view = sparse_control_view(
            source,
            footprint,
            sampled.value,
            floor_ids=floor_ids,
        )
        missing_mask = sampled.value
        policy_name = selected.name
    control = apply_control_mask(view.control, missing_mask, missing_token)
    inside = view.footprint.bool().expand_as(missing_mask)
    covered = (missing_mask.bool() & inside).sum(dim=(1, 2, 3)).float()
    coverage = covered / inside.sum(dim=(1, 2, 3)).clamp_min(1).float()
    return PreparedCondition(
        view=view,
        control=control,
        missing_mask=missing_mask,
        coverage=coverage,
        proposal_probability=proposal_probability,
        soft_mask=soft_mask,
        soft_control=soft_control,
        policy_name=policy_name,
    )


__all__ = ["PreparedCondition", "prepare_masked_condition"]
