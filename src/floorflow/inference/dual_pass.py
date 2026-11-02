"""Single-output draft/remask/refine inference for sparse CAD controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from floorflow.model.sparse_control_attention import (
    MaskedControlFlow,
    sample_sparse_guided_floorplans,
)


@dataclass(frozen=True)
class DualPassResult:
    final: torch.Tensor
    draft: torch.Tensor
    remask: torch.Tensor
    diagnostics: dict[str, float]


@torch.no_grad()
def confidence_map(
    prediction: torch.Tensor,
    footprint: torch.Tensor,
) -> torch.Tensor:
    """Return one minus normalized Bernoulli entropy for each location."""

    probability = prediction.float().sigmoid().clamp(1.0e-4, 1.0 - 1.0e-4)
    entropy = -(
        probability * probability.log()
        + (1.0 - probability) * (1.0 - probability).log()
    )
    confidence = 1.0 - entropy.mean(dim=1, keepdim=True)
    return confidence * footprint.float()


@torch.no_grad()
def select_remask(
    confidence: torch.Tensor,
    footprint: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    flat_confidence = confidence[:, 0].flatten(1)
    flat_region = footprint[:, 0].bool().flatten(1)
    remask = torch.zeros_like(flat_region)
    for index in range(confidence.shape[0]):
        count = min(
            int(flat_region[index].sum()),
            max(0, int(round(int(flat_region[index].sum()) * float(ratio)))),
        )
        if count <= 0:
            continue
        scores = flat_confidence[index].masked_fill(
            ~flat_region[index],
            float("inf"),
        )
        selected = scores.topk(count, largest=False).indices
        remask[index, selected] = True
    return remask.reshape_as(footprint)


@torch.no_grad()
def sample_dual_pass(
    model: MaskedControlFlow,
    footprint: torch.Tensor,
    initial_control: torch.Tensor,
    initial_mask: torch.Tensor,
    missing_token: torch.Tensor,
    *,
    steps_per_pass: int = 4,
    remask_ratio: float = 0.5,
    floor_ids: torch.Tensor | None = None,
) -> DualPassResult:
    """Generate one draft, remask its least confident pixels, then refine."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    if initial_mask.ndim == 3:
        initial_mask = initial_mask[:, None]
    initial_mask = initial_mask.bool().expand_as(initial_control) & footprint.bool()
    draft = sample_sparse_guided_floorplans(
        model,
        footprint,
        initial_control,
        initial_mask,
        steps=steps_per_pass,
        floor_ids=floor_ids,
    )
    confidence = confidence_map(draft, footprint)
    remask = select_remask(confidence, footprint, remask_ratio)
    token = missing_token.to(draft.dtype)[None, :, None, None]
    masked_draft = torch.where(remask.expand_as(draft), token, draft)
    final = sample_sparse_guided_floorplans(
        model,
        footprint,
        masked_draft,
        remask.expand_as(draft),
        steps=steps_per_pass,
        floor_ids=floor_ids,
        initial_state=masked_draft,
    )
    inside = footprint.bool().expand_as(remask)
    diagnostics = {
        "draft_confidence_mean": float(
            confidence[inside].mean().cpu()
        ),
        "remask_fraction": float(
            remask[inside].float().mean().cpu()
        ),
        "draft_final_l1": float(
            (
                (draft.float() - final.float()).abs()
                * footprint.float()
            ).sum().cpu()
        ),
        "passes": 2.0,
        "candidate_count": 1.0,
    }
    return DualPassResult(final, draft, remask, diagnostics)


__all__ = [
    "DualPassResult",
    "confidence_map",
    "sample_dual_pass",
    "select_remask",
]
