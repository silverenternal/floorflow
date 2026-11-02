"""Ranking and calibration losses for constraint-aware action gating."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.model.constraint_aware_gate import hard_masked_log_softmax


def hard_masked_listwise_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    hard_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy over feasible candidates with multi-positive targets."""

    positive = positive_mask.bool() & hard_mask.bool()
    valid_groups = positive.any(dim=-1)
    if not bool(valid_groups.any()):
        return logits.sum() * 0.0
    log_probability = hard_masked_log_softmax(logits, hard_mask, dim=-1)
    per_group = -log_probability.masked_fill(~positive, 0.0).sum(dim=-1)
    return per_group[valid_groups].mean()


def hard_masked_pairwise_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    hard_mask: torch.Tensor,
    *,
    margin: float = 0.2,
) -> torch.Tensor:
    """Margin ranking between feasible positives and feasible negatives."""

    positive = positive_mask.bool() & hard_mask.bool()
    negative = (~positive_mask.bool()) & hard_mask.bool()
    valid = positive.any(dim=-1) & negative.any(dim=-1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    pair_mask = (
        positive[:, :, None]
        & negative[:, None, :]
        & valid[:, None, None]
    )
    differences = (
        logits[:, :, None] - logits[:, None, :] - margin
    )
    pairwise = torch.nn.functional.softplus(-differences)
    return pairwise.masked_select(pair_mask).mean()


def gate_calibration_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    hard_mask: torch.Tensor,
) -> torch.Tensor:
    selected = hard_mask.bool()
    if not bool(selected.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[selected],
        target.float()[selected],
    )


__all__ = [
    "gate_calibration_loss",
    "hard_masked_listwise_loss",
    "hard_masked_pairwise_loss",
]
