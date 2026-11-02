"""Hard-masked utility ranking objectives for model-origin action policies."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.model.constraint_aware_gate import hard_masked_log_softmax


def utility_target_distribution(
    utility: torch.Tensor,
    hard_mask: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    feasible = hard_mask.bool()
    shifted = utility / max(float(temperature), 1.0e-6)
    shifted = shifted.masked_fill(~feasible, -torch.inf)
    target = shifted.softmax(dim=-1)
    target = target.masked_fill(~feasible, 0.0)
    return target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)


def hard_masked_utility_loss(
    logits: torch.Tensor,
    utility: torch.Tensor,
    hard_mask: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    feasible = hard_mask.bool()
    valid = feasible.any(dim=-1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    target = utility_target_distribution(
        utility,
        feasible,
        temperature=temperature,
    )
    log_probability = hard_masked_log_softmax(logits, feasible, dim=-1)
    loss = -(
        target * log_probability.masked_fill(~feasible, 0.0)
    ).sum(dim=-1)
    return loss[valid].mean()


def hard_masked_utility_pairwise_loss(
    logits: torch.Tensor,
    utility: torch.Tensor,
    hard_mask: torch.Tensor,
    *,
    margin: float = 0.01,
) -> torch.Tensor:
    feasible = hard_mask.bool()
    pair = (
        feasible[:, :, None]
        & feasible[:, None, :]
        & (utility[:, :, None] > utility[:, None, :] + 1.0e-8)
    )
    if not bool(pair.any()):
        return logits.sum() * 0.0
    differences = logits[:, :, None] - logits[:, None, :] - margin
    return F.softplus(-differences).masked_select(pair).mean()


def hard_masked_best_action_loss(
    logits: torch.Tensor,
    utility: torch.Tensor,
    hard_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy toward all legal actions with maximal utility."""

    feasible = hard_mask.bool()
    valid = feasible.any(dim=-1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    masked_utility = utility.masked_fill(~feasible, -torch.inf)
    best = masked_utility == masked_utility.max(dim=-1, keepdim=True).values
    target = best.float() / best.float().sum(dim=-1, keepdim=True)
    log_probability = hard_masked_log_softmax(logits, feasible, dim=-1)
    loss = -(
        target * log_probability.masked_fill(~feasible, 0.0)
    ).sum(dim=-1)
    return loss[valid].mean()


__all__ = [
    "hard_masked_best_action_loss",
    "hard_masked_utility_loss",
    "hard_masked_utility_pairwise_loss",
    "utility_target_distribution",
]
