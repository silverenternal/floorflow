"""Exact-total dynamic token budgets adapted from DHSA."""

from __future__ import annotations

import torch


def allocate_group_token_budgets(
    priorities: torch.Tensor,
    capacities: torch.Tensor,
    target_totals: torch.Tensor,
    *,
    minimum_counts: torch.Tensor | None = None,
    maximum_counts: torch.Tensor | None = None,
    redistribution_steps: int = 8,
) -> torch.Tensor:
    """Allocate integer family budgets while preserving each total exactly.

    ``priorities``, ``capacities``, and the optional bounds are ``[B,G]``.
    The source idea is DHSA's ``allocate_dynamic_row_budgets``; this version
    redistributes semantic-family masks rather than attention rows.
    """

    if priorities.ndim != 2:
        raise ValueError("priorities must be [B,G]")
    if capacities.shape != priorities.shape:
        raise ValueError("capacities must match priorities")
    if target_totals.ndim != 1 or target_totals.shape[0] != priorities.shape[0]:
        raise ValueError("target_totals must be [B]")
    if redistribution_steps <= 0:
        raise ValueError("redistribution_steps must be positive")
    device = priorities.device
    capacities = capacities.long().clamp_min(0)
    lower = torch.zeros_like(capacities)
    if minimum_counts is not None:
        lower = torch.minimum(
            capacities,
            torch.maximum(
                lower,
                torch.broadcast_to(minimum_counts.long(), capacities.shape),
            ),
        )
    upper = capacities
    if maximum_counts is not None:
        upper = torch.maximum(
            lower,
            torch.minimum(
                capacities,
                torch.broadcast_to(maximum_counts.long(), capacities.shape),
            ),
        )
    target = torch.minimum(
        upper.sum(dim=1),
        torch.maximum(lower.sum(dim=1), target_totals.long()),
    )
    continuous = lower.float()
    capacity = (upper - lower).float()
    remaining = (target - lower.sum(dim=1)).float()
    for _ in range(int(redistribution_steps)):
        active = capacity > 0
        score = priorities.float().clamp_min(0.0).masked_fill(~active, 0.0)
        score_sum = score.sum(dim=1, keepdim=True)
        uniform = active.float() / active.sum(dim=1, keepdim=True).clamp_min(1)
        weights = torch.where(
            score_sum > 0,
            score / score_sum.clamp_min(1.0e-8),
            uniform,
        )
        proposal = remaining[:, None] * weights
        addition = torch.minimum(proposal, capacity)
        continuous = continuous + addition
        capacity = capacity - addition
        remaining = (remaining - addition.sum(dim=1)).clamp_min(0.0)
    budgets = continuous.floor().long()
    budgets = torch.minimum(budgets, upper)
    remainder = target - budgets.sum(dim=1)
    fractions = (continuous - budgets.float()).masked_fill(
        budgets >= upper,
        -1.0,
    )
    for index in range(priorities.shape[0]):
        count = int(remainder[index])
        if count <= 0:
            continue
        selected = fractions[index].topk(
            min(count, priorities.shape[1])
        ).indices
        budgets[index, selected] += 1
    if not torch.equal(budgets.sum(dim=1), target):
        raise RuntimeError("integer rounding did not preserve dynamic budgets")
    return budgets


def family_minimum_budgets(
    capacities: torch.Tensor,
    minimum_visible: torch.Tensor,
    global_ratio: float,
    *,
    warmup_ratio: float = 0.10,
) -> torch.Tensor:
    """Scale family visibility floors down for very low global mask ratios."""

    if capacities.ndim != 2 or minimum_visible.shape != capacities.shape:
        raise ValueError("capacities and minimum_visible must be [B,G]")
    if not 0.0 <= global_ratio <= 1.0:
        raise ValueError("global_ratio must be in [0,1]")
    strength = min(1.0, max(0.0, float(global_ratio) / warmup_ratio))
    required_masked = capacities.float() * (
        1.0 - minimum_visible.clamp(0.0, 1.0)
    ) * strength
    return required_masked.ceil().long()


def mean_group_priority(
    probability: torch.Tensor,
    footprint: torch.Tensor,
    channel_groups: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,G] priority/capacity summaries over semantic families."""

    if probability.ndim != 4 or footprint.ndim != 4:
        raise ValueError("probability and footprint must be BCHW")
    priorities = []
    capacities = []
    valid = footprint.bool().expand_as(probability)
    for channels in channel_groups:
        channels = channels.to(device=probability.device, dtype=torch.long)
        selected = probability.index_select(1, channels)
        selected_valid = valid.index_select(1, channels)
        count = selected_valid.sum(dim=(1, 2, 3)).float()
        total = (selected * selected_valid).sum(dim=(1, 2, 3))
        priorities.append(total / count.clamp_min(1.0))
        capacities.append(count.long())
    return torch.stack(priorities, dim=1), torch.stack(capacities, dim=1)


__all__ = [
    "allocate_group_token_budgets",
    "family_minimum_budgets",
    "mean_group_priority",
]
