"""Utilities for validation-only, budget-matched candidate subset selection."""

from __future__ import annotations

from itertools import combinations
from typing import Any


def budget_matched_subsets(
    candidate_count: int,
    *,
    budget: int = 4,
    required: tuple[int, ...] = (0,),
) -> list[tuple[int, ...]]:
    """Return all fixed-size candidate sets containing required indices."""

    if candidate_count <= 0 or budget <= 0 or budget > candidate_count:
        raise ValueError("invalid candidate count or budget")
    required = tuple(sorted({int(index) for index in required}))
    if len(required) > budget:
        raise ValueError("required candidate count exceeds budget")
    if required and (required[0] < 0 or required[-1] >= candidate_count):
        raise ValueError("required candidate index is out of range")
    remaining = [
        index
        for index in range(candidate_count)
        if index not in required
    ]
    fill = budget - len(required)
    return [
        tuple(sorted((*required, *extra)))
        for extra in combinations(remaining, fill)
    ]


def validation_rank(result: dict[str, Any]) -> tuple[float, ...]:
    """Rank a subset using validation signal only, higher is better."""

    anchor = result["validation"]["versus_anchor"]
    wins = anchor["wins"]
    relative = anchor["relative"]
    gate_count = int(wins["constraint_safe_rate"] >= 0.35)
    gate_count += int(wins["strict_structural_rate"] >= 0.35)
    gate_count += int(wins["joint_rate"] >= 0.65)
    return (
        float(gate_count),
        float(wins["constraint_safe_rate"]),
        float(wins["strict_structural_rate"]),
        float(wins["joint_rate"]),
        -float(relative["legalized_hard_cost"]),
        -float(relative["legalized_v_rel"]),
        -float(relative["legalized_grouping"]),
        -float(relative["legalized_rho_mae"]),
    )


__all__ = ["budget_matched_subsets", "validation_rank"]
