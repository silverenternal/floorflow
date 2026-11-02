"""Complexity-conditioned compute budgets for recursive decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DynamicBudgets:
    query_budget: torch.Tensor
    expert_budget: torch.Tensor
    graph_steps: torch.Tensor
    recursion_depth: torch.Tensor

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "query_budget": self.query_budget.tolist(),
            "expert_budget": self.expert_budget.tolist(),
            "graph_steps": self.graph_steps.tolist(),
            "recursion_depth": self.recursion_depth.tolist(),
        }


def allocate_dynamic_budgets(
    recursion_strength: torch.Tensor,
    *,
    min_queries: int = 4,
    max_queries: int = 32,
    min_experts: int = 1,
    max_experts: int = 8,
    max_graph_steps: int = 3,
    max_depth: int = 3,
) -> DynamicBudgets:
    """Map continuous predicted complexity to reproducible compute budgets."""

    strength = recursion_strength.detach().float().clamp(0.0, 1.0)
    query_budget = torch.round(
        min_queries + strength * (max_queries - min_queries)
    ).long()
    expert_budget = torch.round(
        min_experts + strength * (max_experts - min_experts)
    ).long()
    graph_steps = torch.round(strength * max_graph_steps).long()
    depth = torch.round(strength * max_depth).long()
    return DynamicBudgets(
        query_budget=query_budget,
        expert_budget=expert_budget,
        graph_steps=graph_steps,
        recursion_depth=depth,
    )


__all__ = ["DynamicBudgets", "allocate_dynamic_budgets"]
