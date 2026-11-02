from __future__ import annotations

import torch

from floorflow.recursion.budget import allocate_dynamic_budgets


def test_dynamic_budgets_increase_with_complexity() -> None:
    strength = torch.tensor([0.0, 0.5, 1.0])
    budget = allocate_dynamic_budgets(strength)
    assert budget.query_budget[0] < budget.query_budget[2]
    assert budget.expert_budget[0] <= budget.expert_budget[2]
    assert budget.graph_steps[0] == 0
    assert budget.graph_steps[2] == 3
    assert budget.recursion_depth[-1] == 3
