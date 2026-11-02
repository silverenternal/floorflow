from __future__ import annotations

import torch

from floorflow.model.dynamic_budget import (
    allocate_group_token_budgets,
    family_minimum_budgets,
    mean_group_priority,
)


def test_dynamic_budgets_are_exact_and_respect_bounds() -> None:
    priorities = torch.tensor([[0.9, 0.1, 0.2, 0.4], [0.1, 0.8, 0.3, 0.2]])
    capacities = torch.tensor([[100, 20, 30, 50], [100, 20, 30, 50]])
    lower = torch.tensor([[10, 0, 5, 5], [10, 5, 0, 5]])
    upper = torch.tensor([[90, 20, 30, 50], [90, 20, 30, 50]])
    target = torch.tensor([100, 80])
    budgets = allocate_group_token_budgets(
        priorities,
        capacities,
        target,
        minimum_counts=lower,
        maximum_counts=upper,
    )
    assert torch.equal(budgets.sum(dim=1), target)
    assert torch.all(budgets >= lower)
    assert torch.all(budgets <= upper)
    assert budgets[0, 0] > budgets[0, 1]


def test_family_priorities_use_only_footprint() -> None:
    probability = torch.ones(1, 4, 8, 8)
    footprint = torch.zeros(1, 1, 8, 8)
    footprint[:, :, 2:6, 2:6] = 1.0
    groups = (
        torch.tensor([0]),
        torch.tensor([1, 2]),
        torch.tensor([3]),
        torch.tensor([0, 3]),
    )
    priority, capacity = mean_group_priority(probability, footprint, groups)
    assert torch.allclose(priority, torch.ones_like(priority))
    assert capacity.tolist() == [[16, 32, 16, 32]]


def test_visibility_floor_is_weakened_for_tiny_ratios() -> None:
    capacity = torch.tensor([[100, 100]])
    visible = torch.tensor([[0.5, 0.2]])
    low = family_minimum_budgets(capacity, visible, 0.01)
    high = family_minimum_budgets(capacity, visible, 0.25)
    assert torch.all(low <= high)
    assert high.tolist() == [[50, 80]]
