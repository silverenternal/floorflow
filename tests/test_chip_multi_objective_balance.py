from __future__ import annotations

import pytest
import torch

from floorflow.chip_flow.multi_objective_balance import normalized_loss_balance


def test_normalized_loss_balance_equalizes_objective_scales() -> None:
    left = torch.tensor(1.0, requires_grad=True)
    right = torch.tensor(1000.0, requires_grad=True)
    total, weights = normalized_loss_balance(
        {"left": left, "right": right},
        floor=1.0e-6,
    )
    assert weights["loss_balance_left"] > weights["loss_balance_right"]
    total.backward()
    assert left.grad is not None and right.grad is not None
    contribution_ratio = float(
        (left.detach() * weights["loss_balance_left"].detach())
        / (right.detach() * weights["loss_balance_right"].detach())
    )
    assert contribution_ratio == pytest.approx(1.0)


def test_normalized_loss_balance_rejects_empty_or_invalid_options() -> None:
    with pytest.raises(ValueError, match="at least one"):
        normalized_loss_balance({})
    with pytest.raises(ValueError, match="must be positive"):
        normalized_loss_balance({"loss": torch.tensor(1.0)}, floor=0.0)
