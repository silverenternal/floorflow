import pytest
import torch
from torch import nn

from floorflow.training.gradient_health import (
    clip_gradients_fail_closed,
    nonfinite_gradient_names,
)


def test_nonfinite_gradient_is_rejected_before_clipping() -> None:
    model = nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[float("inf"), 0.0]])
    assert nonfinite_gradient_names(model) == ("weight",)
    with pytest.raises(FloatingPointError, match="weight"):
        clip_gradients_fail_closed(model, max_norm=1.0)


def test_finite_gradients_are_clipped() -> None:
    model = nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    norm = clip_gradients_fail_closed(model, max_norm=1.0)
    assert norm == pytest.approx(5.0)
    assert model.weight.grad.norm() == pytest.approx(1.0)
