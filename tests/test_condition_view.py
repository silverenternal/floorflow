from __future__ import annotations

import pytest
import torch

from floorflow.flow.condition_view import (
    ConditionView,
    contour_only_view,
    layout_conditioned_view,
    sparse_control_view,
)


def test_hidden_control_values_cannot_be_observed() -> None:
    footprint = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    hidden = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    hidden[:, :, :, :4] = True
    first_target = torch.full((1, 3, 8, 8), 3.0)
    second_target = first_target.clone()
    second_target[:, :, :, :4] = 97.0
    first = sparse_control_view(first_target, footprint, hidden)
    second = sparse_control_view(second_target, footprint, hidden)
    assert torch.equal(first.control, second.control)
    assert torch.equal(first.visibility, second.visibility)
    assert first.manifest()["mode"] == "sparse_control"


def test_visible_control_values_remain_observable() -> None:
    footprint = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    hidden = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    hidden[:, :, :, :4] = True
    first = sparse_control_view(torch.ones(1, 2, 8, 8), footprint, hidden)
    second = sparse_control_view(torch.full((1, 2, 8, 8), 2.0), footprint, hidden)
    assert not torch.equal(first.control, second.control)


def test_contour_only_exposes_no_target_controls() -> None:
    footprint = torch.ones(2, 1, 8, 8, dtype=torch.bool)
    view = contour_only_view(footprint, 5)
    assert view.is_target_control_free
    assert not view.visibility.any()
    assert not view.control.any()
    assert view.hidden_mask.all()


def test_condition_view_rejects_hidden_control_leakage() -> None:
    footprint = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    control = torch.ones(1, 2, 4, 4)
    visibility = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    visibility[:, :, :2] = True
    with pytest.raises(ValueError, match="hidden controls"):
        ConditionView("sparse_control", footprint, control, visibility)


def test_layout_control_requires_full_footprint_visibility() -> None:
    footprint = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    control = torch.ones(1, 2, 4, 4)
    visibility = footprint.clone()
    visibility[:, :, 0, 0] = False
    with pytest.raises(ValueError, match="full interior"):
        ConditionView("layout_conditioned", footprint, control, visibility)
    view = layout_conditioned_view(control, footprint)
    assert torch.equal(view.visibility, footprint)
    assert not view.is_target_control_free
