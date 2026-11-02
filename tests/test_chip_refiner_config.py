from __future__ import annotations

import pytest
import torch
from torch import nn

from floorflow.chip_flow.refiner_config import (
    configure_chip_projector,
    parse_refiner_weight_overrides,
    project_after_refiner,
    refiner_comparison_status,
)


def test_refiner_weight_overrides_preserve_paper_defaults() -> None:
    weights = parse_refiner_weight_overrides(["group=2", "boundary=4.0"])
    assert weights["overlap"] == 10.0
    assert weights["wl"] == 1.0
    assert weights["group"] == 2.0
    assert weights["boundary"] == 4.0


def test_refiner_weight_overrides_do_not_mutate_base() -> None:
    base = {"overlap": 10.0, "group": 1.0}
    weights = parse_refiner_weight_overrides(["group=3"], base=base)
    assert base["group"] == 1.0
    assert weights["group"] == 3.0


@pytest.mark.parametrize(
    "override",
    ["group", "group=nan", "group=-1", "unknown=2"],
)
def test_refiner_weight_overrides_reject_invalid_values(override: str) -> None:
    with pytest.raises(ValueError):
        parse_refiner_weight_overrides([override])


def test_post_refiner_projection_can_be_disabled() -> None:
    latent = torch.randn(2, 3, 3)
    output = project_after_refiner(
        latent,
        latent.clone(),
        torch.zeros(2, 3, 5),
        torch.ones(2, 3),
        iterations=0,
        strength=1.0,
    )
    assert output is latent


def test_post_refiner_projection_rejects_invalid_strength() -> None:
    latent = torch.randn(1, 2, 3)
    with pytest.raises(ValueError):
        project_after_refiner(
            latent,
            latent,
            torch.zeros(1, 2, 5),
            torch.ones(1, 2),
            iterations=1,
            strength=1.1,
        )


def test_configure_chip_projector_sets_boundary_axes() -> None:
    model = nn.Module()
    model.projector = False
    model.projector_iterations = 2
    model.projector_boundary_axes = False
    model.projector_boundary_lock_corners = True
    configured = configure_chip_projector(
        model,
        enabled=True,
        iterations=8,
        boundary_axes=True,
        boundary_lock_corners=False,
    )
    assert configured.projector is True
    assert configured.projector_iterations == 8
    assert configured.projector_boundary_axes is True
    assert configured.projector_boundary_lock_corners is False


def test_single_refiner_setting_is_not_a_rejected_sweep() -> None:
    assert refiner_comparison_status(1, 0.0) == "single_setting_diagnostic"
    assert refiner_comparison_status(2, 0.04) == "refiner_lever_rejected"
    assert refiner_comparison_status(2, 0.05) == "pass"
