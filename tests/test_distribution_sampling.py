import pytest
import torch
from torch import nn

from floorflow.distribution.sampling import sample_distribution_floorplans


class _TinyDistributionModel(nn.Module):
    target_channels = 2

    def contour_initial_state(self, footprint, floor_ids=None, style_ids=None):
        del floor_ids
        value = footprint.new_full((footprint.shape[0], 1, 1, 1), 0.25)
        if style_ids is not None:
            value = value + 0.25 * style_ids[:, None, None, None]
        state = value.expand(-1, 2, *footprint.shape[-2:]).clone()
        return {
            "cfm": {
                "state": state,
                "pixel_state": state,
                "element_state": state + 0.1,
            }
        }

    def forward(
        self,
        state,
        time,
        footprint,
        floor_ids=None,
        style_ids=None,
        source=None,
    ):
        del time, floor_ids, source
        bias = state.new_zeros((state.shape[0], 1, 1, 1))
        if style_ids is not None:
            bias = 0.1 * style_ids[:, None, None, None]
        return bias.expand_as(state) * footprint


def test_distribution_sampler_starts_from_contour_source() -> None:
    model = _TinyDistributionModel()
    footprint = torch.ones(1, 1, 4, 4)
    state, initial = sample_distribution_floorplans(
        model,
        footprint,
        style_ids=torch.tensor([1]),
        steps=2,
    )
    assert initial["cfm"]["state"][0, 0, 0, 0].item() == 0.5
    assert state[0, 0, 0, 0].item() > 0.5


def test_distribution_sampler_can_add_source_noise() -> None:
    model = _TinyDistributionModel()
    footprint = torch.ones(1, 1, 4, 4)
    torch.manual_seed(1)
    left, _ = sample_distribution_floorplans(
        model,
        footprint,
        style_ids=torch.tensor([0]),
        steps=1,
        noise_scale=0.1,
    )
    torch.manual_seed(1)
    right, _ = sample_distribution_floorplans(
        model,
        footprint,
        style_ids=torch.tensor([1]),
        steps=1,
        noise_scale=0.1,
    )
    assert not torch.equal(left, right)


def test_zero_steps_returns_distribution_source_without_flow() -> None:
    model = _TinyDistributionModel()
    footprint = torch.ones(1, 1, 4, 4)
    state, initial = sample_distribution_floorplans(
        model,
        footprint,
        style_ids=torch.tensor([1]),
        steps=0,
    )
    assert torch.equal(state, initial["cfm"]["state"])


def test_distribution_sampler_can_start_from_element_state() -> None:
    model = _TinyDistributionModel()
    footprint = torch.ones(1, 1, 4, 4)
    state, initial = sample_distribution_floorplans(
        model,
        footprint,
        steps=0,
        source_mode="element",
    )
    assert torch.equal(state, initial["cfm"]["element_state"])
    assert not torch.equal(state, initial["cfm"]["pixel_state"])


def test_zero_style_gate_matches_unconditioned_path() -> None:
    model = _TinyDistributionModel()
    footprint = torch.ones(1, 1, 4, 4)
    base, _ = sample_distribution_floorplans(
        model,
        footprint,
        steps=2,
    )
    gated, _ = sample_distribution_floorplans(
        model,
        footprint,
        style_ids=torch.tensor([1]),
        steps=2,
        style_gate=0.0,
    )
    assert torch.equal(base, gated)


def test_alternative_ode_solvers_are_finite_and_explicit() -> None:
    model = _TinyDistributionModel()
    footprint = torch.ones(1, 1, 4, 4)
    for solver in ("shifted_euler", "heun"):
        state, _ = sample_distribution_floorplans(
            model,
            footprint,
            steps=2,
            solver=solver,
        )
        assert state.shape == (1, 2, 4, 4)
        assert torch.isfinite(state).all()
    with pytest.raises(ValueError, match="solver"):
        sample_distribution_floorplans(
            model,
            footprint,
            steps=2,
            solver="invalid",
        )
