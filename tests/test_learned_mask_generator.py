from __future__ import annotations

import pytest
import torch

from floorflow.model.learned_mask_generator import LearnedMaskGenerator


def test_learned_mask_generator_shapes_and_coverage() -> None:
    generator = LearnedMaskGenerator(16, hidden_channels=16)
    control = torch.rand(2, 16, 32, 32)
    footprint = torch.ones(2, 1, 32, 32)
    output = generator(control, footprint, target_ratio=0.25)
    assert output.probability.shape == control.shape
    assert output.hard.shape == control.shape
    assert output.channel.shape == (2, 16)
    assert torch.all(output.coverage >= 0.0)
    assert torch.all(output.coverage <= 1.0)


def test_learned_mask_generator_is_differentiable() -> None:
    generator = LearnedMaskGenerator(8, hidden_channels=8)
    control = torch.rand(1, 8, 16, 16)
    footprint = torch.ones(1, 1, 16, 16)
    output = generator(control, footprint)
    loss = output.probability.mean()
    loss.backward()
    assert generator.spatial_head.weight.grad is not None


def test_learned_mask_generator_ignores_hidden_target_values() -> None:
    generator = LearnedMaskGenerator(4, hidden_channels=8)
    footprint = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    visibility = torch.zeros_like(footprint)
    visibility[:, :, :, 8:] = True
    first = torch.zeros(1, 4, 16, 16)
    second = first.clone()
    second[:, :, :, :8] = 99.0
    with torch.no_grad():
        first_mask = generator(first, footprint, visibility=visibility)
        with pytest.raises(ValueError, match="hidden controls"):
            generator(second, footprint, visibility=visibility)
    assert first_mask.hard.shape == first.shape


def test_known_mask_is_preserved_without_exceeding_budget() -> None:
    generator = LearnedMaskGenerator(4, hidden_channels=8)
    footprint = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    known = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    known[:, :, :4, :4] = True
    output = generator(
        torch.zeros(1, 4, 16, 16),
        footprint,
        target_ratio=0.25,
        known_mask=known,
    )
    assert output.hard[:, :, :4, :4].all()
    expected = int(round(4 * 16 * 16 * 0.25))
    assert int(output.hard.sum()) == expected


def test_semantic_budgets_preserve_total_and_visibility_floors() -> None:
    generator = LearnedMaskGenerator(4, hidden_channels=8)
    footprint = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    output = generator(
        torch.zeros(1, 4, 16, 16),
        footprint,
        target_ratio=0.5,
        channel_groups=(
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([2]),
            torch.tensor([3]),
        ),
        minimum_visible=(0.15, 0.35, 0.5, 0.5),
    )
    expected = int(round(4 * 16 * 16 * 0.5))
    assert int(output.hard.sum()) == expected
    family_mask = output.hard[0].float().amax(dim=1)
    assert float(family_mask[0].mean()) >= 0.15
    assert float(family_mask[1].mean()) >= 0.65
    assert float(family_mask[2].mean()) >= 0.50
    assert float(family_mask[3].mean()) >= 0.50
