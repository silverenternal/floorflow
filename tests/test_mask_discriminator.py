from __future__ import annotations

import torch

from floorflow.model.mask_discriminator import (
    MaskDifficultyDiscriminator,
    difficulty_rank_loss,
    difficulty_rank_targets,
)


def test_mask_difficulty_discriminator_shapes_and_gradients() -> None:
    model = MaskDifficultyDiscriminator(16, hidden_channels=8)
    mask = torch.zeros(2, 16, 32, 32, dtype=torch.bool)
    mask[:, :, :8] = True
    difficulty = torch.rand(2, 1, 32, 32)
    footprint = torch.ones(2, 1, 32, 32)
    logits, diagnostics = model(mask, difficulty, footprint)
    assert logits.shape == (2, 1)
    assert diagnostics["predicted_hardness"].shape == (2,)
    logits.mean().backward()
    assert model.trunk[0].weight.grad is not None


def test_difficulty_rank_targets_cover_percentiles() -> None:
    targets = difficulty_rank_targets(torch.tensor([0.2, 0.01, 0.8, 0.4]))
    assert targets.tolist() == [0.3333333432674408, 0.0, 1.0, 0.6666666865348816]


def test_difficulty_rank_loss_has_finite_gradients() -> None:
    model = MaskDifficultyDiscriminator(4, hidden_channels=8)
    mask = torch.zeros(4, 4, 8, 8, dtype=torch.bool)
    mask[:, :, :2] = True
    difficulty = torch.tensor([0.01, 0.2, 0.8, 2.0])[:, None, None, None]
    footprint = torch.ones_like(difficulty)
    logits, _ = model(mask, difficulty.expand(4, 1, 8, 8), footprint.expand(4, 1, 8, 8))
    loss, diagnostics = difficulty_rank_loss(logits, difficulty)
    assert torch.isfinite(loss)
    assert torch.isfinite(diagnostics["difficulty_spearman"])
    loss.backward()
    assert model.head[0].weight.grad is not None
