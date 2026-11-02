from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.objectives.per_case_binary import (
    per_case_binary_cross_entropy,
    per_case_focal_loss,
    per_case_soft_dice,
)


def test_per_case_bce_matches_manual_case_average() -> None:
    prediction = torch.tensor(
        [
            [0.9, 0.8, 0.1, 0.2],
            [0.7, 0.3, 0.0, 0.0],
        ]
    )
    target = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
        ]
    )
    actual = per_case_binary_cross_entropy(prediction, target, mask)
    logits = torch.logit(prediction.clamp(1.0e-6, 1.0 - 1.0e-6))
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    expected = torch.stack(
        [
            elementwise[0][mask[0]].mean(),
            elementwise[1][mask[1]].mean(),
        ]
    ).mean()
    assert torch.allclose(actual, expected)


def test_per_case_dice_does_not_let_large_cases_dominate() -> None:
    prediction = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.9, 0.9, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    loss = per_case_soft_dice(prediction, target)
    first_case = 1.0 - 1.0 / 5.0
    second_case = 1.0 - 4.6 / 4.8
    assert torch.allclose(loss, torch.tensor((first_case + second_case) / 2.0))
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad.abs().sum() > 0


def test_per_case_focal_matches_masked_case_average() -> None:
    logits = torch.tensor(
        [
            [2.0, -1.0, 0.5, 0.0],
            [-1.0, 2.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
        ]
    )
    loss = per_case_focal_loss(logits, target, mask)
    probability = logits.sigmoid()
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    p_t = probability * target + (1 - probability) * (1 - target)
    alpha_t = 0.25 * target + 0.75 * (1 - target)
    focal = alpha_t * (1 - p_t).square() * elementwise
    expected = torch.stack(
        [
            focal[0][mask[0]].mean(),
            focal[1][mask[1]].mean(),
        ]
    ).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_per_case_bce_accepts_scalar_tensor_pos_weight_for_batches() -> None:
    prediction = torch.full((2, 3), 0.5, requires_grad=True)
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    loss = per_case_binary_cross_entropy(
        prediction,
        target,
        pos_weight=torch.tensor(2.0),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None


def test_per_case_bce_accepts_singleton_batch_pos_weight() -> None:
    prediction = torch.full((1, 3), 0.5, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]])
    loss = per_case_binary_cross_entropy(
        prediction,
        target,
        pos_weight=torch.tensor([[[2.0]]]),
    )
    assert torch.isfinite(loss)
