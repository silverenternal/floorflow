"""Tests for the dense residual room-identity corrector."""

from __future__ import annotations

import torch

from floorflow.heads.predicted_room_identity import (
    PredictedRoomIdentityHead,
    room_identity_loss,
)


def test_dense_identity_head_starts_as_base_logits_and_has_gradients() -> None:
    head = PredictedRoomIdentityHead(6, 3, hidden_channels=16, blocks=1)
    features = torch.randn(2, 6, 8, 8, requires_grad=True)
    base = torch.randn(2, 3, 8, 8)
    output = head(features, base)
    assert torch.equal(output["room_logits"], base)
    loss = output["room_logits"].square().mean()
    loss.backward()
    assert head.input[0].weight.grad is not None


def test_room_identity_loss_is_finite_and_accepts_class_weights() -> None:
    logits = torch.randn(1, 2, 8, 8, requires_grad=True)
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[:, :, :4] = 10
    labels[:, :, 4:] = 20
    loss, components = room_identity_loss(
        logits,
        labels,
        torch.ones(1, 8, 8),
        (10, 20),
        class_weights=torch.tensor([1.0, 2.0]),
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())


def test_room_identity_loss_handles_fully_invalid_target_without_branch_sync() -> None:
    logits = torch.randn(1, 2, 8, 8, requires_grad=True)
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    loss, components = room_identity_loss(
        logits,
        labels,
        torch.ones(1, 8, 8),
        (10, 20),
    )
    assert loss.item() == 0.0
    assert all(value.item() == 0.0 for value in components.values())
    loss.backward()
    assert logits.grad is not None


def test_image_residual_keeps_identity_start_and_receives_gradient() -> None:
    head = PredictedRoomIdentityHead(
        6,
        3,
        hidden_channels=16,
        blocks=1,
        use_image=True,
    )
    features = torch.randn(2, 6, 8, 8)
    image = torch.randn(2, 3, 8, 8, requires_grad=True)
    base = torch.randn(2, 3, 8, 8)

    output = head(features, base, image=image)
    output["room_logits"].square().mean().backward()

    assert torch.equal(output["room_logits"], base)
    assert head.image_output is not None
    assert head.image_output.weight.grad is not None
    assert torch.count_nonzero(head.image_output.weight.grad) == 0
    assert image.grad is not None
