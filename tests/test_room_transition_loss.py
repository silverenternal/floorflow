from __future__ import annotations

import torch

from floorflow.objectives.topology_closure import room_transition_loss


def test_room_transition_loss_uses_internal_class_boundaries() -> None:
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :, 4:] = 1
    loss = room_transition_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_room_transition_loss_is_finite_without_boundaries() -> None:
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    loss = room_transition_loss(logits, target)
    assert float(loss.detach()) == 0.0
