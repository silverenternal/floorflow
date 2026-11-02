from __future__ import annotations

import numpy as np
import torch

from floorflow.model.program_instance_router import ProgramInstanceRouter
from floorflow.objectives.room_matching_types import RoomQueryMatch
from floorflow.training.program_instance_training import (
    program_instance_assignment_loss,
)


def test_program_router_is_schedule_conditioned_and_trainable() -> None:
    router = ProgramInstanceRouter(16, 3)
    queries = torch.randn(2, 5, 16, requires_grad=True)
    first = router(
        queries,
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    )
    second = router(
        queries,
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
    )
    assert first["assignment_logits"].shape == (2, 5, 4)
    assert first["room_class_assignment_logits"].shape == (2, 5, 4)
    assert torch.equal(
        first["room_class_assignment_logits"][..., :-1],
        first["assignment_logits"][..., 1:],
    )
    assert torch.equal(
        first["room_class_assignment_logits"][..., -1],
        first["assignment_logits"][..., 0],
    )
    assert not torch.allclose(
        first["foreground_assignment_logits"],
        second["foreground_assignment_logits"],
    )
    first["assignment_logits"].square().mean().backward()
    assert router.class_embedding.grad is not None
    assert queries.grad is not None


def test_program_instance_loss_uses_matching_and_schedule() -> None:
    target_masks = torch.zeros(1, 2, 4, 4)
    target_masks[0, 0, :2] = 1.0
    target_masks[0, 1, 2:] = 1.0
    assignment_logits = torch.randn(1, 2, 4, requires_grad=True)
    mask_logits = torch.full((1, 2, 4, 4), -4.0)
    mask_logits[0, 0, :2] = 4.0
    mask_logits[0, 1, 2:] = 4.0
    loss, components = program_instance_assignment_loss(
        assignment_logits,
        mask_logits,
        target_classes=torch.tensor([[0, 2]]),
        target_masks=target_masks,
        target_valid=torch.tensor([[True, True]]),
        program=torch.tensor([[1.0, 0.0, 1.0]]),
    )
    assert torch.isfinite(loss)
    assert 0.0 <= float(
        components["program_instance_assignment_accuracy"]
    ) <= 1.0
    loss.backward()
    assert assignment_logits.grad is not None


def test_program_schedule_loss_is_autocast_safe() -> None:
    target_masks = torch.zeros(1, 2, 4, 4)
    target_masks[0, 0, :2] = 1.0
    target_masks[0, 1, 2:] = 1.0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _components = program_instance_assignment_loss(
            torch.randn(1, 2, 4),
            torch.randn(1, 2, 4, 4),
            target_classes=torch.tensor([[0, 2]]),
            target_masks=target_masks,
            target_valid=torch.tensor([[True, True]]),
            program=torch.tensor([[1.0, 0.0, 1.0]]),
        )
    assert torch.isfinite(loss)


def test_program_assignment_reuses_shared_room_identity_matching() -> None:
    target_masks = torch.zeros(1, 2, 4, 4)
    target_masks[0, 0, :2] = 1.0
    target_masks[0, 1, 2:] = 1.0
    mask_logits = torch.full((1, 2, 4, 4), -4.0)
    mask_logits[0, 1, :2] = 4.0
    mask_logits[0, 0, 2:] = 4.0
    assignment_logits = torch.zeros(1, 2, 4)
    assignment_logits[0, 1, 1] = 8.0
    assignment_logits[0, 0, 3] = 8.0
    assignment_logits.requires_grad_(True)
    matches = [
        RoomQueryMatch(
            predicted_indices=np.asarray([1, 0]),
            target_indices=np.asarray([0, 1]),
        )
    ]
    loss, components = program_instance_assignment_loss(
        assignment_logits,
        mask_logits,
        target_classes=torch.tensor([[0, 2]]),
        target_masks=target_masks,
        target_valid=torch.tensor([[True, True]]),
        program=torch.tensor([[1.0, 0.0, 1.0]]),
        matches=matches,
    )
    assert components["program_instance_assignment_accuracy"] == 1.0
    loss.backward()
    assert assignment_logits.grad is not None
