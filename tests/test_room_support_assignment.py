import torch

from floorflow.objectives.room_support_assignment import (
    room_support_assignment_loss,
)


def _owner_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    owner = torch.zeros(1, 2, 8, 8)
    owner[:, 0, :, :4] = 1.0
    owner[:, 1, :, 4:] = 1.0
    classes = torch.tensor(
        [
            [
                [4.0, 0.0],
                [0.0, 4.0],
            ]
        ]
    )
    target_masks = torch.zeros(1, 2, 8, 8)
    target_masks[0, 0, :, 4:] = 1.0
    target_masks[0, 1, :, :4] = 1.0
    target_classes = torch.tensor([[1, 0]])
    return owner, classes, target_masks, target_classes


def test_room_support_assignment_matches_owner_partition() -> None:
    owner, classes, targets, target_classes = _owner_case()
    loss, components = room_support_assignment_loss(
        owner,
        classes,
        target_classes,
        targets,
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    assert components["room_support_owner_exact"] > 0.99
    assert components["room_support_valid_pixels"] == 64


def test_room_support_assignment_is_sensitive_to_owner_swap() -> None:
    owner, classes, targets, target_classes = _owner_case()
    correct, _ = room_support_assignment_loss(
        owner,
        classes,
        target_classes,
        targets,
        torch.ones(1, 2, dtype=torch.bool),
    )
    ambiguous = torch.full_like(owner, 0.5)
    incorrect, components = room_support_assignment_loss(
        ambiguous,
        classes,
        target_classes,
        targets,
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert incorrect > correct
    assert components["room_support_owner_exact"] < 0.75


def test_room_support_assignment_reaches_owner_probability() -> None:
    owner, classes, targets, target_classes = _owner_case()
    owner = owner * 0.9 + 0.05
    owner.requires_grad_(True)
    classes.requires_grad_(True)
    loss, _ = room_support_assignment_loss(
        owner,
        classes,
        target_classes,
        targets,
        torch.ones(1, 2, dtype=torch.bool),
    )
    loss.backward()
    assert owner.grad is not None
    assert owner.grad.abs().sum() > 0
