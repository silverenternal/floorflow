from __future__ import annotations

import torch

from floorflow.heads.room_owner_readout import (
    exclusive_query_owner_probability,
    room_owner_readout,
)


def _predictions() -> dict[str, torch.Tensor]:
    masks = torch.full((1, 2, 4, 4), -8.0)
    masks[:, 0, :, :2] = 8.0
    masks[:, 1, :, 2:] = 8.0
    classes = torch.zeros(1, 2, 3)
    classes[..., 0] = 4.0
    classes[..., -1] = -4.0
    return {
        "mask_logits": masks.requires_grad_(),
        "class_logits": classes.requires_grad_(),
        "query_existence_logits": torch.full((1, 2), 8.0),
    }


def test_owner_readout_preserves_same_class_instance_boundary() -> None:
    predictions = _predictions()
    output = room_owner_readout(
        predictions,
        2,
        footprint=torch.ones(1, 4, 4),
    )
    assert output["owner_map"][0, 0, 0].item() == 0
    assert output["owner_map"][0, 0, 3].item() == 1
    assert output["owner_boundary"][0, :, 2].all()
    assert not output["owner_boundary"][0, :, 0].any()
    assert output["owner_preserving_logits"].shape == (1, 2, 4, 4)
    output["owner_preserving_logits"].softmax(dim=1).sum().backward()
    assert predictions["mask_logits"].grad is not None
    assert predictions["class_logits"].grad is not None


def test_exclusive_owner_probability_assigns_one_background_partition() -> None:
    masks = torch.zeros(1, 2, 2, 2)
    masks[:, 0] = 2.0
    masks[:, 1] = 1.0
    owner = exclusive_query_owner_probability(masks)
    assert torch.all(owner.sum(dim=1) < 1.0)
    assert owner.argmax(dim=1).eq(0).all()


def test_owner_readout_respects_footprint() -> None:
    predictions = _predictions()
    footprint = torch.ones(1, 4, 4)
    footprint[:, 0, :] = 0.0
    output = room_owner_readout(
        predictions,
        2,
        footprint=footprint,
    )
    assert not output["owner_foreground"][0, 0].any()


def test_owner_confidence_uses_selected_slot_as_authoritative_gate() -> None:
    predictions = _predictions()
    predictions["mask_logits"].data.fill_(8.0)
    predictions["query_existence_logits"] = torch.tensor([[-8.0, 8.0]])
    predictions["query_count_selection_mask"] = torch.tensor(
        [[True, False]]
    )
    output = room_owner_readout(
        predictions,
        2,
        footprint=torch.ones(1, 4, 4),
    )
    assert output["owner_foreground"].all()
    assert output["owner_availability"].eq(1.0).all()


def test_soft_owner_foreground_keeps_training_gradient_alive() -> None:
    predictions = _predictions()
    predictions["mask_logits"].data.fill_(8.0)
    predictions["query_existence_logits"] = torch.tensor([[-8.0, 8.0]])
    predictions["query_count_selection_mask"] = torch.tensor(
        [[True, False]]
    )
    output = room_owner_readout(
        predictions,
        2,
        fallback_logits=torch.full((1, 2, 4, 4), -10.0),
        soft_foreground=True,
    )
    assert output["owner_soft_weight"].min() > 0.99
    output["owner_preserving_logits"].sum().backward()
    assert predictions["mask_logits"].grad is not None
