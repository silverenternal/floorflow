import torch

from floorflow.heads.room_mask_decoder import room_mask_loss
from floorflow.objectives.furniture_vector_refine import (
    furniture_containment_loss,
)


def test_positive_weight_changes_sparse_mask_objective() -> None:
    predictions = {
        "class_logits": torch.zeros(1, 1, 2, requires_grad=True),
        "mask_logits": torch.zeros(1, 1, 8, 8, requires_grad=True),
    }
    classes = torch.zeros(1, 1, dtype=torch.long)
    masks = torch.zeros(1, 1, 8, 8)
    masks[0, 0, 3:5, 3:5] = 1.0
    valid = torch.ones(1, 1, dtype=torch.bool)
    baseline, _ = room_mask_loss(
        predictions,
        classes,
        masks,
        valid,
        mask_pos_weight=1.0,
    )
    balanced, components = room_mask_loss(
        predictions,
        classes,
        masks,
        valid,
        mask_pos_weight=4.0,
    )
    assert balanced > baseline
    assert components["room_mask_dice_bce_loss"] >= 0.0
    assert components["room_mask_matched_dice"] > 0.0
    assert components["room_mask_matched_iou"] > 0.0
    assert components["room_mask_matched_predicted_area"] > 0.0
    assert components["room_mask_matched_target_area"] > 0.0
    assert 0.0 <= components["room_mask_matched_centroid_error"] <= 2.0**0.5
    balanced.backward()
    gradient = predictions["mask_logits"].grad
    assert gradient is not None
    assert gradient[0, 0, 3:5, 3:5].mean() < gradient[
        0,
        0,
        0:2,
        0:2,
    ].mean()


def test_unmatched_query_mask_is_supervised_as_empty() -> None:
    predictions = {
        "class_logits": torch.zeros(1, 2, 2, requires_grad=True),
        "mask_logits": torch.zeros(1, 2, 8, 8, requires_grad=True),
    }
    classes = torch.zeros(1, 1, dtype=torch.long)
    masks = torch.zeros(1, 1, 8, 8)
    masks[0, 0, 3:5, 3:5] = 1.0
    valid = torch.ones(1, 1, dtype=torch.bool)
    loss, components = room_mask_loss(
        predictions,
        classes,
        masks,
        valid,
        unmatched_mask_weight=0.5,
    )
    loss.backward()
    assert components["room_mask_unmatched_loss"] > 0.0
    gradient = predictions["mask_logits"].grad
    assert gradient is not None
    assert gradient[0, 1].abs().mean() > 0.0


def test_matched_mask_centroid_supervises_reference_anchors() -> None:
    reference = torch.tensor(
        [[[[0.1, 0.1]]]],
        requires_grad=True,
    )
    predictions = {
        "class_logits": torch.zeros(1, 1, 2),
        "mask_logits": torch.zeros(1, 1, 8, 8),
        "adaptive_query_diagnostics": {"reference_samples": reference},
    }
    masks = torch.zeros(1, 1, 8, 8)
    masks[0, 0, 5:7, 5:7] = 1.0
    loss, components = room_mask_loss(
        predictions,
        torch.zeros(1, 1, dtype=torch.long),
        masks,
        torch.ones(1, 1, dtype=torch.bool),
        reference_anchor_weight=0.2,
        mask_centroid_weight=0.5,
    )
    assert components["room_mask_reference_anchor_loss"] > 0.0
    assert components["room_mask_centroid_loss"] > 0.0
    loss.backward()
    assert reference.grad is not None


def test_tversky_is_live_and_containment_penalizes_outside_room() -> None:
    predictions = {
        "class_logits": torch.zeros(1, 1, 2, requires_grad=True),
        "mask_logits": torch.zeros(1, 1, 8, 8, requires_grad=True),
    }
    classes = torch.zeros(1, 1, dtype=torch.long)
    masks = torch.zeros(1, 1, 8, 8)
    masks[0, 0, 3:5, 3:5] = 1.0
    valid = torch.ones(1, 1, dtype=torch.bool)
    loss, components = room_mask_loss(
        predictions,
        classes,
        masks,
        valid,
        mask_tversky_weight=0.4,
    )
    assert components["room_mask_tversky_loss"] > 0.0
    outside_mask = torch.full((1, 1, 8, 8), 8.0, requires_grad=True)
    room_support = torch.zeros(1, 8, 8)
    room_support[:, 2:6, 2:6] = 1.0
    containment = furniture_containment_loss(outside_mask, room_support)
    total = loss + containment
    total.backward()
    assert predictions["mask_logits"].grad is not None
    assert outside_mask.grad is not None
