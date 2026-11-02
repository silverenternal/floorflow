"""Tests for learned room-instance boundary readout."""

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.room_instance_boundary import (
    BoundaryAwareRoomSegmenter,
    room_boundary_loss,
    room_boundary_metrics,
    room_instance_boundary_target,
    split_room_instances,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Bath": 4}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_boundary_target_separates_same_class_instances() -> None:
    taxonomy = _taxonomy()
    class_id = taxonomy.class_id("room:Bedroom")
    labels = torch.zeros(1, 8, 10, dtype=torch.long)
    labels[:, :, :4] = class_id
    labels[:, :, 5:] = class_id
    footprint = torch.ones(1, 8, 10)
    target = room_instance_boundary_target(labels, footprint, taxonomy)
    assert target[0, :, 3].all()
    assert target[0, :, 5].all()
    assert not target[0, :, 2].any()


def test_boundary_target_ignores_different_class_rooms() -> None:
    taxonomy = _taxonomy()
    bedroom = taxonomy.class_id("room:Bedroom")
    bath = taxonomy.class_id("room:Bath")
    labels = torch.zeros(1, 8, 10, dtype=torch.long)
    labels[:, :, :4] = bedroom
    labels[:, :, 5:] = bath
    target = room_instance_boundary_target(
        labels,
        torch.ones(1, 8, 10),
        taxonomy,
    )
    assert not target.any()


def test_boundary_readout_is_exact_when_threshold_is_one() -> None:
    taxonomy = _taxonomy()
    room_id = taxonomy.class_id("room:Bedroom")
    labels = torch.full((1, 8, 8), room_id, dtype=torch.long)
    footprint = torch.ones(1, 8, 8)
    logits = torch.full((1, 1, 8, 8), 20.0)
    split = split_room_instances(
        labels,
        logits,
        footprint,
        torch.tensor(taxonomy.room_ids),
        threshold=1.1,
    )
    assert torch.equal(split, labels)


def test_boundary_loss_reaches_logits() -> None:
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    target = torch.zeros(1, 8, 8, dtype=torch.bool)
    target[:, 3:5] = True
    loss, components = room_boundary_loss(
        logits,
        target,
        torch.ones(1, 8, 8),
    )
    loss.backward()
    assert components["room_boundary_bce"] > 0.0
    assert logits.grad is not None


def test_tversky_term_penalizes_false_negatives() -> None:
    target = torch.zeros(1, 8, 8, dtype=torch.bool)
    target[:, 3:5] = True
    logits = torch.full((1, 1, 8, 8), -8.0, requires_grad=True)
    loss, components = room_boundary_loss(
        logits,
        target,
        torch.ones(1, 8, 8),
        tversky_weight=0.5,
    )
    loss.backward()
    assert components["room_boundary_tversky"] > 0.0


def test_multiscale_boundary_decoder_has_expected_shape_and_budget() -> None:
    taxonomy = _taxonomy()
    model = BoundaryAwareRoomSegmenter(
        taxonomy,
        decoder_channels=16,
        boundary_channels=16,
        boundary_mode="multiscale",
    ).eval()
    output = model(
        torch.ones(1, 32, 32),
        control_image=torch.randn(1, 3, 32, 32),
    )
    assert output["boundary_logits"].shape == (1, 1, 32, 32)
    assert sum(
        parameter.numel() for parameter in model.boundary.parameters()
    ) < 100_000


def test_boundary_metrics_report_precision_recall_and_f1() -> None:
    target = torch.zeros(1, 8, 8, dtype=torch.bool)
    target[:, 3:5] = True
    logits = torch.full((1, 1, 8, 8), -20.0)
    logits[:, :, 3:5] = 20.0
    metrics = room_boundary_metrics(
        logits,
        target,
        torch.ones(1, 8, 8),
        threshold=0.5,
    )
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_independent_boundary_decoder_has_expected_shape_and_budget() -> None:
    taxonomy = _taxonomy()
    model = BoundaryAwareRoomSegmenter(
        taxonomy,
        decoder_channels=16,
        boundary_channels=16,
        boundary_mode="independent",
    ).eval()
    output = model(
        torch.ones(1, 32, 32),
        control_image=torch.randn(1, 3, 32, 32),
    )
    assert output["boundary_logits"].shape == (1, 1, 32, 32)
    assert sum(
        parameter.numel() for parameter in model.boundary.parameters()
    ) < 100_000


def test_dense_conditioned_boundary_decoder_has_expected_shape() -> None:
    taxonomy = _taxonomy()
    model = BoundaryAwareRoomSegmenter(
        taxonomy,
        decoder_channels=16,
        boundary_channels=16,
        boundary_mode="dense",
    ).eval()
    output = model(
        torch.ones(1, 32, 32),
        control_image=torch.randn(1, 3, 32, 32),
    )
    assert output["boundary_logits"].shape == (1, 1, 32, 32)
