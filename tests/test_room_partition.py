import numpy as np
import torch

from floorflow.heads.room_partition import (
    RoomPartitionHead,
    build_room_partition_target,
    room_partition_confidence,
    room_partition_loss,
    room_partition_metrics,
)
from floorflow.inference.retrieval_prior import ContourRoomRetriever, prior_one_hot
from floorflow.heads.structured_objects import (
    StructuredObjectHeads,
    compatible_furniture_mask,
    compose_structured_objects,
    constrain_structured_objects,
    group_ce_dice_loss,
    initialize_object_priors,
    object_group_metrics,
    object_group_predictions,
    partition_wall_mask,
    wall_bce_dice_loss,
)
from floorflow.data.taxonomy import build_taxonomy
from collections import Counter


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Toilet": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_partition_target_fills_objects_inside_footprint() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((9, 9), dtype=np.uint8)
    labels[1:8, 1:5] = taxonomy.class_id("room:Bedroom")
    labels[1:8, 5:8] = taxonomy.class_id("room:Kitchen")
    labels[4, 6] = taxonomy.class_id("wall")
    footprint = np.zeros((9, 9), dtype=np.uint8)
    footprint[1:8, 1:8] = 1
    target = build_room_partition_target(labels, footprint, taxonomy)
    assert (target[1:8, 1:8] >= 0).all()
    assert (target[~footprint.astype(bool)] == -100).all()


def test_partition_loss_and_metrics() -> None:
    taxonomy = _taxonomy()
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    logits = torch.zeros(1, 2, 8, 8)
    losses = room_partition_loss(logits, target, tversky_weight=0.4)
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["partition_tversky"])
    metrics = room_partition_metrics(logits, target, taxonomy)
    assert 0.0 <= metrics["room_partition_iou"] <= 1.0


def test_partition_loss_is_case_independent_and_macro_averaged() -> None:
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[0, :, 4:] = 1
    target[1, :4] = 1
    logits = torch.randn(2, 2, 8, 8, generator=torch.Generator().manual_seed(7))
    joint = room_partition_loss(logits, target, tversky_weight=0.4)
    first = room_partition_loss(logits[:1], target[:1], tversky_weight=0.4)
    second = room_partition_loss(logits[1:], target[1:], tversky_weight=0.4)
    assert torch.allclose(
        joint["partition_ce_by_case"][0],
        first["partition_ce_by_case"][0],
    )
    assert torch.allclose(
        joint["partition_dice_by_case"][1],
        second["partition_dice_by_case"][0],
    )
    assert torch.allclose(
        joint["loss"],
        torch.stack([first["loss"], second["loss"]]).mean(),
    )


def test_partition_confidence_prefers_observed_rooms() -> None:
    taxonomy = _taxonomy()
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[:, 1:7, 1:7] = taxonomy.class_id("room:Bedroom")
    labels[:, 1:7, 4] = taxonomy.class_id("furniture:Toilet")
    labels[:, 4, 1:7] = taxonomy.class_id("wall")
    footprint = torch.zeros(1, 8, 8)
    footprint[:, 1:7, 1:7] = 1.0
    weights = room_partition_confidence(
        labels,
        footprint,
        taxonomy,
        furniture_weight=0.15,
        wall_weight=0.05,
    )
    assert weights[0, 2, 2] == 1.0
    assert weights[0, 2, 4] == 0.15
    assert weights[0, 4, 2] == 0.05
    assert weights[0, 0, 0] == 0.0


def test_structured_object_heads_and_losses() -> None:
    taxonomy = _taxonomy()
    heads = StructuredObjectHeads(16, taxonomy)
    features = torch.randn(2, 16, 8, 8)
    outputs = heads(features)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[:, 0, 0] = taxonomy.class_id("room:Bedroom")
    target[:, 1, 1] = taxonomy.opening_ids[0]
    footprint = torch.ones(2, 8, 8)
    opening_ids = torch.tensor(taxonomy.opening_ids)
    opening_loss, _, _ = group_ce_dice_loss(
        outputs["opening_logits"], target, opening_ids, footprint
    )
    wall_loss = wall_bce_dice_loss(
        outputs["wall_logits"], target, taxonomy.class_id("wall"), footprint
    )
    assert torch.isfinite(opening_loss) and torch.isfinite(wall_loss)
    predictions = object_group_predictions(outputs, taxonomy)
    vector_predictions = object_group_predictions(
        outputs,
        taxonomy,
        opening_threshold=torch.zeros(len(taxonomy.opening_ids)),
        furniture_threshold=torch.zeros(len(taxonomy.furniture_ids)),
        score_scale="probability",
    )
    assert vector_predictions["opening"].shape == target.shape
    metrics = object_group_metrics(predictions, target, taxonomy)
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    room_labels = torch.full((2, 8, 8), taxonomy.class_id("room:Bath"), dtype=torch.long)
    furniture_labels = torch.zeros((2, 8, 8), dtype=torch.long)
    furniture_labels[:, 2:4, 2:4] = taxonomy.class_id("furniture:Toilet")
    accepted = compatible_furniture_mask(furniture_labels, room_labels, taxonomy)
    assert accepted[:, 2:4, 2:4].all()
    room_labels.fill_(taxonomy.class_id("room:Bedroom"))
    objects = object_group_predictions(outputs, taxonomy)
    objects["wall"].fill_(1)
    objects["furniture"].fill_(taxonomy.class_id("furniture:Toilet"))
    composed = compose_structured_objects(room_labels, objects, footprint, taxonomy)
    assert torch.all(composed == taxonomy.class_id("wall"))


def test_uncertain_object_logits_do_not_override_rooms() -> None:
    taxonomy = _taxonomy()
    zeros = torch.zeros(1, len(taxonomy.opening_ids), 8, 8)
    furniture = torch.zeros(1, len(taxonomy.furniture_ids), 8, 8)
    predictions = object_group_predictions(
        {
            "opening_logits": zeros,
            "furniture_logits": furniture,
            "wall_logits": torch.full((1, 8, 8), -4.0),
            "opening_background_logits": torch.zeros(1, 8, 8),
            "furniture_background_logits": torch.zeros(1, 8, 8),
        },
        taxonomy,
        opening_threshold=0.5,
        furniture_threshold=0.5,
        wall_threshold=0.5,
        score_scale="probability",
    )
    assert not predictions["opening"].any()
    assert not predictions["furniture"].any()
    assert not predictions["wall"].any()


def test_retrieval_prior_and_head_integration() -> None:
    taxonomy = _taxonomy()
    footprint = np.zeros((32, 32), dtype=np.uint8)
    footprint[4:28, 4:28] = 1
    target = np.zeros((32, 32), dtype=np.uint8)
    target[4:28, 4:28] = taxonomy.class_id("room:Bedroom")
    retriever = ContourRoomRetriever([target], [footprint], taxonomy)
    prior = retriever.retrieve(footprint[None])
    nearest_labels = retriever.retrieve_labels(footprint[None])
    similarity = retriever.nearest_similarity(footprint[None])
    assert nearest_labels.shape == target[None].shape
    assert similarity.shape == (1,)
    one_hot = prior_one_hot(prior, len(taxonomy.room_ids))
    head = RoomPartitionHead(8, len(taxonomy.room_ids), prior_channels=len(taxonomy.room_ids))
    output = head(torch.randn(1, 8, 32, 32), one_hot)
    assert output.shape == (1, len(taxonomy.room_ids), 32, 32)
    assert torch.equal(output.argmax(dim=1), torch.from_numpy(prior))


def test_partition_boundaries_become_walls() -> None:
    rooms = torch.zeros(1, 8, 8, dtype=torch.long)
    rooms[:, :, 4:] = 2
    walls = partition_wall_mask(rooms)
    assert walls[:, :, 3:5].any()


def test_structured_gate_preserves_partition_topology() -> None:
    taxonomy = _taxonomy()
    rooms = torch.zeros(1, 16, 16, dtype=torch.long)
    rooms[:, :, :8] = taxonomy.class_id("room:Bedroom")
    rooms[:, :, 8:] = taxonomy.class_id("room:Kitchen")
    footprint = torch.ones(1, 16, 16)
    objects = {
        "wall": torch.ones(1, 16, 16, dtype=torch.long),
        "opening": torch.full(
            (1, 16, 16),
            taxonomy.opening_ids[0],
            dtype=torch.long,
        ),
        "furniture": torch.full(
            (1, 16, 16),
            taxonomy.class_id("furniture:Toilet"),
            dtype=torch.long,
        ),
    }
    constrained = constrain_structured_objects(objects, rooms, footprint, taxonomy)
    assert (constrained["wall"] == 0).sum() > objects["wall"].numel() // 2
    assert constrained["opening"][:, :, :4].eq(0).all()
    # Toilet is compatible with Bath, not Bedroom or Kitchen.
    assert constrained["furniture"].eq(0).all()


def test_object_background_priors_follow_pixel_frequency() -> None:
    taxonomy = _taxonomy()
    heads = StructuredObjectHeads(16, taxonomy)
    counts = torch.ones(taxonomy.num_classes, dtype=torch.long)
    counts[taxonomy.class_id("door:Swing Beside")] = 2
    counts[taxonomy.class_id("furniture:Toilet")] = 3
    priors = initialize_object_priors(heads, counts)
    assert 0.0 < priors["opening_prior"] < 1.0
    assert 0.0 < priors["furniture_prior"] < 1.0
    assert heads.opening_background.bias.detach().item() > 0.0
