from collections import Counter

import pytest
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.polygon_loop import (
    PolygonLoopHead,
    polygon_loop_loss,
    room_query_metrics,
)
from floorflow.representation.polygon_targets import (
    decode_polygon_loops,
    polygon_instance_targets,
    rasterize_room_polygons,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Kitchen": 4}),
            "furniture": Counter({"Sink": 4}),
            "door": Counter({"Swing Beside": 4}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_polygon_targets_are_closed_and_normalized() -> None:
    taxonomy = _taxonomy()
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[0, 4:16, 3:15] = taxonomy.room_ids[0]
    labels[0, 18:28, 17:29] = taxonomy.room_ids[1]
    classes, polygons, valid = polygon_instance_targets(
        labels,
        taxonomy,
        max_instances=4,
        points=12,
    )
    assert classes.shape == valid.shape == (1, 4)
    assert polygons.shape == (1, 4, 12, 2)
    assert valid.sum() == 2
    assert torch.all((polygons >= 0.0) & (polygons <= 1.0))
    rasterized = rasterize_room_polygons(
        polygons,
        classes,
        valid,
        taxonomy,
        height=32,
        width=32,
    )
    assert (rasterized == taxonomy.room_ids[0]).any()
    assert (rasterized == taxonomy.room_ids[1]).any()


def test_polygon_head_and_hungarian_loss_backpropagate() -> None:
    taxonomy = _taxonomy()
    labels = torch.zeros(2, 32, 32, dtype=torch.long)
    labels[:, 4:16, 3:15] = taxonomy.room_ids[0]
    labels[:, 18:28, 17:29] = taxonomy.room_ids[1]
    classes, polygons, valid = polygon_instance_targets(
        labels,
        taxonomy,
        max_instances=6,
        points=12,
    )
    head = PolygonLoopHead(
        12,
        len(taxonomy.room_ids),
        hidden_dim=24,
        num_queries=6,
        points=12,
        layers=1,
    )
    predictions = head(torch.randn(2, 12, 32, 32))
    assert predictions["class_logits"].shape == (2, 6, 2)
    assert predictions["polygons"].shape == (2, 6, 12, 2)
    loss, components = polygon_loop_loss(
        predictions,
        classes,
        polygons,
        valid,
    )
    assert torch.isfinite(loss)
    assert set(components) == {
        "room_polygon_loss",
        "room_polygon_class_loss",
        "room_polygon_corner_loss",
        "room_polygon_chamfer_loss",
        "room_polygon_mask_dice_loss",
        "room_polygon_valid_loss",
        "room_polygon_point_valid_loss",
    }
    loss.backward()
    assert head.corner_head.weight.grad is not None
    metrics = room_query_metrics(
        predictions,
        labels,
        taxonomy,
        validity_threshold=-100.0,
    )
    assert metrics["count_abs_error"] >= 0.0
    assert 0.0 <= metrics["instance_f1_at_threshold"] <= 1.0
    decoded = decode_polygon_loops(
        predictions,
        taxonomy,
        height=32,
        width=32,
        validity_threshold=-100.0,
    )
    assert decoded.shape == (2, 32, 32)
    assert torch.isin(
        decoded,
        torch.tensor(taxonomy.room_ids),
    ).any()


@pytest.mark.parametrize(
    ("rectangles", "expected_count"),
    [
        ([], 0),
        ([(4, 3, 15, 14)], 1),
        ([(3, 3, 12, 12), (18, 18, 29, 29)], 2),
    ],
)
def test_room_query_metrics_cover_empty_single_and_multiroom(
    rectangles,
    expected_count,
) -> None:
    taxonomy = _taxonomy()
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    for index, (y0, x0, y1, x1) in enumerate(rectangles):
        labels[0, y0:y1, x0:x1] = taxonomy.room_ids[index % 2]
    classes, polygons, valid = polygon_instance_targets(
        labels,
        taxonomy,
        max_instances=3,
        points=12,
    )
    class_logits = torch.full((1, 3, len(taxonomy.room_ids)), -10.0)
    class_logits.scatter_(2, classes[..., None], 10.0)
    validity_logits = torch.where(valid, 10.0, -10.0)
    predictions = {
        "class_logits": class_logits,
        "polygons": polygons,
        "valid_logits": validity_logits,
    }
    metrics = room_query_metrics(predictions, labels, taxonomy)
    assert metrics["count_abs_error"] == 0.0
    assert metrics["instance_f1_at_threshold"] == 1.0
    assert metrics["matched_type_accuracy"] == 1.0
    assert int(valid.sum()) == expected_count


def test_polygon_loss_is_monotone_under_boundary_perturbation() -> None:
    taxonomy = _taxonomy()
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[:, 4:24, 4:24] = taxonomy.room_ids[0]
    classes, polygons, valid = polygon_instance_targets(
        labels,
        taxonomy,
        max_instances=2,
        points=16,
    )
    class_logits = torch.full((1, 2, len(taxonomy.room_ids)), -10.0)
    class_logits.scatter_(2, classes[..., None], 10.0)
    valid_logits = torch.where(valid, 10.0, -10.0)
    perfect = {
        "class_logits": class_logits,
        "polygons": polygons,
        "valid_logits": valid_logits,
    }
    perturbed_polygons = polygons.clone()
    perturbed_polygons[..., 0, :] += 0.05
    perturbed = {
        "class_logits": class_logits,
        "polygons": perturbed_polygons,
        "valid_logits": valid_logits,
    }
    perfect_loss, _ = polygon_loop_loss(
        perfect,
        classes,
        polygons,
        valid,
    )
    perturbed_loss, _ = polygon_loop_loss(
        perturbed,
        classes,
        polygons,
        valid,
    )
    assert perturbed_loss > perfect_loss
