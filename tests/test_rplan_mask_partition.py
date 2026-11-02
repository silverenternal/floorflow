import torch

from floorflow.data.rplan_boundary_adapter import rplan_taxonomy
from floorflow.evaluation.rplan_mask_partition import (
    mask_partition_labels,
    mask_partition_query_metrics,
)


def test_mask_partition_labels_are_exclusive_and_confidence_bounded() -> None:
    probabilities = torch.full((2, 8, 8), 0.1)
    probabilities[0, :, :4] = 0.9
    probabilities[1, :, 4:] = 0.9
    footprint = torch.ones(8, 8, dtype=torch.bool)
    labels = mask_partition_labels(
        probabilities,
        torch.tensor([4, 5]),
        footprint,
    )
    assert labels[:, :4].eq(4).all()
    assert labels[:, 4:].eq(5).all()
    low_confidence = mask_partition_labels(
        probabilities * 0.1,
        torch.tensor([4, 5]),
        footprint,
    )
    assert low_confidence.eq(0).all()


def test_mask_partition_query_metrics_are_perfect_for_separable_rooms() -> None:
    taxonomy = rplan_taxonomy()
    mask_logits = torch.full((1, 4, 8, 8), -8.0)
    mask_logits[0, 0, :, :4] = 8.0
    mask_logits[0, 1, :, 4:] = 8.0
    predictions = {
        "room_instance_mask_logits": mask_logits,
        "room_count_logits": torch.tensor([[0.0, 0.0, 8.0, 0.0, 0.0]]),
        "room_exists": torch.tensor([[8.0, 7.0, -8.0, -8.0]]),
        "room_class": torch.tensor(
            [
                [
                    [8.0, -8.0],
                    [-8.0, 8.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        ),
    }
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :, :4] = taxonomy.room_ids[0]
    target[:, :, 4:] = taxonomy.room_ids[1]
    metrics = mask_partition_query_metrics(
        predictions,
        target,
        torch.ones(1, 1, 8, 8),
        taxonomy,
    )
    assert metrics["matched_geometric_iou"] == 1.0
    assert metrics["instance_f1_at_threshold"] == 1.0
    assert metrics["matched_type_accuracy"] == 1.0
    assert metrics["count_abs_error"] == 0.0
