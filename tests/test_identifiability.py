from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.identifiability import (
    ContourRecord,
    contour_descriptor,
    duplicate_contour_layout_ious,
    entropy,
    macro_iou,
    nearest_indices,
    program_neighbor_entropy,
)


def test_contour_descriptor_is_shape_normalized() -> None:
    footprint = torch.zeros(1, 32, 32)
    footprint[:, 4:28, 6:26] = 1.0
    descriptor = contour_descriptor(footprint, size=8)
    assert descriptor.shape == (65,)
    assert 0.0 < float(descriptor[-1]) < 1.0


def test_entropy_and_nearest_indices() -> None:
    assert entropy(["a", "a"]) == 0.0
    assert entropy(["a", "b"]) > 0.69
    bank = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.1, 0.1]])
    queries = torch.tensor([[0.05, 0.05]])
    indices, distances = nearest_indices(queries, bank, k=1)
    assert indices.tolist() == [[0]]
    assert distances.shape == (1, 1)


def test_program_entropy_matches_neighbor_uncertainty() -> None:
    train = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    test = torch.tensor([[1.0, 0.0]])
    neighbors = torch.tensor([[0, 1]])
    values = program_neighbor_entropy(test, train, neighbors)
    assert values.shape == (1,)
    assert abs(float(values[0]) - 1.0) < 1.0e-5


def test_macro_iou_uses_present_room_classes() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    room = taxonomy.room_ids[0]
    target = torch.zeros(8, 8, dtype=torch.long)
    target[:4] = room
    prediction = target.clone()
    assert macro_iou(prediction, target, taxonomy) == 1.0


def test_duplicate_contours_capture_target_ambiguity() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    room = taxonomy.room_ids[0]
    labels_a = torch.zeros(8, 8, dtype=torch.long)
    labels_a[:, :4] = room
    labels_b = torch.zeros(8, 8, dtype=torch.long)
    labels_b[:, 4:] = room
    records = [
        ContourRecord(
            name=f"case-{index}",
            exact_hash=7,
            descriptor=torch.zeros(4),
            room_labels=labels,
            room_program=torch.zeros(1),
        )
        for index, labels in enumerate((labels_a, labels_b))
    ]
    result = duplicate_contour_layout_ious(records, taxonomy)
    assert result["duplicate_pairs"] == 1
    assert result["fraction_below_095"] == 1.0
