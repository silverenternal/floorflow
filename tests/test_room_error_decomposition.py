import torch
from collections import Counter

from floorflow.evaluation.room_error_decomposition import (
    aggregate_decomposition,
    decompose_rooms,
    room_confusion,
)
from floorflow.data.taxonomy import build_taxonomy


def _taxonomy_and_ids():
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Sink": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    return taxonomy, taxonomy.room_ids[0], taxonomy.room_ids[1]


def test_room_confusion_ignores_non_room_labels() -> None:
    taxonomy, room_a, room_b = _taxonomy_and_ids()
    target = torch.tensor([[room_a, room_a, 1]])
    prediction = torch.tensor([[room_a, room_b, 1]])
    confusion = room_confusion(prediction, target, taxonomy)

    assert confusion.shape == (len(taxonomy.room_ids) + 1,) * 2
    assert int(confusion[1, 1]) == 1
    assert int(confusion[1, 2]) == 1
    assert int(confusion[0, 0]) == 1


def test_decomposition_identifies_missing_mismatch_and_correct_pixels() -> None:
    taxonomy, room_a, room_b = _taxonomy_and_ids()
    target = torch.tensor(
        [[
            [room_a, room_a, room_a],
            [room_b, room_b, 0],
            [0, 0, 0],
        ]]
    )
    prediction = torch.tensor(
        [[
            [room_a, room_a, room_a],
            [room_a, 0, 0],
            [0, 0, 0],
        ]]
    )
    footprint = torch.ones(1, 3, 3)
    result = decompose_rooms(prediction, target, footprint, taxonomy)

    assert result["strict_macro_iou"] == 0.375
    assert result["allocation"]["correct_identity_fraction"] == 0.6
    assert result["allocation"]["missing_room_fraction"] == 0.2
    assert result["allocation"]["semantic_mismatch_fraction"] == 0.2
    assert result["missing_class_count"] == 1
    assert result["class_rows"][1]["dominant_prediction_fraction"] == 0.5


def test_aggregate_is_case_average_and_does_not_select_cases() -> None:
    taxonomy, room_a, _room_b = _taxonomy_and_ids()
    footprint = torch.ones(1, 2, 2)
    perfect = decompose_rooms(
        torch.full((1, 2, 2), room_a),
        torch.full((1, 2, 2), room_a),
        footprint,
        taxonomy,
    )
    zero = decompose_rooms(
        torch.zeros(1, 2, 2, dtype=torch.long),
        torch.full((1, 2, 2), room_a),
        footprint,
        taxonomy,
    )
    report = aggregate_decomposition([perfect, zero])

    assert report["strict_macro_iou"] == 0.5
    assert report["occupancy"]["iou"] == 0.5
    assert report["allocation"]["missing_room_fraction"] == 0.5
