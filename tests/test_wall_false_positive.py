import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.wall_false_positive import (
    wall_false_positive_decomposition,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "room:Bedroom",
            "room:LivingRoom",
        ),
        room_map={},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={},
    )


def test_wall_false_positives_are_partitioned_by_provenance() -> None:
    target = np.zeros((20, 20), dtype=np.int64)
    target[:, :10] = 2
    target[:, 10:] = 3
    target[2:18, 3] = 1
    prediction = target.copy()
    prediction[8, 2] = 1
    prediction[0, 15] = 1
    prediction[8, 10] = 1
    prediction[9, 14] = 1
    result = wall_false_positive_decomposition(
        prediction,
        target,
        np.ones_like(target),
        _taxonomy(),
    )
    assert result["wall_false_positive_pixels"] == 4
    assert result["wall_target_extension_rate"] > 0
    assert result["wall_exterior_duplication_rate"] > 0
    assert result["wall_room_boundary_fragment_rate"] > 0
    assert result["wall_unsupported_free_rate"] > 0


def test_wall_false_positive_decomposition_is_noop_for_perfect_walls() -> None:
    target = np.zeros((8, 8), dtype=np.int64)
    target[:, 3] = 1
    result = wall_false_positive_decomposition(
        target,
        target,
        np.ones_like(target),
        _taxonomy(),
    )
    assert result["wall_false_positive_pixels"] == 0
    assert result["wall_false_negative_pixels"] == 0
    assert result["wall_true_positive_pixels"] == 8
