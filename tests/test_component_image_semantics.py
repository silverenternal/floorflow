"""Tests for structure-separated native room components."""

import numpy as np

from floorflow.heads.component_image_semantics import (
    native_component_records,
)
from floorflow.heads.component_semantic import assign_component_classes


def test_structure_mask_separates_same_class_room_support() -> None:
    room_labels = np.full((8, 8), 10, dtype=np.int64)
    target = np.full((8, 8), 10, dtype=np.int64)
    target[:, 4:] = 11
    footprint = np.ones((8, 8), dtype=bool)
    probability = np.full((2, 8, 8), 0.5, dtype=np.float32)
    image = np.zeros((3, 8, 8), dtype=np.float32)
    structure = np.zeros((8, 8), dtype=bool)
    structure[:, 4] = True

    joined, _targets, _ids = native_component_records(
        room_labels,
        target,
        probability,
        image,
        footprint,
        (10, 11),
    )
    separated, separated_targets, separated_ids = native_component_records(
        room_labels,
        target,
        probability,
        image,
        footprint,
        (10, 11),
        structure_mask=structure,
    )

    assert len(joined) == 1
    assert len(separated) == 2
    assert separated_ids.tolist() == [1, 2]
    assert set(separated_targets.tolist()) == {0, 1}


def test_component_assignment_clears_structural_gap_pixels() -> None:
    room_labels = np.full((4, 6), 10, dtype=np.int64)
    structure = np.zeros((4, 6), dtype=bool)
    structure[:, 3] = True
    class_probability = np.asarray([[0.9, 0.1], [0.1, 0.9]])

    assigned = assign_component_classes(
        room_labels,
        class_probability,
        np.asarray([1, 2]),
        (10, 11),
        structure_mask=structure,
    )

    assert not assigned[:, 3].any()
    assert (assigned[:, :3] == 10).all()
    assert (assigned[:, 4:] == 11).all()
