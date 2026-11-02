from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.delta_mse_decomposition import (
    decompose_delta_error,
    decomposition_masks,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "door:Swing Beside",
            "room:Kitchen",
            "room:LivingRoom",
        ),
        room_map={},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={},
    )


def test_decomposition_masks_are_exclusive_inside_edit_region() -> None:
    taxonomy = _taxonomy()
    before = np.zeros((16, 16), dtype=np.uint8)
    before[2:14, 2:14] = taxonomy.class_id("room:Kitchen")
    before[8, 2:14] = taxonomy.class_id("wall")
    after = before.copy()
    after[8, 8] = taxonomy.class_id("door:Swing Beside")
    valid = np.ones_like(before, dtype=bool)
    edit = np.zeros_like(valid)
    edit[4:12, 4:12] = True

    masks = decomposition_masks(before, after, valid, edit, taxonomy)
    exclusive = (
        masks["structure_boundary"].astype(np.int8)
        + masks["room_boundary"].astype(np.int8)
        + masks["edit_interior"].astype(np.int8)
    )
    assert exclusive.max() == 1
    assert np.all(masks["edit_region"] == (edit & valid))


def test_decomposition_reports_error_mass() -> None:
    taxonomy = _taxonomy()
    before = np.zeros((8, 8), dtype=np.uint8)
    before[2:6, 2:6] = taxonomy.class_id("room:Kitchen")
    after = before.copy()
    after[4, 4] = taxonomy.class_id("room:LivingRoom")
    prediction_before = before.copy()
    prediction_after = prediction_before.copy()
    valid = np.ones_like(before, dtype=bool)
    edit = np.zeros_like(valid)
    edit[3:6, 3:6] = True

    result = decompose_delta_error(
        before,
        after,
        prediction_before,
        prediction_after,
        valid,
        edit,
        taxonomy,
    )

    assert np.isclose(result["edit_region"]["mse"], 1.0 / 9.0)
    assert result["outside"]["mse"] == 0.0
    assert result["edit_region"]["error_mass_fraction_of_edit"] == 1.0
