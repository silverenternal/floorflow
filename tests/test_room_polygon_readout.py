from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.room_polygon_readout import vectorize_room_masks


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=("background", "wall", "room:LivingRoom", "room:Bedroom"),
        room_map={
            "LivingRoom": "room:LivingRoom",
            "Bedroom": "room:Bedroom",
        },
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def test_vectorized_room_masks_restore_seams_and_exterior() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((32, 32), dtype=np.int64)
    labels[3:29, 3:16] = taxonomy.class_id("room:LivingRoom")
    labels[3:29, 16:29] = taxonomy.class_id("room:Bedroom")
    footprint = np.zeros_like(labels)
    footprint[2:30, 2:30] = 1
    output = vectorize_room_masks(
        labels,
        footprint,
        taxonomy,
        epsilon=1.5,
    )
    assert output.shape == labels.shape
    assert np.any(output == taxonomy.class_id("room:LivingRoom"))
    assert np.any(output == taxonomy.class_id("room:Bedroom"))
    assert np.any(output == taxonomy.class_id("wall"))
    assert np.all(output[footprint == 0] == 0)
