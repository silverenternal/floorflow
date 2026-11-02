from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import (
    ReadoutWall,
    raster_labels_to_wall_plan,
)


def test_opening_prefers_long_host_wall_over_closer_fragment() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    window_id = next(
        class_id
        for class_id in taxonomy.opening_ids
        if taxonomy.class_names[class_id].startswith("window:")
    )
    labels = np.zeros((64, 64), dtype=np.int64)
    labels[20, 10:30] = taxonomy.class_id("wall")
    labels[20, 18:25] = window_id

    original = raster_labels_to_wall_plan
    plan = original(labels, taxonomy)
    assert plan.openings
    host = plan.walls[plan.openings[0].wall_index]
    length = np.linalg.norm(
        np.asarray(host.end) - np.asarray(host.start)
    )
    assert length >= 0.8 * plan.openings[0].width
    assert isinstance(ReadoutWall(host.start, host.end, 1.0), ReadoutWall)
