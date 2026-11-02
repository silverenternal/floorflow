from collections import Counter

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.structure_compositor import owner_cell_plan
from floorflow.representation.plan_canonical_adapter import (
    adaptive_coordinate_quantizer,
    canonical_token_roundtrip,
    raster_plan_to_canonical,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter(),
            "door": Counter({"Door": 2}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_plan_canonical_adapter_preserves_topology_and_tokens() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((24, 24), dtype=np.int64)
    labels[:, :11] = taxonomy.room_ids[0]
    labels[:, 13:] = taxonomy.room_ids[1]
    labels[:, 11:13] = taxonomy.class_id("wall")
    labels[8:12, 11:13] = taxonomy.opening_ids[0]
    footprint = np.ones((24, 24), dtype=bool)
    raster_plan = owner_cell_plan(
        labels,
        footprint,
        taxonomy,
        minimum_area=4,
    )
    canonical, error = raster_plan_to_canonical(
        raster_plan,
        footprint,
        taxonomy,
        case_id="case",
        track="cubicasa_reconstruction",
    )
    assert error <= 0.25
    assert len(canonical.rooms) == len(raster_plan.rooms)
    assert len(canonical.walls) == len(raster_plan.walls)
    assert len(canonical.openings) == len(raster_plan.openings)
    assert all(
        opening.wall_id in {wall.id for wall in canonical.walls}
        for opening in canonical.openings
    )
    assert canonical.junctions
    tokenized, metrics = canonical_token_roundtrip(canonical, taxonomy)
    assert tokenized.token_count > 0
    assert metrics["canonical_fingerprint_exact"]
    assert metrics["canonical_room_count_exact"]
    assert metrics["canonical_wall_count_exact"]
    assert metrics["canonical_opening_count_exact"]
    assert metrics["canonical_prefix_violation_count"] == 0
    assert adaptive_coordinate_quantizer((192, 192)).scale == 8
    assert adaptive_coordinate_quantizer((384, 384)).scale == 16
