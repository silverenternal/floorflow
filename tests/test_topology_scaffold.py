from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.topology_scaffold import (
    compose_topology_candidate,
    select_guarded_topology,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=("background", "wall", "door:Swing Beside", "room:Kitchen"),
        room_map={},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={},
    )


def test_compose_topology_candidate_is_exact_outside_region() -> None:
    before = np.zeros((8, 8), dtype=np.uint8)
    after = np.ones((8, 8), dtype=np.uint8)
    region = np.zeros((8, 8), dtype=bool)
    region[2:6, 2:6] = True

    candidate = compose_topology_candidate(before, after, region)

    assert np.all(candidate[region] == 1)
    assert np.all(candidate[~region] == 0)


def test_guard_prefers_local_then_falls_back() -> None:
    taxonomy = _taxonomy()
    before = np.zeros((8, 8), dtype=np.uint8)
    after = np.ones((8, 8), dtype=np.uint8)
    region = np.zeros((8, 8), dtype=bool)
    region[2:6, 2:6] = True

    def local_pass(labels: np.ndarray, _taxonomy: FloorplanTaxonomy) -> dict:
        return {"valid": True, "watertight": bool(labels[0, 0] == 0)}

    guarded, detail = select_guarded_topology(
        before,
        after,
        region,
        taxonomy,
        verifier=local_pass,
    )
    assert detail["source"] == "local_delta"
    assert np.array_equal(guarded, compose_topology_candidate(before, after, region))

    def local_fail(labels: np.ndarray, _taxonomy: FloorplanTaxonomy) -> dict:
        return {"valid": True, "watertight": bool(labels[0, 0] == 1)}

    guarded, detail = select_guarded_topology(
        before,
        after,
        region,
        taxonomy,
        verifier=local_fail,
    )
    assert detail["source"] == "post_edit_fallback"
    assert np.array_equal(guarded, after)
