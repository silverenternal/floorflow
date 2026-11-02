"""Tests for typed non-uniform edit routing."""

from __future__ import annotations

import numpy as np

from floorflow.inference.typed_edit_graph import (
    build_typed_edit_graph,
    compose_typed_edit,
    routing_mask,
)


def _inputs():
    before = np.zeros((24, 24), dtype=np.int16)
    after = before.copy()
    before_rooms = before.copy()
    after_rooms = after.copy()
    before_topology = before.copy()
    after_topology = after.copy()
    footprint = np.ones_like(before, dtype=bool)
    edit_region = np.zeros_like(footprint)
    changed = np.zeros_like(footprint)
    edit_region[6:15, 6:15] = True
    changed[8:12, 8:12] = True
    before[6:15, 6:15] = 1
    after[4:17, 4:17] = 2
    before_rooms[6:15, 6:15] = 1
    after_rooms[6:15, 6:15] = 2
    before[6:15, 15] = 3
    after[6:15, 15] = 3
    before_topology[:] = before
    after_topology[:] = after
    return {
        "before": before,
        "after": after,
        "before_rooms": before_rooms,
        "after_rooms": after_rooms,
        "before_topology": before_topology,
        "after_topology": after_topology,
        "footprint": footprint,
        "edit_region": edit_region,
        "changed": changed,
    }


def _graph(inputs):
    return build_typed_edit_graph(
        inputs["before"],
        inputs["edit_region"],
        inputs["changed"],
        after_labels=inputs["after"],
        room_ids=(1, 2),
        wall_id=3,
        opening_ids=(4,),
        furniture_ids=(5,),
    )


def test_typed_routing_is_nonuniform_and_suppresses_weak_region() -> None:
    inputs = _inputs()
    graph = _graph(inputs)
    typed = routing_mask(
        graph,
        inputs["edit_region"],
        inputs["footprint"],
        policy="typed_edit_graph",
    )
    dense = routing_mask(
        graph,
        inputs["edit_region"],
        inputs["footprint"],
        policy="dense_cross_attention",
    )

    assert typed[8:12, 8:12].all()
    assert not typed[0, 0]
    assert not graph.weak_connections[0, 0]
    assert dense[4:6, 6:15].all()
    assert int(dense.sum()) > int(inputs["edit_region"].sum())
    assert graph.changed_families == ("room",)


def test_typed_graph_uses_added_wall_family_from_after_labels() -> None:
    before = np.ones((12, 12), dtype=np.int16)
    after = before.copy()
    after[4:8, 4:8] = 3
    region = np.zeros_like(before, dtype=bool)
    region[3:9, 3:9] = True
    changed = after != before
    graph = build_typed_edit_graph(
        before,
        region,
        changed,
        after_labels=after,
        room_ids=(1,),
        wall_id=3,
        opening_ids=(4,),
        furniture_ids=(5,),
    )
    assert graph.changed_families == ("room", "wall")
    assert graph.adjacent_walls.any()


def test_typed_composition_preserves_outside_region() -> None:
    inputs = _inputs()
    graph = _graph(inputs)
    composition = compose_typed_edit(
        inputs["before"],
        inputs["after"],
        inputs["before_rooms"],
        inputs["after_rooms"],
        inputs["before_topology"],
        inputs["after_topology"],
        graph,
        inputs["edit_region"],
        inputs["footprint"],
        policy="typed_edit_graph",
    )

    outside = inputs["footprint"] & ~inputs["edit_region"]
    assert np.array_equal(composition.visible[outside], inputs["before"][outside])
    assert composition.outside_region_changes == 0


def test_dense_and_global_policies_expose_their_contract_boundaries() -> None:
    inputs = _inputs()
    graph = _graph(inputs)
    dense = compose_typed_edit(
        inputs["before"],
        inputs["after"],
        inputs["before_rooms"],
        inputs["after_rooms"],
        inputs["before_topology"],
        inputs["after_topology"],
        graph,
        inputs["edit_region"],
        inputs["footprint"],
        policy="dense_cross_attention",
    )
    global_edit = compose_typed_edit(
        inputs["before"],
        inputs["after"],
        inputs["before_rooms"],
        inputs["after_rooms"],
        inputs["before_topology"],
        inputs["after_topology"],
        graph,
        inputs["edit_region"],
        inputs["footprint"],
        policy="no_edit_gating",
    )

    assert dense.outside_region_changes > 0
    assert global_edit.outside_region_changes > dense.outside_region_changes
