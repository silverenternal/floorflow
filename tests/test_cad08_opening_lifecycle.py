"""Focused CAD08 opening add/replace transaction contracts."""

from shapely.geometry import box, shape

from floorflow.inference.cad08_transaction import apply_transaction
from floorflow.representation.cad08_state import (
    LayeredCad08State,
    encoded_geometry,
    mask_geometry,
    object_record,
    rle_encode,
)


def state_with_registered_host() -> LayeredCad08State:
    namespace = "opening-lifecycle-test"
    frame = (100, 100)
    wall = box(10, 48, 90, 52)
    state = LayeredCad08State(
        {
            "schema": "floorflow-layered-cad08-v1",
            "frame": list(frame),
            "namespace": namespace,
            "footprint": encoded_geometry(box(0, 0, 100, 100)),
            "objects": [
                object_record(namespace, "rooms", 2, box(10, 10, 90, 48), identity="room:a"),
                object_record(namespace, "wall_material", 1, wall, identity="wall:main"),
            ],
            "wall_hosts": [
                {
                    "id": "host:main",
                    "geometry": encoded_geometry(wall),
                    "start": [10.0, 50.0],
                    "end": [90.0, 50.0],
                    "thickness": 4.0,
                    "source": "test_fixture_registered_host",
                }
            ],
            "room_aliases": {},
            "relationships": {},
            "lineage": [],
            "visible_reference": {},
            "diagnostic_rasters": {},
            "provenance": {
                "decoded_fields_unchanged": True,
                "minimum_export_component_area": 1,
                "small_components_not_merged_or_filled": True,
                "hidden_plane_overlap_preserved": True,
                "room_owner_predictions_available": False,
            },
        }
    )
    state.refresh_relationships()
    state.document["visible_reference"] = rle_encode(state.render()["prediction"])
    return state


def add_action(geometry, opening_id="opening:main", class_id=4):
    return {
        "index": 1,
        "operation": "opening_add",
        "host_id": "host:main",
        "opening_id": opening_id,
        "opening_class_id": class_id,
        "opening_geometry": geometry.__geo_interface__,
    }


def test_opening_add_and_same_host_replace_preserve_identity_and_scope():
    before = state_with_registered_host()
    added, add_trace = apply_transaction(before, add_action(box(30, 48, 40, 52)))

    assert add_trace["accepted"]
    assert add_trace["binding"]["binding_uses_after"] is False
    assert before.objects("openings") == []
    assert len(added.objects("openings")) == 1
    assert added.objects("openings")[0]["id"] == "opening:main"
    assert added.document["relationships"]["opening_host"]["opening:main"]["host_id"] == "host:main"
    assert added.render()["wall"][49, 35] == 0

    replaced, replace_trace = apply_transaction(
        added,
        {
            "index": 2,
            "operation": "opening_replace",
            "host_id": "host:main",
            "source_opening_id": "opening:main",
            "opening_class_id": 5,
            "opening_geometry": box(60, 48, 70, 52).__geo_interface__,
        },
    )

    assert replace_trace["accepted"]
    assert replace_trace["selected_before_ids"] == ["opening:main"]
    assert replace_trace["selected_after_ids"] == ["opening:main"]
    assert len(replaced.objects("openings")) == 1
    assert replaced.objects("openings")[0]["id"] == "opening:main"
    assert replaced.objects("openings")[0]["label"] == 5
    assert replaced.render()["wall"][49, 35] == 1
    assert replaced.render()["wall"][49, 65] == 0
    assert shape(replaced.objects("wall_material")[0]["geometry"]).is_valid
    assert before.objects("openings") == []
    assert mask_geometry(replaced.render()["footprint"]).area == 10000


def test_opening_add_fails_closed_for_unregistered_host_or_geometry_outside_host():
    before = state_with_registered_host()
    no_host, no_host_trace = apply_transaction(
        before,
        {**add_action(box(30, 48, 40, 52)), "host_id": "host:missing"},
    )
    outside, outside_trace = apply_transaction(before, add_action(box(30, 60, 40, 64)))

    assert not no_host_trace["accepted"]
    assert no_host_trace["reasons"] == ["input_host_id_not_unique"]
    assert not outside_trace["accepted"]
    assert outside_trace["reasons"] == ["opening_geometry_not_covered_by_selected_host"]
    assert no_host.to_dict() == before.to_dict()
    assert outside.to_dict() == before.to_dict()
