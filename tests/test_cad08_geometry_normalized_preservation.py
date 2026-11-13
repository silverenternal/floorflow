"""Strict CAD preservation under lossless DXF geometry ordering changes."""

from __future__ import annotations

import copy

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon, box, mapping, shape

from floorflow.evaluation.cad08_metrics import effect, geometry_normalized_record_key, vector_preservation
from floorflow.representation.cad08_state import LayeredCad08State, object_record, rle_encode


def _state_pair() -> tuple[LayeredCad08State, LayeredCad08State, dict]:
    host = {
        "id": "host-1",
        "geometry": mapping(box(0, 14, 32, 18)),
        "start": [0.0, 16.0],
        "end": [32.0, 16.0],
        "thickness": 4.0,
        "source": "frozen-host",
    }
    before = LayeredCad08State.from_dict(
        {
            "schema": "floorflow-layered-cad08-v1",
            "frame": [32, 32],
            "namespace": "preservation-regression",
            "footprint": mapping(box(0, 0, 32, 32)),
            "objects": [
                object_record("preservation-regression", "rooms", 2, box(0, 0, 16, 32), identity="room-1"),
                object_record("preservation-regression", "rooms", 3, box(16, 0, 32, 32), identity="room-2"),
                object_record("preservation-regression", "wall_material", 1, box(0, 14, 32, 18), identity="wall-1"),
                object_record("preservation-regression", "openings", 4, box(6, 14, 10, 18), identity="opening-1"),
                object_record("preservation-regression", "openings", 4, box(20, 14, 24, 18), identity="opening-2"),
            ],
            "wall_hosts": [host],
            "relationships": {},
            "lineage": [],
            "diagnostic_rasters": {},
            "provenance": {"room_owner_predictions_available": False},
        }
    )
    before.refresh_relationships()
    before.document["visible_reference"] = rle_encode(before.render()["prediction"])

    after = LayeredCad08State.from_dict(before.to_dict())
    room = next(item for item in after.objects("rooms") if item["id"] == "room-1")
    room["label"] = 5
    after.document["lineage"].append(
        {"operation": "room_type", "parents": ["room-1"], "children": ["room-1"]}
    )
    after.refresh_relationships()
    after.document["visible_reference"] = rle_encode(after.render()["prediction"])

    source_mask = np.zeros((32, 32), dtype=bool)
    source_mask[:, :16] = True
    action = {
        "index": 0,
        "operation": "room_type",
        "source_mask": source_mask,
        "split_mask": np.zeros_like(source_mask),
        "destination_mask": np.zeros_like(source_mask),
        "source_class_id": 2,
        "target_class_id": 5,
        "destination_ambiguous": False,
    }
    return before, after, action


def _p(before: LayeredCad08State, after: LayeredCad08State, action: dict) -> dict:
    return vector_preservation(before, after, action, effect(before, after, action))


def _rotate_reverse_polygon(geometry: dict) -> dict:
    polygon = shape(geometry)
    ring = list(polygon.exterior.coords[:-1])
    reordered = ring[2:] + ring[:2]
    reordered.reverse()
    reordered.append(reordered[0])
    return mapping(Polygon(reordered, [list(interior.coords) for interior in polygon.interiors]))


def test_dxf_ring_reordering_passes_normalized_preservation_without_changing_coordinates():
    before, after, action = _state_pair()
    room = next(item for item in after.objects("rooms") if item["id"] == "room-2")
    original_geometry = room["geometry"]
    reordered_geometry = _rotate_reverse_polygon(original_geometry)
    assert shape(original_geometry).equals(shape(reordered_geometry))
    room["geometry"] = reordered_geometry

    result = _p(before, after, action)

    assert result["status"] == "pass"
    assert result["unselected_geometry_and_class_exact"] is True
    assert result["unselected_object_records_geometry_normalized_exact"] is True
    assert result["unselected_object_records_byte_exact"] is False
    assert geometry_normalized_record_key(
        {"id": "x", "family": "rooms", "label": 3, "geometry": original_geometry}
    ) == geometry_normalized_record_key(
        {"id": "x", "family": "rooms", "label": 3, "geometry": reordered_geometry}
    )


def test_strict_preservation_rejects_geometry_host_relation_id_and_unselected_record_changes():
    before, after, action = _state_pair()
    shifted = copy.deepcopy(after)
    room = next(item for item in shifted.objects("rooms") if item["id"] == "room-2")
    room["geometry"] = mapping(translate(shape(room["geometry"]), xoff=0.01))
    assert _p(before, shifted, action)["status"] == "fail"

    before, after, action = _state_pair()
    missing_host = copy.deepcopy(after)
    missing_host.document["wall_hosts"] = []
    assert _p(before, missing_host, action)["status"] == "fail"

    before, after, action = _state_pair()
    changed_relation = copy.deepcopy(after)
    changed_relation.document["relationships"]["opening_host"]["opening-2"]["host_id"] = "missing-host"
    relation_result = _p(before, changed_relation, action)
    assert relation_result["status"] == "fail"
    assert relation_result["unselected_opening_host_relationships_exact"] is False

    before, after, action = _state_pair()
    renamed = copy.deepcopy(after)
    next(item for item in renamed.objects("rooms") if item["id"] == "room-2")["id"] = "room-renamed"
    assert _p(before, renamed, action)["status"] == "fail"

    before, after, action = _state_pair()
    changed_record = copy.deepcopy(after)
    next(item for item in changed_record.objects("rooms") if item["id"] == "room-2")["identity_source"] = "mutated"
    assert _p(before, changed_record, action)["status"] == "fail"
