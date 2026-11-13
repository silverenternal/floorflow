"""CPU analytical checks for the sole room_add permission correction."""

import copy

import numpy as np
from shapely.geometry import box, shape

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation import cad08_state_metrics_v1
from floorflow.evaluation.cad08_metrics import effect, geometry_validity, vector_preservation
from floorflow.inference.cad08_adapter import compose_visible
from floorflow.inference.cad08_transaction import apply_transaction
from floorflow.inference.cad08_transaction_permission_v1 import apply_permission_transaction
from floorflow.representation.cad08_state import (
    LayeredCad08State,
    host_rows,
    object_record,
    raster_geometry,
    rle_encode,
)


def setup_case():
    taxonomy = FloorplanTaxonomy(
        ["background", "wall", "room:A", "room:B", "door:Door", "furniture:Sink"], {}, {}, {}, {}, {}
    )
    frame = (48, 48)
    zero = np.zeros(frame, np.uint8)
    values = {
        "footprint": np.ones(frame, np.uint8),
        "room_labels": np.full(frame, 2, np.uint8),
        "wall": zero.copy(),
        "opening": zero.copy(),
        "furniture": zero.copy(),
    }
    values["wall"][:2] = 1
    values["prediction"] = compose_visible(values)
    before = LayeredCad08State.from_decoded(values, taxonomy, "permission_analytic")
    split = np.zeros(frame, bool)
    split[:, 24:] = True
    action = {
        "index": 1,
        "operation": "room_add",
        "source_mask": np.ones(frame, bool),
        "split_mask": split,
        "destination_mask": np.zeros(frame, bool),
        "source_class_id": 2,
        "target_class_id": 3,
        "destination_ambiguous": False,
    }
    return before, action, values


def test_same_proposal_reproduces_root_permission_counterexample():
    before, action, proposal = setup_case()
    proposal["room_labels"][:, 4:] = 3
    old, _ = apply_transaction(before, action, proposal=proposal)
    new, trace = apply_permission_transaction(before, action, proposal=proposal)
    a = before.render()["room_labels"]
    outside = ~action["split_mask"]
    assert np.count_nonzero((old.render()["room_labels"] != a) & outside) == 960
    assert np.count_nonzero((new.render()["room_labels"] != a) & outside) == 0
    assert effect(before, old, action)["E"] == "fail"
    assert effect(before, new, action)["E"] == "pass"
    assert trace["proposal_consumed"] and trace["accepted"]


def test_partial_proposal_is_not_painted_into_complete_requested_split():
    before, action, proposal = setup_case()
    proposal["room_labels"][:, 40:] = 3
    new, trace = apply_permission_transaction(before, action, proposal=proposal)
    assert action["split_mask"].sum() == 1152
    assert np.count_nonzero(new.render()["room_labels"] == 3) == 384
    assert effect(before, new, action)["E"] == "fail"
    assert trace["accepted"] and trace["proposal_consumed"]


def test_absent_target_proposal_cannot_construct_a_new_room():
    before, action, proposal = setup_case()
    new, trace = apply_permission_transaction(before, action, proposal=proposal)
    assert new.to_dict() == before.to_dict()
    assert not trace["accepted"] and trace["proposal_consumed"]
    assert effect(before, new, action)["E"] == "fail"


def test_proposal_only_outside_permission_cannot_construct_a_new_room():
    before, action, proposal = setup_case()
    proposal["room_labels"][:, :20] = 3
    new, trace = apply_permission_transaction(before, action, proposal=proposal)
    assert new.to_dict() == before.to_dict() and not trace["accepted"]
    assert effect(before, new, action)["E"] == "fail"


def test_program_only_request_keeps_original_unknown_and_noop():
    before, action, proposal = setup_case()
    action["split_mask"][:] = False
    proposal["room_labels"][:] = 3
    new, trace = apply_permission_transaction(before, action, proposal=proposal)
    assert new.to_dict() == before.to_dict() and not trace["proposal_consumed"]
    assert effect(before, new, action)["E"] == "unknown"


def test_subminimum_predicted_part_keeps_original_rejection():
    before, action, proposal = setup_case()
    proposal["room_labels"][10, 30:37] = 3
    new, trace = apply_permission_transaction(before, action, proposal=proposal)
    assert new.to_dict() == before.to_dict() and not trace["accepted"]
    assert effect(before, new, action)["E"] == "fail"


def test_disconnected_requested_parts_keep_original_unknown():
    before, action, proposal = setup_case()
    action["split_mask"][:] = False
    action["split_mask"][4:12, 28:36] = True
    action["split_mask"][28:36, 28:36] = True
    proposal["room_labels"][action["split_mask"]] = 3
    new, trace = apply_permission_transaction(before, action, proposal=proposal)
    assert trace["accepted"]
    child = next(v for v in new.objects("rooms") if v["label"] == 3)
    assert shape(child["geometry"]).geom_type == "MultiPolygon"
    assert effect(before, new, action)["E"] == "unknown"


def test_fragmented_partial_proposal_keeps_all_parts_and_fails_original_effect():
    before, action, proposal = setup_case()
    proposal["room_labels"][:, 26:28] = 3
    proposal["room_labels"][:, 40:42] = 3
    new, trace = apply_permission_transaction(before, action, proposal=proposal)
    child = next(v for v in new.objects("rooms") if v["label"] == 3)
    assert trace["accepted"] and shape(child["geometry"]).geom_type == "MultiPolygon"
    assert shape(child["geometry"]).area == 192
    assert effect(before, new, action)["E"] == "fail"


def test_too_small_requested_remainder_keeps_original_unknown():
    before, action, proposal = setup_case()
    action["split_mask"][:] = True
    action["split_mask"][0, :4] = False
    proposal["room_labels"][:] = 3
    new, trace = apply_permission_transaction(before, action, proposal=proposal)
    assert new.to_dict() == before.to_dict() and not trace["accepted"]
    assert effect(before, new, action)["E"] == "unknown"


def test_unselected_records_and_reference_dependency_closure_inherit():
    before, action, proposal = setup_case()
    source = before.objects("rooms")[0]
    source["geometry"] = object_record("a", "rooms", 2, box(0, 0, 40, 48))["geometry"]
    other = object_record("a", "rooms", 2, box(40, 0, 48, 48), identity="runtime:unselected_room")
    fixture = object_record("a", "fixtures", 5, box(42, 16, 46, 20), identity="runtime:unselected_fixture")
    opening = object_record("a", "openings", 4, box(42, 0, 46, 2), identity="runtime:unselected_opening")
    before.document["objects"].extend([other, fixture, opening])
    before.document["wall_hosts"] = host_rows(
        [{"start": [40, 1], "end": [48, 1], "thickness": 2}], np.ones((48, 48), bool), "a"
    )
    before.refresh_relationships()
    before.document["visible_reference"] = rle_encode(before.render()["prediction"])
    action["source_mask"][:, 40:] = False
    action["split_mask"][:, 40:] = False
    proposal["room_labels"][:, 24:40] = 3
    new, trace = apply_permission_transaction(before, action, proposal=proposal)
    old_records = {v["id"]: v for v in before.document["objects"]}
    new_records = {v["id"]: v for v in new.document["objects"]}
    for record in (other, fixture, opening):
        assert old_records[record["id"]] == new_records[record["id"]]
    for table, identity in (("fixture_room", fixture["id"]), ("opening_host", opening["id"])):
        assert before.document["relationships"][table][identity] == new.document["relationships"][table][identity]
    e = effect(before, new, action)
    row = {"operation": "room_add", "effect": e, "transaction": trace, "P": "pass"}
    g = cad08_state_metrics_v1.validity_state(new.to_dict(), geometry_validity(new), before.to_dict())
    p = cad08_state_metrics_v1.preservation_state(before.to_dict(), new.to_dict(), row, g, action["split_mask"])
    assert p["P_state"] == "pass"
    assert vector_preservation(before, new, action, e, trace)["unselected_object_records_byte_exact"]
    assert len(new.document["lineage"][-1]["children"]) == 2


def test_permission_cannot_reach_another_room_and_inputs_remain_immutable():
    before, action, proposal = setup_case()
    source = before.objects("rooms")[0]
    source["geometry"] = object_record("a", "rooms", 2, box(0, 0, 24, 48))["geometry"]
    before.document["objects"].append(object_record("a", "rooms", 2, box(24, 0, 48, 48)))
    before.refresh_relationships()
    action["source_mask"][:, 24:] = False
    action["split_mask"][:] = True
    proposal["room_labels"][:, 12:] = 3
    original = copy.deepcopy((before.to_dict(), action, proposal))
    new, _ = apply_permission_transaction(before, action, proposal=proposal)
    target = next(v for v in new.objects("rooms") if v["label"] == 3)
    mask = raster_geometry(shape(target["geometry"]), (48, 48))
    assert not mask[:, 24:].any()
    assert before.to_dict() == original[0]
    for obj, saved in zip((action, proposal), original[1:], strict=True):
        for key, value in saved.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(obj[key], value)
            else:
                assert obj[key] == value


def test_other_families_and_direct_mode_delegate_exactly_to_original():
    before, action, proposal = setup_case()
    for op, proposed in (("room_type", proposal), ("room_add", None)):
        request = dict(action, operation=op)
        old, a = apply_transaction(before, request, proposal=proposed)
        new, b = apply_permission_transaction(before, request, proposal=proposed)
        assert old.to_dict() == new.to_dict() and a == b
