"""Analytical CPU geometry cases; never instantiate or execute a neural model."""

import copy
import json

import numpy as np
import pytest
from shapely.geometry import box, shape

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.cad08_metrics import effect, geometry_validity, vector_preservation
from floorflow.inference.cad08_adapter import compose_visible, decode_saved_native, joint_presence_type
from floorflow.inference.cad08_transaction import apply_transaction, bind_before, create_binding, resolve_room_alias
from floorflow.inference.cad08_transaction_guarded_v1 import apply_transaction_with_state_guard
from floorflow.representation.cad08_state import LayeredCad08State, fingerprint, host_rows, object_record


def taxonomy():
    return FloorplanTaxonomy(
        ["background", "wall", "room:A", "room:B", "door:Door", "furniture:Sink"], {}, {}, {}, {}, {}
    )


def fields():
    fp = np.ones((48, 48), np.uint8)
    value = {
        "footprint": fp,
        "room_labels": np.full_like(fp, 2),
        "wall": np.zeros_like(fp),
        "opening": np.zeros_like(fp),
        "furniture": np.zeros_like(fp),
    }
    value["wall"][:2] = 1
    value["prediction"] = compose_visible(value)
    return value


def state(value=None, hosts=None):
    return LayeredCad08State.from_decoded(fields() if value is None else value, taxonomy(), "analytic", hosts=hosts)


def action(op="room_type", index=1):
    mask = np.ones((48, 48), bool)
    split = np.zeros_like(mask)
    split[:, 24:] = True
    return {
        "index": index,
        "operation": op,
        "source_mask": mask,
        "split_mask": split,
        "destination_mask": np.zeros_like(mask),
        "source_class_id": 2,
        "target_class_id": 3,
        "destination_ambiguous": False,
    }


def test_independent_layers_and_holes_render_exactly():
    value = fields()
    value["room_labels"][16:24, 16:24] = 0
    value["furniture"][4:14, 4:14] = 5
    value["furniture"][7:10, 7:10] = 0
    value["wall"][5:7, 5:12] = 1
    value["prediction"] = compose_visible(value)
    result = state(value)
    for key, rendered in result.render().items():
        np.testing.assert_array_equal(rendered, value[key])
    assert len(shape(result.objects("rooms")[0]["geometry"]).interiors) == 1
    assert len(shape(result.objects("fixtures")[0]["geometry"]).interiors) == 1
    assert (
        shape(result.objects("fixtures")[0]["geometry"])
        .intersection(shape(result.objects("wall_material")[0]["geometry"]))
        .area
        > 0
    )


def test_saved_opening_outside_footprint_is_preserved_and_reported_as_geometry_leak():
    value = fields()
    value["footprint"][:, :4] = 0
    value["room_labels"][:, :4] = 0
    value["wall"][:, :4] = 0
    value["opening"][8:12, 2:6] = 4
    value["prediction"] = compose_visible(value)
    exported = state(value)
    np.testing.assert_array_equal(exported.render()["opening"], value["opening"])
    np.testing.assert_array_equal(exported.render()["prediction"], value["prediction"])
    assert geometry_validity(exported)["geometry_status"] == "fail"
    assert geometry_validity(exported)["footprint_leak_objects"]


def test_presence_mass_is_not_destroyed_by_conditional_softmax():
    logits = np.log(np.array([0.9, 0.1]))[:, None, None]
    low = joint_presence_type(np.array([[0.01]]), logits)
    high = joint_presence_type(np.array([[1.0]]), logits)
    np.testing.assert_array_equal(low["conditional_type"], high["conditional_type"])
    assert low["joint_type"].sum() == pytest.approx(0.01)
    assert high["joint_type"].sum() == pytest.approx(1)
    assert low["presence"].item() < 0.5


def test_native_decoder_keeps_program_presence_and_type_as_separate_gates():
    frame = (48, 48)
    logits = np.zeros((2, *frame), np.float32)
    logits[0] = 1
    logits[1, :, 24:] = 3
    native = {
        "room_logits": logits,
        "program": np.array([True, False]),
        "opening_candidate_support": np.ones(frame, bool),
        "opening_probability_decoder": np.full(frame, 0.01),
        "opening_type_logits_decoder": np.full((1, *frame), 12.0),
        "wall_probability_decoder": np.full(frame, 0.6),
        "furniture_probability_decoder": np.full(frame, 0.01),
        "furniture_type_logits_decoder": np.full((1, *frame), 12.0),
        "support_logits_auxiliary": np.ones((1, *frame)),
    }
    decoded = decode_saved_native(native, np.ones(frame, np.uint8), taxonomy())
    assert np.all(decoded["room_labels"][:, :24] == 2)
    assert np.all(decoded["room_labels"][:, 24:] == 0)
    assert not decoded["opening"].any() and not decoded["furniture"].any()
    assert decoded["wall"].all()
    native["support_logits_auxiliary"] = np.full((1, *frame), -4.4703484e-8, np.float32)
    # The original decoder evaluates sigmoid in float32 before >= .5.
    # Replacing it by logit >= 0 loses a saved near-zero auxiliary cell.
    rounded = decode_saved_native(native, np.ones(frame, np.uint8), taxonomy())
    assert rounded["room_support"].all()


def test_cross_class_retype_retains_geometry_and_runtime_id():
    before = state()
    after, trace = apply_transaction(before, action())
    assert trace["accepted"] and after.objects("rooms")[0]["id"] == before.objects("rooms")[0]["id"]
    assert after.objects("rooms")[0]["geometry"] == before.objects("rooms")[0]["geometry"]
    assert effect(before, after, action())["E"] == "pass"
    assert before.objects("rooms")[0]["label"] == 2
    lineage = after.document["lineage"][-1]
    assert lineage["identity_preserved"] and lineage["geometry_sha256"] == fingerprint(before.objects("rooms")[0]["geometry"])


def test_adjacent_same_class_explicit_objects_survive_retype_and_serialization():
    before = state()
    left = object_record("analytic", "rooms", 2, box(0, 0, 24, 48), identity="runtime:left")
    right = object_record("analytic", "rooms", 3, box(24, 0, 48, 48), identity="runtime:right")
    before.document["objects"] = [v for v in before.document["objects"] if v["family"] != "rooms"] + [left, right]
    before.refresh_relationships()
    request = action()
    request["source_mask"][:, 24:] = False
    after, _ = apply_transaction(before, request)
    assert len(after.objects("rooms")) == 2
    assert {v["label"] for v in after.objects("rooms")} == {3}
    assert after.objects("rooms")[1] == right
    assert LayeredCad08State.from_dict(json.loads(json.dumps(after.to_dict()))).to_dict() == after.to_dict()


def test_initial_cc_readout_does_not_claim_persistent_author_ids():
    before = state()
    assert len(before.objects("rooms")) == 1
    assert not before.document["provenance"]["room_owner_predictions_available"]
    assert before.objects("rooms")[0]["identity_source"].startswith("system_created")


def test_ambiguous_before_binding_cannot_select_using_after():
    before = state()
    before.document["objects"] = [
        object_record("analytic", "rooms", 2, box(0, 0, 24, 48)),
        object_record("analytic", "rooms", 2, box(24, 0, 48, 48)),
    ]
    assert bind_before(before, np.ones((48, 48), bool), "rooms", 2)["status"] == "ambiguous"
    after, trace = apply_transaction(before, action(), proposal=fields())
    assert not trace["proposal_consumed"] and after.to_dict() == before.to_dict()


def test_state_bound_proposal_is_rejected_after_source_state_changes():
    before = state()
    request = action("room_add", 61)
    proposal = fields()
    proposal["room_labels"][:, 18:] = 3
    proposal["_source_state_sha256"] = fingerprint(before.document)

    proposed, accepted = apply_transaction_with_state_guard(before, request, proposal=proposal)
    assert accepted["accepted"] and accepted["proposal_consumed"]
    assert len(proposed.objects("rooms")) == 2
    assert accepted["source_state_sha256"] == proposal["_source_state_sha256"]

    retype = action("room_type", 62)
    changed, retype_trace = apply_transaction(before, retype)
    assert retype_trace["accepted"]
    assert changed.to_dict() != before.to_dict()

    unchanged, stale_trace = apply_transaction_with_state_guard(changed, request, proposal=proposal)
    assert not stale_trace["accepted"] and not stale_trace["proposal_consumed"]
    assert stale_trace["reasons"] == ["stale_proposal_source_state_mismatch"]
    assert stale_trace["expected_source_state_sha256"] == proposal["_source_state_sha256"]
    assert stale_trace["actual_source_state_sha256"] == fingerprint(changed.document)
    assert unchanged.to_dict() == changed.to_dict()


def test_replaying_a_committed_split_does_not_duplicate_child_identity():
    before = state()
    request = action("room_add", 63)

    split, first_trace = apply_transaction(before, request)
    assert first_trace["accepted"]
    assert len(split.objects("rooms")) == 2

    parent_id = before.objects("rooms")[0]["id"]
    retry = action("room_add", 63)
    retry["source_room_id"] = parent_id
    retry["source_mask"][:, 24:] = False
    proposal = fields()
    proposal["room_labels"][:, :14] = 3
    proposal["_source_state_sha256"] = fingerprint(split.document)

    replayed, replay_trace = apply_transaction(split, retry, proposal=proposal)
    assert not replay_trace["accepted"]
    assert replay_trace["reasons"] == ["split_child_runtime_id_already_exists"]
    assert replayed.to_dict() == split.to_dict()


def test_split_merge_records_lineage_and_keeps_other_layers():
    before = state()
    request = action("room_add")
    split, trace = apply_transaction(before, request)
    assert effect(before, split, request)["E"] == "pass"
    assert len(split.document["lineage"][0]["children"]) == 2
    assert vector_preservation(before, split, request, effect(before, split, request), trace)[
        "unselected_object_records_byte_exact"
    ]
    merge = action("room_remove", 2)
    merge["source_mask"][:, 24:] = False
    merge["destination_mask"][:, 24:] = True
    joined, _ = apply_transaction(split, merge)
    assert len(joined.objects("rooms")) == 1
    assert len(joined.document["lineage"][-1]["parents"]) == 2
    assert effect(split, joined, merge)["E"] == "pass"


def test_split_records_determinate_faces_and_keeps_ambiguous_fixture_relation_unknown():
    hosts = [{
        "id": "runtime:host:door-wall",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 8], [0, 8], [0, 0]]]},
        "start": [0, 4],
        "end": [10, 4],
        "thickness": 8.0,
    }]
    before = state(hosts=hosts)
    fixture = object_record("analytic", "fixtures", 5, box(22, 20, 26, 24), identity="runtime:fixture:crossing")
    opening = object_record("analytic", "openings", 4, box(2, 2, 4, 4), identity="runtime:opening:door")
    before.document["objects"].extend([fixture, opening])
    before.refresh_relationships()
    original_opening_host = copy.deepcopy(before.document["relationships"]["opening_host"])

    split, trace = apply_transaction(before, action("room_add", 41))

    assert trace["accepted"]
    event = split.document["lineage"][-1]
    source_id, child_id = event["children"]
    assert event["child_faces"][source_id]["role"] == "retained_remainder"
    assert event["child_faces"][child_id]["role"] == "requested_split"
    assert event["child_faces"][source_id]["geometry_sha256"] == fingerprint(
        next(room["geometry"] for room in split.objects("rooms") if room["id"] == source_id)
    )
    relation = split.document["relationships"]["fixture_room"][fixture["id"]]
    assert relation["status"] == "unknown" and relation["room_id"] is None
    assert trace["room_relation_revalidation"]["fixture_room"]["rows"][fixture["id"]]["reason"] == (
        "no_single_room_contains_full_fixture_geometry"
    )
    assert split.document["relationships"]["opening_host"] == original_opening_host
    assert event["relation_revalidation"]["opening_host"]["rows_exact"]

    opening_action = {
        "index": 42,
        "operation": "opening_delete",
        "source_mask": np.pad(np.ones((2, 2), dtype=bool), ((2, 44), (2, 44))),
        "split_mask": np.zeros((48, 48), dtype=bool),
        "destination_mask": np.zeros((48, 48), dtype=bool),
        "source_class_id": 4,
        "target_class_id": None,
        "destination_ambiguous": False,
    }
    continued, opening_trace = apply_transaction(split, opening_action)
    assert opening_trace["accepted"]
    assert continued.document["relationships"]["fixture_room"][fixture["id"]]["status"] == "unknown"


def test_merge_records_flat_room_alias_and_revalidates_fixture_relation():
    before = state()
    left = object_record("analytic", "rooms", 2, box(0, 0, 24, 48), identity="runtime:left")
    right = object_record("analytic", "rooms", 3, box(24, 0, 48, 48), identity="runtime:right")
    fixture = object_record("analytic", "fixtures", 5, box(32, 10, 36, 14), identity="runtime:fixture:right")
    before.document["objects"] = [
        obj for obj in before.document["objects"] if obj["family"] not in ("rooms", "fixtures")
    ] + [left, right, fixture]
    before.document["room_aliases"] = {"runtime:older-room": "runtime:right"}
    before.refresh_relationships()
    request = action("room_remove", 52)
    request["source_mask"][:, :24] = False
    request["source_class_id"] = 3
    request["destination_mask"][:, :24] = True
    request["target_class_id"] = 2

    merged, trace = apply_transaction(before, request)

    assert trace["accepted"]
    assert merged.document["room_aliases"] == {
        "runtime:older-room": "runtime:left",
        "runtime:right": "runtime:left",
    }
    assert resolve_room_alias(merged, "runtime:right") == {
        "status": "alias_resolved",
        "room_id": "runtime:left",
        "path": ["runtime:right", "runtime:left"],
    }
    assert resolve_room_alias(merged, "runtime:older-room")["room_id"] == "runtime:left"
    event = merged.document["lineage"][-1]
    assert event["retired_source_id"] == "runtime:right"
    assert event["retained_destination_id"] == "runtime:left"
    assert event["room_aliases_added"] == {"runtime:right": "runtime:left"}
    relation = merged.document["relationships"]["fixture_room"][fixture["id"]]
    assert relation["status"] == "unique" and relation["room_id"] == "runtime:left"
    assert event["relation_revalidation"]["fixture_room"]["rows"][fixture["id"]]["status"] == "unique"

    next_request = action("room_type", 53)
    next_request["source_mask"][:] = False
    next_request["source_mask"][4:12, 4:12] = True
    next_request["source_room_id"] = "runtime:right"
    binding = create_binding(merged, next_request)
    assert binding["source"]["status"] == "unique"
    assert binding["source"]["id"] == "runtime:left"
    assert binding["source"]["alias_resolution"]["status"] == "alias_resolved"
    retyped, retype_trace = apply_transaction(merged, next_request)
    assert retype_trace["accepted"]
    assert retyped.objects("rooms")[0]["id"] == "runtime:left"
    assert retyped.objects("rooms")[0]["label"] == 3


def test_input_merge_ambiguity_is_unknown_even_if_after_looks_correct():
    request = action("room_remove")
    request["destination_ambiguous"] = True
    before = state()
    after, trace = apply_transaction(before, request, proposal=fields())
    assert after.to_dict() == before.to_dict() and not trace["accepted"]
    assert effect(before, after, request)["E"] == "unknown"


def test_proposal_split_uses_saved_after_geometry_and_not_requested_mask_paint():
    before = state()
    proposal = fields()
    proposal["room_labels"][:, 28:] = 3
    request = action("room_add")
    a, trace = apply_transaction(before, request, proposal=proposal)
    other = copy.deepcopy(request)
    other["split_mask"] = np.zeros((48, 48), bool)
    other["split_mask"][:, :10] = True
    b, _ = apply_transaction(before, other, proposal=proposal)
    assert a.to_dict() == b.to_dict() and trace["proposal_consumed"]
    child = next(v for v in a.objects("rooms") if v["label"] == 3)
    assert shape(child["geometry"]).bounds == (28, 0, 48, 48)


def test_proposal_class_can_block_action_while_direct_baseline_does_not_use_it():
    before = state()
    denied, trace = apply_transaction(before, action(), proposal=fields())
    direct, direct_trace = apply_transaction(before, action())
    assert denied.to_dict() == before.to_dict() and trace["proposal_consumed"]
    assert not direct_trace["proposal_consumed"] and effect(before, direct, action())["E"] == "pass"


def test_same_host_multiple_openings_material_restoration_and_preservation():
    value = fields()
    value["wall"][8:12] = 1
    value["opening"][8:12, 8:12] = 4
    value["opening"][8:12, 28:32] = 4
    value["wall"][value["opening"] != 0] = 0
    value["prediction"] = compose_visible(value)
    hosts = host_rows(
        [{"start": [0, 10], "end": [48, 10], "thickness": 4}], value["footprint"].astype(bool), "analytic"
    )
    before = state(value, hosts)
    request = action("opening_delete")
    request["source_mask"] = np.zeros((48, 48), bool)
    request["source_mask"][8:12, 8:12] = True
    after, trace = apply_transaction(before, request)
    e = effect(before, after, request)
    assert e["E"] == "pass" and e["host_material_restored"] == "pass"
    assert len(after.objects("openings")) == 1
    assert vector_preservation(before, after, request, e, trace)["status"] == "pass"
    assert np.all(after.render()["wall"][8:12, 8:12] == 1)
    assert np.all(after.render()["opening"][8:12, 28:32] == 4)


def test_missing_host_is_explicit_unknown_without_synthesizing_wall():
    value = fields()
    value["opening"][8:12, 8:12] = 4
    value["prediction"] = compose_visible(value)
    before = state(value)
    request = action("opening_delete")
    request["source_mask"] = value["opening"] != 0
    after, trace = apply_transaction(before, request)
    assert after.to_dict() == before.to_dict() and not trace["accepted"]
    assert effect(before, after, request)["E"] == "unknown"
    assert geometry_validity(before)["relationship_status"] == "unknown"


def test_fixture_hole_deletion_preserves_other_geometry_class_and_room_assignment():
    value = fields()
    value["furniture"][4:16, 4:16] = 5
    value["furniture"][8:12, 8:12] = 0
    value["furniture"][30:34, 30:34] = 5
    value["prediction"] = compose_visible(value)
    before = state(value)
    request = action("furniture_delete")
    request["source_mask"] = (value["furniture"] != 0) & (np.indices((48, 48))[0] < 20)
    after, trace = apply_transaction(before, request)
    e = effect(before, after, request)
    p = vector_preservation(before, after, request, e, trace)
    assert e["E"] == "pass" and p["status"] == "pass"
    assert p["unselected_object_records_byte_exact"] and p["unselected_fixture_room_assignments_exact"]
