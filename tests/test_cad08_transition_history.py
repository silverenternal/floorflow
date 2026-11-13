import copy
import json

import numpy as np

from floorflow.representation.cad08_content_digest import content_digest
from floorflow.representation.cad08_transition_history import make_transition, verify_history


def test_history_preserves_stage_evidence_across_rebuild_and_deleted_parent():
    room_event = {"operation": "room_type", "parents": ["room-a"], "children": ["room-a"]}
    opening_event = {"operation": "opening_delete", "parents": ["opening-b"], "children": []}
    room_trace = {"operation": "room_type", "accepted": True,
                  "selected_before_ids": ["room-a"], "selected_after_ids": ["room-a"]}
    opening_trace = {"operation": "opening_delete", "accepted": True,
                     "selected_before_ids": ["opening-b"], "selected_after_ids": []}
    s0 = {"objects": [{"id": "room-a", "label": 1}, {"id": "opening-a"}], "lineage": []}
    s1 = {"objects": [{"id": "room-a", "label": 2}, {"id": "opening-a"}], "lineage": [room_event]}
    # A rebuild changes runtime identities and does not carry the original lineage.
    s2 = {"objects": [{"id": "room-b", "label": 2}, {"id": "opening-b"}], "lineage": []}
    s3 = {"objects": [{"id": "room-b", "label": 2}], "lineage": [opening_event]}
    native = [{"room_logits": np.array([i], dtype=np.float32)} for i in range(4)]
    stage_pairs = [(s0, s1), (s1, s2), (s2, s3)]
    records, snapshots = [], []
    head = "externally-trusted-predecessor"
    for i, ((before, after), kind, operation, event, trace) in enumerate(zip(
        stage_pairs, ("event", "rebuild", "event"), ("room_type", None, "opening_delete"),
        (room_event, None, opening_event), (room_trace, None, opening_trace), strict=True,
    )):
        snapshots.append({"before_cad": before, "after_cad": after,
                          "before_native": native[i], "after_native": native[i + 1]})
        record = make_transition(branch="direct", sequence=i, previous_head=head, kind=kind, operation=operation,
                                 event=event, trace=trace, **snapshots[-1])
        records.append(record)
        head = record["record_hash"]

    # JSON roundtrip retains original event and trace records.
    records = json.loads(json.dumps(records))
    result = verify_history(records, snapshots, branch="direct", trusted_previous_head="externally-trusted-predecessor",
                            trusted_head=head)
    assert result["link_integrity"] == "pass"
    assert result["status"] == result["cross_rebuild_identity"] == "unknown"
    assert result["original_epgj"] == "not_evaluated"
    assert opening_event["parents"][0] not in {obj["id"] for obj in s3["objects"]}

    absent = verify_history(records, [snapshots[0], None, snapshots[2]], branch="direct",
                            trusted_previous_head="externally-trusted-predecessor", trusted_head=head)
    assert absent["link_integrity"] == "unknown"
    untrusted = verify_history(records, snapshots, branch="direct",
                               trusted_previous_head="externally-trusted-predecessor", trusted_head=None)
    assert untrusted["link_integrity"] == "unknown"
    incomplete_trace = dict(opening_trace)
    del incomplete_trace["operation"]
    partial = make_transition(branch="direct", sequence=0, previous_head="anchor", kind="event",
                              operation="opening_delete", event=opening_event, trace=incomplete_trace, **snapshots[2])
    partial_result = verify_history([partial], [snapshots[2]], branch="direct", trusted_previous_head="anchor",
                                    trusted_head=partial["record_hash"])
    assert partial_result["link_integrity"] == "unknown"
    action = {"operation": "room_type"}
    attested = make_transition(branch="direct", sequence=0, previous_head="anchor", kind="event",
                               operation="opening_delete", event=opening_event, trace=incomplete_trace,
                               action_content_sha256=content_digest(action), **snapshots[2])
    conflict = verify_history([attested], [snapshots[2]], branch="direct", trusted_previous_head="anchor",
                              trusted_head=attested["record_hash"],
                              action_evidence=[{"action": action, "trusted_sha256": content_digest(action)}])
    assert conflict["link_integrity"] == "fail"
    assert "action_operation_mismatch" in conflict["stages"][0]["failures"]
    changed = copy.deepcopy(snapshots)
    changed[2]["before_cad"]["objects"][1]["id"] = "different-opening"
    contradicted = verify_history(records, changed, branch="direct",
                                  trusted_previous_head="externally-trusted-predecessor", trusted_head=head)
    assert contradicted["link_integrity"] == contradicted["status"] == "fail"


def test_no_emission_preserves_both_states_without_claiming_no_proposal():
    cad = {"objects": [], "lineage": []}
    native = {"program": np.array([True, False]), "room_logits": np.array([1.0], dtype=np.float16)}
    snapshots = [{"before_cad": cad, "after_cad": copy.deepcopy(cad),
                  "before_native": native, "after_native": {k: v.copy() for k, v in native.items()}}]
    record = make_transition(branch="neural", sequence=0, previous_head="anchor", kind="no_emission",
                             operation="room_type", event=None, trace={"accepted": False}, **snapshots[0])
    result = verify_history([record], snapshots, branch="neural", trusted_previous_head="anchor",
                            trusted_head=record["record_hash"])
    assert result["link_integrity"] == "pass" and result["status"] == "unknown"
    assert result["permissions_and_semantic_effect"] == "unknown"

    action = {"operation": "room_type"}
    contradicted = make_transition(
        branch="neural", sequence=0, previous_head="anchor", kind="no_emission", operation="room_type",
        event=None, trace={"accepted": False, "operation": "room_remove"},
        action_content_sha256=content_digest(action), **snapshots[0],
    )
    conflict = verify_history(
        [contradicted], snapshots, branch="neural", trusted_previous_head="anchor",
        trusted_head=contradicted["record_hash"],
        action_evidence=[{"action": action, "trusted_sha256": content_digest(action)}],
    )
    assert conflict["link_integrity"] == "fail"
    assert "trace_operation_mismatch" in conflict["stages"][0]["failures"]
    matching = make_transition(
        branch="neural", sequence=0, previous_head="anchor", kind="no_emission", operation="room_type",
        event=None, trace={"accepted": False, "operation": "room_type"},
        action_content_sha256=content_digest(action), **snapshots[0],
    )
    matched = verify_history(
        [matching], snapshots, branch="neural", trusted_previous_head="anchor",
        trusted_head=matching["record_hash"],
        action_evidence=[{"action": action, "trusted_sha256": content_digest(action)}],
    )
    assert matched["link_integrity"] == "pass"

    changed_native = {"program": np.array([False, False]), "room_logits": native["room_logits"].copy()}
    changed = make_transition(branch="neural", sequence=0, previous_head="anchor", kind="no_emission",
                              operation="room_type", event=None, trace={"accepted": False},
                              before_cad=cad, after_cad=cad, before_native=native, after_native=changed_native)
    changed_snapshots = [{"before_cad": cad, "after_cad": cad, "before_native": native, "after_native": changed_native}]
    invalid = verify_history([changed], changed_snapshots, branch="neural", trusted_previous_head="anchor",
                             trusted_head=changed["record_hash"])
    assert invalid["status"] == "fail"
