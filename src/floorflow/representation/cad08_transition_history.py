"""External transition evidence; not a CAD lineage or a safety score.

The caller supplies actual snapshots and an independently trusted predecessor
head. Content hashes bind bytes to the submitted record, not to an author or
physical object. No existing CAD document or scoring path is modified.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np

from floorflow.representation.cad08_content_digest import content_digest

SCHEMA = "floorflow-cad08-transition-history-v1"
CONTENT_SCHEMA = "floorflow-oracle-bound-content-v1"


def _record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256((SCHEMA + "\0").encode() + payload).hexdigest()


def _snapshot_digest(value: Any | None) -> str | None:
    return None if value is None else content_digest(value)


def make_transition(
    *,
    branch: str,
    sequence: int,
    previous_head: str | None,
    kind: str,
    operation: str | None,
    before_cad: dict[str, Any] | None,
    after_cad: dict[str, Any] | None,
    before_native: dict[str, np.ndarray] | None,
    after_native: dict[str, np.ndarray] | None,
    event: dict[str, Any] | None = None,
    trace: dict[str, Any] | None = None,
    action_content_sha256: str | None = None,
) -> dict[str, Any]:
    """Capture supplied content identities and original event/trace without scoring.

    Missing snapshots are retained as null, so a partial record is not silently
    treated as complete. Event and trace must be JSON-serializable as saved.
    """
    if kind not in {"event", "no_emission", "rebuild"}:
        raise ValueError("kind must be event, no_emission or rebuild")
    record = {
        "schema": SCHEMA,
        "content_schema": CONTENT_SCHEMA,
        "branch": branch,
        "sequence": sequence,
        "previous_head": previous_head,
        "kind": kind,
        "operation": operation,
        "before_cad": _snapshot_digest(before_cad),
        "after_cad": _snapshot_digest(after_cad),
        "before_native": _snapshot_digest(before_native),
        "after_native": _snapshot_digest(after_native),
        "event": copy.deepcopy(event),
        "trace": copy.deepcopy(trace),
    }
    if action_content_sha256 is not None:
        record["action_evidence_schema"] = "floorflow-action-content-attestation-v1"
        record["action_content_sha256"] = action_content_sha256
    record["record_hash"] = _record_hash(record)
    return record


def _ids(document: dict[str, Any]) -> set[str] | None:
    objects = document.get("objects")
    if not isinstance(objects, list) or any(not isinstance(obj, dict) or not isinstance(obj.get("id"), str) for obj in objects):
        return None
    ids = [obj["id"] for obj in objects]
    return set(ids) if len(set(ids)) == len(ids) else None


def verify_history(
    records: list[dict[str, Any]],
    snapshots: list[dict[str, Any] | None],
    *,
    branch: str,
    trusted_previous_head: str | None,
    trusted_head: str | None,
    action_evidence: list[dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Verify internal stage linkage against given snapshots and a caller's anchor.

    A passing ``link_integrity`` means only that supplied bytes, local ordering
    and checkable event/trace relations agree. Overall remains unknown unless a
    contradiction is found: permissions, cross-rebuild identity, history prior
    to the anchor, semantic effect and original CAD scores are not established.
    For a record with ``action_content_sha256``, independently trusted per-stage
    evidence must provide both the action and its trusted digest. This can supply
    an operation check when the unchanged original trace has no operation field;
    it does not establish that a callback acted faithfully on that action.
    """
    statuses: list[str] = []
    details: list[dict[str, Any]] = []
    previous = trusted_previous_head
    last_after: tuple[str | None, str | None] | None = None
    for index, record in enumerate(records):
        failures: list[str] = []
        missing: list[str] = []
        core = {k: v for k, v in record.items() if k != "record_hash"}
        if record.get("schema") != SCHEMA or record.get("content_schema") != CONTENT_SCHEMA:
            failures.append("schema_mismatch")
        if record.get("record_hash") != _record_hash(core):
            failures.append("record_hash_mismatch")
        if record.get("branch") != branch or record.get("sequence") != index:
            failures.append("branch_or_sequence_mismatch")
        if index == 0 and trusted_previous_head is None:
            missing.append("trusted_previous_head_missing")
        elif record.get("previous_head") != previous:
            failures.append("previous_head_mismatch")
        current_before = (record.get("before_cad"), record.get("before_native"))
        if last_after is not None:
            if None in (*last_after, *current_before):
                missing.append("transition_boundary_missing")
            elif last_after != current_before:
                failures.append("transition_boundary_mismatch")
        snap = snapshots[index] if index < len(snapshots) else None
        if snap is None:
            missing.append("snapshots_missing")
        else:
            for key in ("before_cad", "after_cad", "before_native", "after_native"):
                value = snap.get(key)
                if value is None or record.get(key) is None:
                    missing.append(key + "_missing")
                elif _snapshot_digest(value) != record[key]:
                    failures.append(key + "_hash_mismatch")
        kind, event, trace = record.get("kind"), record.get("event"), record.get("trace")
        action_digest = record.get("action_content_sha256")
        attestation = action_evidence[index] if action_evidence is not None and index < len(action_evidence) else None
        action_verified = False
        if action_digest is None and "action_evidence_schema" in record:
            missing.append("action_content_hash_missing")
        if action_digest is not None:
            if record.get("action_evidence_schema") != "floorflow-action-content-attestation-v1":
                failures.append("action_evidence_schema_mismatch")
            elif not isinstance(attestation, dict) or attestation.get("action") is None or attestation.get("trusted_sha256") is None:
                missing.append("trusted_action_evidence_missing")
            elif (content_digest(attestation["action"]) != action_digest
                  or attestation["trusted_sha256"] != action_digest):
                failures.append("action_content_hash_mismatch")
            elif not isinstance(attestation["action"], dict) or attestation["action"].get("operation") is None:
                missing.append("action_operation_missing")
            elif attestation["action"]["operation"] != record.get("operation"):
                failures.append("action_operation_mismatch")
            else:
                action_verified = True
        if kind not in {"event", "no_emission", "rebuild"}:
            failures.append("unknown_transition_kind")
        elif kind == "event":
            if not isinstance(event, dict) or not isinstance(trace, dict):
                missing.append("event_or_trace_missing")
            else:
                if "accepted" not in trace:
                    missing.append("trace_accepted_missing")
                elif trace["accepted"] is not True:
                    failures.append("event_trace_not_accepted")
                if event.get("operation") is None or record.get("operation") is None:
                    missing.append("event_operation_missing")
                elif event["operation"] != record["operation"]:
                    failures.append("event_operation_mismatch")
                if trace.get("operation") is not None and trace["operation"] != record.get("operation"):
                    failures.append("trace_operation_mismatch")
                if trace.get("operation") is None and not action_verified:
                    missing.append("trace_operation_missing")
                parents, children = event.get("parents"), event.get("children")
                if not isinstance(parents, list) or not isinstance(children, list):
                    missing.append("event_ids_missing")
                elif not parents or len(parents) != len(set(parents)) or len(children) != len(set(children)):
                    failures.append("event_trace_ids_mismatch")
                elif "selected_before_ids" not in trace or "selected_after_ids" not in trace:
                    missing.append("trace_selected_ids_missing")
                elif parents != trace["selected_before_ids"] or children != trace["selected_after_ids"]:
                    failures.append("event_trace_ids_mismatch")
                elif snap is not None and snap.get("before_cad") is not None and snap.get("after_cad") is not None:
                    before_doc, after_doc = snap["before_cad"], snap["after_cad"]
                    b, a = _ids(before_doc), _ids(after_doc)
                    if b is None or a is None:
                        missing.append("snapshot_object_ids_unavailable")
                    elif not set(parents) <= b or not set(children) <= a:
                        failures.append("event_ids_outside_actual_stage")
                    before_lineage, after_lineage = before_doc.get("lineage"), after_doc.get("lineage")
                    if not isinstance(before_lineage, list) or not isinstance(after_lineage, list):
                        missing.append("stage_lineage_unavailable")
                    elif after_lineage != before_lineage + [event]:
                        failures.append("event_not_actual_lineage_suffix")
                else:
                    missing.append("event_snapshot_missing")
        elif kind == "no_emission":
            if event is not None:
                failures.append("no_emission_has_event")
            if not isinstance(trace, dict) or "accepted" not in trace:
                missing.append("no_emission_trace_missing")
            elif trace["accepted"] is not False:
                failures.append("no_emission_trace_accepted")
            if (
                isinstance(trace, dict)
                and trace.get("operation") is not None
                and trace["operation"] != record.get("operation")
            ):
                failures.append("trace_operation_mismatch")
            if any(record.get(k) is None for k in ("before_cad", "after_cad", "before_native", "after_native")):
                missing.append("no_emission_state_missing")
            elif record["before_cad"] != record["after_cad"] or record["before_native"] != record["after_native"]:
                failures.append("no_emission_changed_state")
        elif kind == "rebuild" and event is not None:
            failures.append("rebuild_has_event")
        status = "fail" if failures else "unknown" if missing else "pass"
        statuses.append(status)
        details.append({"sequence": index, "status": status, "failures": failures, "missing": missing})
        previous = record.get("record_hash")
        last_after = (record.get("after_cad"), record.get("after_native"))
    if len(snapshots) > len(records):
        statuses.append("fail")
        details.append({"status": "fail", "failures": ["extra_snapshots"], "missing": []})
    if trusted_head is None or not records:
        statuses.append("unknown")
        details.append({"status": "unknown", "failures": [], "missing": ["trusted_head_or_history_missing"]})
    elif records[-1].get("record_hash") != trusted_head:
        statuses.append("fail")
        details.append({"status": "fail", "failures": ["trusted_head_mismatch"], "missing": []})
    link = "fail" if "fail" in statuses else "unknown" if not records or "unknown" in statuses else "pass"
    return {
        "link_integrity": link,
        "status": "fail" if link == "fail" else "unknown",
        "history_before_anchor": "unknown",
        "cross_rebuild_identity": "unknown",
        "permissions_and_semantic_effect": "unknown",
        "original_epgj": "not_evaluated",
        "stages": details,
    }
