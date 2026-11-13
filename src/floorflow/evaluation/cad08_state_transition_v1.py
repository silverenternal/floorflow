"""Incremental issue and relationship auditing for CAD08 state transitions."""

from __future__ import annotations

from collections import Counter

from shapely.geometry import shape

from floorflow.evaluation.cad08_state_metrics_v1 import EPS


def transition_audit(before_g: dict, after_g: dict, p: dict) -> dict:
    before_issues = {v["key"]: v for v in before_g["issues"]}
    after_issues = {v["key"]: v for v in after_g["issues"]}
    inherited, introduced, resolved = [], [], []
    for key, issue in after_issues.items():
        if key in before_issues:
            old = before_issues[key]
            record = {
                "key": key,
                "status": issue["status"],
                "code": issue["code"],
                "source": issue["source"],
                "target": issue["target"],
                "evidence_unchanged": old["evidence"] == issue["evidence"],
            }
            inherited.append(record)
            if "violation_geometry" in issue["evidence"] and "violation_geometry" in old["evidence"]:
                extra = (
                    shape(issue["evidence"]["violation_geometry"])
                    .difference(shape(old["evidence"]["violation_geometry"]))
                    .area
                )
                if extra > EPS:
                    introduced.append({**record, "introduced_violation_area": float(extra)})
        else:
            introduced.append({k: issue[k] for k in ("key", "status", "code", "source", "target")})
    resolved = [
        {k: v[k] for k in ("key", "status", "code", "source", "target")}
        for k, v in before_issues.items()
        if k not in after_issues
    ]
    changes = {(v["table"] + "/" + v["source"]): v for v in p["relationship_changes"]}
    known, counts = [], Counter()
    for key, old in before_g["reference_validity"]["references"].items():
        if old["status"] != "pass":
            continue
        new = after_g["reference_validity"]["references"].get(key)
        change = changes.get(key)
        if new is None:
            category = (
                "authorized_selected_object_removed"
                if old["source"] in p["selected_before_IDs"]
                else "known_reference_lost"
            )
        elif new["target"] == old["target"]:
            if new["status"] != "pass":
                category = "known_reference_degraded_" + new["status"]
            elif new["record_sha256"] == old["record_sha256"]:
                category = "known_reference_retained_valid"
            elif change and change["category"] == "authorized_selected_room_update":
                category = "authorized_same_target_diagnostic_refresh"
            else:
                category = "unauthorized_known_reference_record_change"
        elif change and change["category"] == "authorized_selected_room_update":
            category = "authorized_reassignment_" + new["status"]
        else:
            category = "unauthorized_known_reference_change"
        counts[category] += 1
        known.append(
            {
                "key": key,
                "before_target": old["target"],
                "after_target": new["target"] if new else None,
                "after_status": new["status"] if new else "removed",
                "category": category,
            }
        )
    # A changed runtime target is also reported separately from a new invalid
    # condition: an unresolved owner can predate an edit even if its target ID
    # changed. Neither diagnostic replaces full G_after or J_state.
    condition = lambda v: (v["status"], v["code"], v["source"], v.get("evidence", {}).get("table"))
    prior_conditions = {condition(v) for v in before_issues.values()}
    inherited_conditions = [v for v in after_issues.values() if condition(v) in prior_conditions]
    new_conditions = [v for v in after_issues.values() if condition(v) not in prior_conditions]
    return {
        "G_before": before_g["G_state"],
        "G_after": after_g["G_state"],
        "inherited_issues": inherited,
        "introduced_issues": introduced,
        "resolved_issues": resolved,
        "counts": {
            "inherited_invalid": sum(v["status"] == "fail" for v in inherited),
            "inherited_unknown": sum(v["status"] == "unknown" for v in inherited),
            "introduced_invalid": sum(v["status"] == "fail" for v in introduced),
            "introduced_unknown": sum(v["status"] == "unknown" for v in introduced),
            "resolved": len(resolved),
        },
        "known_before_relationship_counts": dict(counts),
        "known_before_relationships": known,
        "condition_level_counts": {
            "inherited_invalid": sum(v["status"] == "fail" for v in inherited_conditions),
            "inherited_unknown": sum(v["status"] == "unknown" for v in inherited_conditions),
            "introduced_invalid": sum(v["status"] == "fail" for v in new_conditions),
            "introduced_unknown": sum(v["status"] == "unknown" for v in new_conditions),
        },
        "target_sensitive_issue_counts_and_condition_counts_are_both_reported": True,
        "no_new_issue_is_not_complete_J_state": True,
    }

__all__ = ["transition_audit"]
