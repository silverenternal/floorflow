#!/usr/bin/env python3
"""Audit the scoped evidence freeze without opening any new evaluation split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/floorplan_submission_evidence_freeze_v2.json"
OUTPUT = ROOT / "outputs/reports/floorplan_evidence_freeze_audit_v2.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_errors(name: str, rows: dict[str, dict[str, Any]]) -> list[str]:
    errors = []
    for key, row in rows.items():
        if row.get("exists") is False:
            errors.append(f"{name}.{key}: missing")
            continue
        path = ROOT / row["path"]
        if not path.is_file():
            errors.append(f"{name}.{key}: path missing: {row['path']}")
        elif path.stat().st_size != row.get("bytes"):
            errors.append(f"{name}.{key}: size mismatch")
        elif _sha256(path) != row.get("sha256"):
            errors.append(f"{name}.{key}: hash mismatch")
    return errors


def main() -> None:
    freeze = _load(FREEZE)
    registry = _load(ROOT / freeze["registry"]["path"])
    oracle = _load(ROOT / "outputs/reports/oracle_submission_statistics_v2.json")
    predicted = _load(ROOT / "outputs/reports/predicted_control_plausibility_fiveseed_v12.json")
    operation = _load(ROOT / "outputs/reports/predicted_operation_evidence_v2.json")
    provenance = _load(ROOT / "outputs/reports/floorplan_source_provenance_audit_v2.json")
    errors: list[str] = []
    errors.extend(_identity_errors("source_files", freeze["source_files"]))
    errors.extend(_identity_errors("checkpoints", freeze["checkpoints"]))
    errors.extend(_identity_errors("evaluators", freeze["evaluators"]))
    errors.extend(_identity_errors("release_sources", freeze["release_sources"]))
    registry_path = ROOT / freeze["registry"]["path"]
    if _sha256(registry_path) != freeze["registry"]["sha256"]:
        errors.append("registry hash mismatch")
    if freeze.get("mode") != "existing_evidence_scoped":
        errors.append("unexpected freeze mode")
    if freeze.get("status") != "frozen_scoped_release":
        errors.append("unexpected freeze status")
    if registry.get("status") != "passed_scoped_release":
        errors.append("registry is not passed_scoped_release")
    if provenance.get("status") != "passed":
        errors.append("source provenance audit is not passed")
    if oracle.get("status") != "passed_scoped_release":
        errors.append("oracle statistics are not passed_scoped_release")
    if not predicted.get("all_seeds_pass") or len(predicted.get("seeds", {})) != 5:
        errors.append("predicted five-seed release is incomplete")
    safety = operation.get("safety", {})
    if safety.get("split_values_in_rows") != ["test"] or safety.get("exposed") is True:
        pass
    if safety.get("prompts") != 738 or safety.get("executed") != 438 or safety.get("abstained") != 300:
        errors.append("operation prompt counts changed")
    if safety.get("outside_pixel_exact_min") != 1.0 or safety.get("validity_min") != 1.0:
        errors.append("operation safety exactness or validity is below the scoped claim")
    if "development-exposed" not in safety.get("exposure", ""):
        errors.append("operation exposure is not labeled development-exposed")
    semantic = operation.get("semantic_diagnostics", {}).get("operations", {})
    expected_counts = {
        "room_addition": (100, 92),
        "room_type_change": (100, 55),
        "room_deletion": (100, 20),
        "opening_pressure": (59, 30),
        "furniture_low_salience": (15, 0),
    }
    for key, (prompts, successes) in expected_counts.items():
        row = semantic.get(key, {})
        if row.get("prompts") != prompts or row.get("successful_prompts") != successes:
            errors.append(f"semantic diagnostic changed for {key}")
    heldout = freeze.get("heldout", {})
    if heldout.get("status") != "optional_generalization_not_evaluated":
        errors.append("optional held-out state is not separated")
    if heldout.get("blocks_scoped_submission") is not False:
        errors.append("optional held-out incorrectly blocks the scoped release")
    if freeze.get("historical_execution_hashes", {}).get("available") is not False:
        errors.append("historical execution hashes must not be fabricated")
    if not freeze.get("historical_execution_hashes", {}).get("limitation"):
        errors.append("historical hash limitation is not explicit")
    if freeze.get("clean_worktree_required") is not False:
        errors.append("scoped freeze incorrectly requires a clean worktree")
    audit = {
        "schema_version": "floorplan-evidence-freeze-audit-v2",
        "status": "passed_scoped_release" if not errors else "failed",
        "freeze": str(FREEZE.relative_to(ROOT)),
        "freeze_sha256": _sha256(FREEZE),
        "checks": {
            "scoped_mode": freeze.get("mode") == "existing_evidence_scoped",
            "source_registry_hash_locked": _sha256(registry_path) == freeze["registry"]["sha256"],
            "source_files_hash_locked": not _identity_errors("source_files", freeze["source_files"]),
            "checkpoint_identity_locked": not _identity_errors("checkpoints", freeze["checkpoints"]),
            "evaluator_identity_locked": not _identity_errors("evaluators", freeze["evaluators"]),
            "release_source_identity_locked": not _identity_errors("release_sources", freeze["release_sources"]),
            "oracle_statistics_complete": oracle.get("status") == "passed_scoped_release",
            "predicted_five_seed_release_complete": bool(predicted.get("all_seeds_pass")),
            "operation_safety_and_semantics_separated": operation.get("status")
            == "passed_scoped_release_with_semantic_boundary",
            "historical_execution_hashes_not_claimed": freeze["historical_execution_hashes"]["available"] is False,
            "optional_heldout_not_blocking": heldout.get("blocks_scoped_submission") is False,
            "no_clean_worktree_requirement": freeze.get("clean_worktree_required") is False,
        },
        "optional_generalization_not_evaluated": {
            "status": "optional_generalization_not_evaluated",
            "required_for_scoped_submission": False,
            "note": "A new held-out claim requires a separate pre-access freeze and one-shot run.",
        },
        "errors": errors,
        "boundary": (
            "This audit verifies identities and scoped evidence consistency. It does not certify that the "
            "historical seed reports were generated from the current source tree, and it does not open a new split."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
