#!/usr/bin/env python3
"""Audit floorplan submission artifacts without opening model evaluation splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/reports/floorplan_submission_integrity_v1.json"
PACKAGE = ROOT / "outputs/floorplan_submission"

REQUIRED_REPORTS = (
    "outputs/reports/residential_edit_test503_v1.json",
    "outputs/reports/residential_sota_main_results_v5.json",
    "outputs/reports/residential_sota_claim_gate_v5.json",
    "outputs/reports/residential_editing_aware_baseline_v1.json",
    "outputs/reports/predicted_control_plausibility_fiveseed_v12.json",
    "outputs/reports/predicted_operation_success_val191_v1.json",
    "outputs/reports/matched_local_editing_baselines_v1.json",
)
REQUIRED_PAPER_FILES = (
    "paper/tables/oracle_control_cvpr_dossier_v1.md",
    "paper/tables/predicted_control_reasonableness_dossier_v1.md",
    "paper/tables/main_oracle_anchor_v1.md",
    "paper/tables/main_predicted_reliability_v1.md",
    "paper/tables/main_resource_and_ablation_v1.md",
    "paper/tables/main_external_context_v1.md",
    "paper/figures/floorplan_protocol_overview_v1.pdf",
    "paper/figures/predicted_control_operation_panels_v1.pdf",
    "paper/figures/floorplan_failure_taxonomy_v1.pdf",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan_checks() -> tuple[dict, list[str]]:
    plan = _load(ROOT / "floorplansubmissiontodo.json")
    task_ids = [task["id"] for task in plan["tasks"]]
    known = set(task_ids)
    errors = []
    if len(task_ids) != len(known):
        errors.append("duplicate task id")
    for task in plan["tasks"]:
        missing = set(task["depends_on"]) - known
        if missing:
            errors.append(f"{task['id']} missing dependencies {sorted(missing)}")
    for phase in plan["phases"]:
        missing = set(phase["tasks"]) - known
        if missing:
            errors.append(f"{phase['id']} missing tasks {sorted(missing)}")
    return plan, errors


def _package_checks() -> list[str]:
    manifest_path = PACKAGE / "artifact_manifest.json"
    if not manifest_path.is_file():
        return ["artifact manifest missing"]
    manifest = _load(manifest_path)
    errors = []
    for row in manifest["files"]:
        path = PACKAGE / row["path"]
        if not path.is_file():
            errors.append(f"package file missing: {row['path']}")
        elif path.stat().st_size != row["bytes"]:
            errors.append(f"package size mismatch: {row['path']}")
        elif _sha256(path) != row["sha256"]:
            errors.append(f"package hash mismatch: {row['path']}")
    return errors


def main() -> None:
    plan, plan_errors = _plan_checks()
    missing_reports = [path for path in REQUIRED_REPORTS if not (ROOT / path).is_file()]
    missing_paper = [path for path in REQUIRED_PAPER_FILES if not (ROOT / path).is_file()]
    package_errors = _package_checks()
    predicted = _load(ROOT / "outputs/reports/predicted_operation_success_val191_v1.json")
    integrity = {
        "schema_version": "floorplan-submission-integrity-v1",
        "status": "pending_model_freeze_and_heldout",
        "checks": {
            "plan_json_valid": True,
            "plan_dependencies_valid": not plan_errors,
            "required_reports_present": not missing_reports,
            "required_paper_assets_present": not missing_paper,
            "artifact_manifest_valid": not package_errors,
            "predicted_audit_excludes_test": predicted["test_opened"] is False,
            "predicted_model_frozen": predicted["model_frozen"] is True,
            "heldout_result_present": (
                ROOT / "outputs/reports/residential_predicted_edit_heldout_v1.json"
            ).is_file(),
            "manuscript_tex_present": (ROOT / "paper/main.tex").is_file(),
            "supplement_tex_present": (ROOT / "paper/supplement.tex").is_file(),
        },
        "errors": {
            "plan": plan_errors,
            "missing_reports": missing_reports,
            "missing_paper_assets": missing_paper,
            "artifact_package": package_errors,
        },
        "task_status_counts": _task_status_counts(plan),
        "boundary": (
            "This audit does not modify or open the held-out split. A final "
            "release requires a frozen predicted model and one held-out run. "
            "The current manuscript is a candidate draft."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(integrity, indent=2))


def _task_status_counts(plan: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in plan["tasks"]:
        status = str(task["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


if __name__ == "__main__":
    main()
