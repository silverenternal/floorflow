#!/usr/bin/env python3
"""Audit the scoped FloorFlow submission release without opening any new split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/reports/floorplan_submission_integrity_v1.json"
PACKAGE = ROOT / "outputs/floorplan_submission"
REGISTRY = ROOT / "docs/floorplan_submission_source_registry_v2.json"
PROVENANCE = ROOT / "outputs/reports/floorplan_source_provenance_audit_v2.json"
FREEZE = ROOT / "configs/floorplan_submission_evidence_freeze_v2.json"
FREEZE_AUDIT = ROOT / "outputs/reports/floorplan_evidence_freeze_audit_v2.json"
HELDOUT = ROOT / "outputs/reports/floorplan_heldout_availability_v1.json"

TABLES = (
    "paper/tables/main_oracle_anchor_v1",
    "paper/tables/main_predicted_reliability_v1",
    "paper/tables/main_resource_and_ablation_v1",
    "paper/tables/main_external_context_v1",
)
FIGURES = (
    "paper/figures/floorplan_protocol_overview_v1",
    "paper/figures/predicted_control_operation_panels_v1",
    "paper/figures/floorplan_failure_taxonomy_v1",
    "paper/figures/floorplan_qualitative_cases_v1",
)
REQUIRED_TEXT = (
    "paper/main.tex",
    "paper/supplement.tex",
    "paper/references.bib",
    "paper/writing_source_policy_v1.md",
    "docs/floorplan_claim_evidence_matrix_v1.md",
    "paper/tables/oracle_control_cvpr_dossier_v1.md",
    "paper/tables/predicted_control_reasonableness_dossier_v1.md",
    "docs/floorplan_submission_freeze_checklist_v1.md",
    "docs/floorplan_heldout_availability_v1.md",
    "outputs/reports/floorplan_cvpr_review_v1.md",
    "outputs/reports/floorplan_revision_roadmap_v1.json",
    "docs/floorplan_submission_review_v1.md",
)
STALE_PATTERNS = (
    "0.035690-0.164930",
    "0.9848-0.9903",
    "0.000168-0.000259",
    "Delta-IoU = 0.874-0.899",
    "pending_model_freeze_and_heldout",
    "model and held-out test not frozen",
)
PLACEHOLDERS = ("TODO", "TBD", "PLACEHOLDER")
TERMINAL_STATUSES = {
    "completed",
    "completed_scoped_release",
    "completed_packaged",
    "completed_ready_for_user_submission",
    "deferred_optional_not_applicable",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_pointer(document: Any, pointer: str) -> Any:
    if pointer in ("", "/"):
        return document
    current = document
    for raw in pointer.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


def _plan_checks(plan: dict) -> list[str]:
    errors: list[str] = []
    tasks = plan.get("tasks", [])
    ids = [task["id"] for task in tasks]
    known = set(ids)
    if len(ids) != len(known):
        errors.append("duplicate task id")
    optional = {task["id"] for task in tasks if task["id"] == "FP-SUB-06"}
    by_id = {task["id"]: task for task in tasks}
    for task in tasks:
        missing = set(task.get("depends_on", [])) - known
        if missing:
            errors.append(f"{task['id']} missing dependencies {sorted(missing)}")
        if task["id"] not in optional:
            for required_optional in optional & set(task.get("depends_on", [])):
                errors.append(f"required task {task['id']} depends on optional {required_optional}")
        if task["id"] not in optional:
            for deliverable in task.get("deliverables", []):
                if not (ROOT / deliverable).is_file():
                    errors.append(f"{task['id']} missing deliverable {deliverable}")
        if task.get("status") not in TERMINAL_STATUSES:
            errors.append(f"{task['id']} still has status {task.get('status')}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            errors.append(f"dependency cycle at {node}")
            return
        if node in visited:
            return
        visiting.add(node)
        for dependency in by_id.get(node, {}).get("depends_on", []):
            if dependency in by_id:
                visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for task_id in by_id:
        visit(task_id)
    return errors


def _registry_checks(registry: dict) -> list[str]:
    errors: list[str] = []
    keys: set[str] = set()
    source_cache: dict[str, Any] = {}
    for entry in registry.get("entries", []):
        key = entry.get("key")
        if not key or key in keys:
            errors.append(f"duplicate or empty registry key: {key}")
            continue
        keys.add(key)
        source = ROOT / entry["source_path"]
        if not source.is_file():
            errors.append(f"registry source missing: {entry['source_path']}")
            continue
        if entry["source_path"] not in source_cache:
            try:
                source_cache[entry["source_path"]] = _load(source)
            except json.JSONDecodeError:
                source_cache[entry["source_path"]] = None
                errors.append(f"registry source is not JSON: {entry['source_path']}")
        document = source_cache[entry["source_path"]]
        if document is None or not entry.get("json_pointer"):
            continue
        try:
            _resolve_pointer(document, entry["json_pointer"])
        except (KeyError, IndexError, TypeError, ValueError):
            errors.append(f"unresolved pointer {entry['source_path']}#{entry['json_pointer']}")
    for table, bindings in registry.get("table_bindings", {}).items():
        unknown = set(bindings) - keys
        if unknown:
            errors.append(f"{table} has unknown bindings {sorted(unknown)}")
    return errors


def _freeze_checks(freeze: dict, registry: dict) -> list[str]:
    errors: list[str] = []
    if freeze.get("mode") != "existing_evidence_scoped":
        errors.append("unexpected freeze mode")
    if freeze.get("status") != "frozen_scoped_release":
        errors.append("unexpected freeze status")
    if freeze.get("registry", {}).get("sha256") != _sha256(REGISTRY):
        errors.append("freeze registry hash is stale")
    for group in ("source_files", "checkpoints", "evaluators", "release_sources"):
        for name, row in freeze.get(group, {}).items():
            path = ROOT / row.get("path", "")
            if not path.is_file():
                errors.append(f"frozen {group} path missing: {name}")
            elif row.get("sha256") and row["sha256"] != _sha256(path):
                errors.append(f"frozen {group} hash mismatch: {name}")
    heldout = freeze.get("heldout", {})
    if heldout.get("status") != "optional_generalization_not_evaluated":
        errors.append("optional held-out state is not separated")
    if heldout.get("blocks_scoped_submission") is not False:
        errors.append("optional held-out incorrectly blocks the scoped submission")
    if freeze.get("historical_execution_hashes", {}).get("available") is not False:
        errors.append("historical execution hashes are overclaimed")
    if freeze.get("clean_worktree_required") is not False:
        errors.append("scoped release incorrectly requires a clean worktree")
    if registry.get("status") != "passed_scoped_release":
        errors.append("registry is not passed_scoped_release")
    return errors


def _table_checks(entries: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for stem in TABLES:
        markdown = ROOT / f"{stem}.md"
        latex = ROOT / f"{stem}.tex"
        if not markdown.is_file() or not latex.is_file():
            errors.append(f"table outputs missing: {stem}")
            continue
        text = markdown.read_text(encoding="utf-8")
        if "registry bindings:" not in text:
            errors.append(f"table lacks registry binding annotation: {stem}")
        latex_text = latex.read_text(encoding="utf-8")
        if "\\begin{{" in latex_text or "\\end{{" in latex_text:
            errors.append(f"table latex contains an unformatted f-string brace: {stem}")
    t1 = (ROOT / "paper/tables/main_oracle_anchor_v1.md").read_text(encoding="utf-8")
    for value in ("0.999199", "0.999801", "0.999756", "0.956703", "0.987812", "2,479"):
        if value not in t1:
            errors.append(f"T1 missing frozen value {value}")
    t2 = (ROOT / "paper/tables/main_predicted_reliability_v1.md").read_text(encoding="utf-8")
    for value in ("0.816701", "0.997963", "438 / 438", "300 / 300", "92 / 100", "0 / 15"):
        if value not in t2:
            errors.append(f"T2 missing frozen value {value}")
    if "Raw watertight" not in t2 or "Functional watertight" not in t2:
        errors.append("T2 does not separate raw and functional watertightness")
    t3 = (ROOT / "paper/tables/main_resource_and_ablation_v1.md").read_text(encoding="utf-8")
    for value in ("0.137160", "0.166756", "0.177266", "0.189491", "0.145367", "3.1925", "0.2666"):
        if value not in t3:
            errors.append(f"T3 missing frozen value {value}")
    t4 = (ROOT / "paper/tables/main_external_context_v1.md").read_text(encoding="utf-8")
    for value in ("0.608000", "0.001900", "0.269701", "0.885268"):
        if value not in t4:
            errors.append(f"T4 missing frozen value {value}")
    return errors


def _figure_checks() -> list[str]:
    errors: list[str] = []
    for stem in FIGURES:
        for suffix in (".pdf", ".svg"):
            if not (ROOT / f"{stem}{suffix}").is_file():
                errors.append(f"figure missing: {stem}{suffix}")
    manifest_path = ROOT / "outputs/reports/floorplan_figure_case_manifest_v1.json"
    if not manifest_path.is_file():
        return errors + ["figure case manifest missing"]
    manifest = _load(manifest_path)
    if len(manifest.get("cases", [])) < 2:
        errors.append("figure case manifest lacks both oracle and predicted cases")
    for case in manifest.get("cases", []):
        for source in case.get("sources", [case.get("source")]):
            if source and not (ROOT / source).is_file():
                errors.append(f"figure case source missing: {source}")
        if case.get("role") == "predicted_failure" and not case.get("failure_reason"):
            errors.append("predicted-failure figure case lacks a failure reason")
    return errors


def _text_checks() -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_TEXT:
        if not (ROOT / name).is_file():
            errors.append(f"required text missing: {name}")
    active = list(REQUIRED_TEXT) + [f"{stem}.md" for stem in TABLES]
    for name in active:
        path = ROOT / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in STALE_PATTERNS:
            if pattern in text:
                errors.append(f"{name} contains stale text {pattern!r}")
        if path.suffix == ".md":
            for token in PLACEHOLDERS:
                if token in text:
                    errors.append(f"{name} contains placeholder token {token}")
    policy = (ROOT / "paper/writing_source_policy_v1.md").read_text(encoding="utf-8")
    for phrase in ("source_registry_v2", "evidence_freeze_v2", "main_oracle_anchor_v1", "main_external_context_v1"):
        if phrase not in policy:
            errors.append(f"source policy lacks {phrase}")
    if "cvpr_package_fp_v17" in policy and "not manuscript sources" not in policy:
        errors.append("source policy still treats v17 as canonical")
    main = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    supplement = (ROOT / "paper/supplement.tex").read_text(encoding="utf-8")
    for phrase in ("\\usepackage[review]{cvpr}", "Anonymous CVPR submission", "LLM Usage Statement"):
        if phrase not in main:
            errors.append(f"main.tex lacks {phrase}")
    for stem in TABLES:
        if f"tables/{Path(stem).name}" not in main:
            errors.append(f"main.tex does not input {stem}")
    for figure in ("floorplan_protocol_overview_v1", "floorplan_qualitative_cases_v1"):
        if figure not in main:
            errors.append(f"main.tex does not reference {figure}")
    for figure in ("predicted_control_operation_panels_v1", "floorplan_failure_taxonomy_v1"):
        if figure not in supplement:
            errors.append(f"supplement.tex does not reference {figure}")
    return errors


def _evidence_checks() -> list[str]:
    errors: list[str] = []
    operation = _load(ROOT / "outputs/reports/predicted_operation_evidence_v2.json")
    safety = operation.get("safety", {})
    if safety.get("split_values_in_rows") != ["test"]:
        errors.append("operation rows are not labeled as test exposure")
    if "development-exposed" not in safety.get("exposure", ""):
        errors.append("operation exposure is not development-exposed")
    if safety.get("prompts") != 738 or safety.get("executed") != 438 or safety.get("abstained") != 300:
        errors.append("operation prompt counts changed")
    if safety.get("outside_pixel_exact_min") != 1.0 or safety.get("validity_min") != 1.0:
        errors.append("operation safety contract changed")
    semantic = operation.get("semantic_diagnostics", {})
    if semantic.get("split") != "val191":
        errors.append("semantic diagnostic is not labeled val191")
    expected = {"room_addition": (100, 92), "room_type_change": (100, 55), "room_deletion": (100, 20), "opening_pressure": (59, 30), "furniture_low_salience": (15, 0)}
    for key, (prompts, successes) in expected.items():
        row = semantic.get("operations", {}).get(key, {})
        if row.get("prompts") != prompts or row.get("successful_prompts") != successes:
            errors.append(f"semantic row changed: {key}")
    heldout = _load(HELDOUT)
    if heldout.get("status") != "optional_generalization_not_evaluated":
        errors.append("held-out availability report does not isolate the optional extension")
    if (ROOT / "outputs/reports/residential_predicted_edit_heldout_v1.json").exists():
        errors.append("unexpected fresh held-out result exists")
    return errors


def _package_checks() -> list[str]:
    errors: list[str] = []
    manifest_path = PACKAGE / "artifact_manifest.json"
    if not manifest_path.is_file():
        return ["artifact manifest missing"]
    manifest = _load(manifest_path)
    if manifest.get("status") != "ready_for_user_submission":
        errors.append("artifact manifest is not ready_for_user_submission")
    if manifest.get("anonymous") is not True:
        errors.append("artifact is not marked anonymous")
    if manifest.get("contains_datasets_or_checkpoints") is not False:
        errors.append("artifact incorrectly includes datasets or checkpoints")
    for row in manifest.get("files", []):
        path = PACKAGE / row["path"]
        if not path.is_file():
            errors.append(f"package file missing: {row['path']}")
        elif path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            errors.append(f"package identity mismatch: {row['path']}")
        if path.suffix in {".npz", ".npy", ".pt", ".pth", ".ckpt"}:
            errors.append(f"package contains disallowed binary: {row['path']}")
    checklist_path = PACKAGE / "submission_checklist.json"
    if not checklist_path.is_file():
        errors.append("submission checklist missing")
    else:
        checklist = _load(checklist_path)
        if checklist.get("status") != "ready_for_user_submission":
            errors.append("submission checklist is not ready_for_user_submission")
        if not all(checklist.get("checks", {}).values()):
            errors.append("submission checklist has a failed local check")
        if not checklist.get("external_actions_requiring_user_authorization"):
            errors.append("submission checklist omits external user actions")
    return errors


def main() -> None:
    registry = _load(REGISTRY)
    provenance = _load(PROVENANCE)
    freeze = _load(FREEZE)
    freeze_audit = _load(FREEZE_AUDIT)
    plan = _load(ROOT / "floorplansubmissiontodo.json")
    entries = {entry["key"]: entry for entry in registry["entries"]}
    errors_by_area = {
        "plan": _plan_checks(plan),
        "registry": _registry_checks(registry),
        "freeze": _freeze_checks(freeze, registry),
        "tables": _table_checks(entries),
        "figures": _figure_checks(),
        "texts": _text_checks(),
        "evidence": _evidence_checks(),
        "package": _package_checks(),
    }
    status_checks = {
        "registry_passed": registry.get("status") == "passed_scoped_release",
        "provenance_passed": provenance.get("status") == "passed",
        "freeze_audit_passed": freeze_audit.get("status") == "passed_scoped_release",
        "freeze_hash_current": freeze_audit.get("freeze_sha256") == _sha256(FREEZE),
        "optional_heldout_not_blocking": freeze.get("heldout", {}).get("blocks_scoped_submission") is False,
        "no_clean_worktree_requirement": freeze.get("clean_worktree_required") is False,
    }
    if not all(status_checks.values()):
        errors_by_area["statuses"] = [name for name, passed in status_checks.items() if not passed]
    errors = [error for area in errors_by_area.values() for error in area]
    report = {
        "schema_version": "floorplan-submission-integrity-v2",
        "status": "passed_scoped_release" if not errors else "failed",
        "scope": "existing oracle, predicted, operation-safety, and composition evidence",
        "checks": status_checks,
        "areas": {area: {"passed": not area_errors, "errors": area_errors} for area, area_errors in errors_by_area.items()},
        "optional_generalization_not_evaluated": {
            "status": "optional_generalization_not_evaluated",
            "required_for_scoped_submission": False,
            "report": str(HELDOUT.relative_to(ROOT)),
        },
        "boundary": (
            "This audit verifies the selected release and package identities. It does not open a new split, "
            "create an OpenReview submission, or issue a venue receipt."
        ),
        "errors": errors,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

