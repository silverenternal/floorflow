#!/usr/bin/env python3
"""Build the minimal anonymous FloorFlow CVPR release artifact."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/floorplan_submission"
REGISTRY = ROOT / "docs/floorplan_submission_source_registry_v2.json"
FREEZE = ROOT / "configs/floorplan_submission_evidence_freeze_v2.json"
PROVENANCE = ROOT / "outputs/reports/floorplan_source_provenance_audit_v2.json"
FREEZE_AUDIT = ROOT / "outputs/reports/floorplan_evidence_freeze_audit_v2.json"

ROOT_FILES = (".gitignore", ".python-version", "README.md", "pyproject.toml", "uv.lock")
SOURCE_GLOBS = (
    "src/floorflow/**/*.py",
    "scripts/audit_floorplan_evidence_freeze_v2.py",
    "scripts/audit_floorplan_submission_integrity_v1.py",
    "scripts/build_floorplan_matched_baselines_v1.py",
    "scripts/build_floorplan_submission_artifact_v1.py",
    "scripts/build_floorplan_submission_evidence_v2.py",
    "scripts/build_floorplan_submission_figures_v1.py",
    "scripts/build_floorplan_submission_tables_v2.py",
    "scripts/build_residential_edit_fiveseed_report_v1.py",
    "scripts/build_residential_sota_main_table_v2.py",
    "scripts/evaluate_predicted_operation_success_v1.py",
    "scripts/prepare_predicted_control_submission_freeze_v1.py",
    "tests/test_fevp_metric_contracts.py",
    "tests/test_fevp_protocol.py",
    "tests/test_floorplan_token_head.py",
    "tests/test_floorplan_token_training.py",
    "tests/test_fp_control_baselines.py",
    "tests/test_fp_control_corruption.py",
    "tests/test_layered_floorplan.py",
    "tests/test_predicted_control_edit.py",
    "tests/test_predicted_control_fidelity.py",
    "tests/test_predicted_control_reasonableness_contract.py",
    "tests/test_predicted_control_state.py",
    "tests/test_predicted_operation_success.py",
    "tests/test_unified_floorplan_generator.py",
    "tests/test_unified_floorplan_tokens.py",
    "configs/fevp_metric_contract_v2.json",
    "configs/floorplan_submission_evidence_freeze_v2.json",
    "configs/fp_control_contract_v1.json",
    "configs/predicted_control_reasonableness_contract_v1.json",
    "configs/predicted_operation_success_contract_v1.json",
    "docs/floorplan_*.md",
    "docs/fevp_claim_language_v1.md",
    "docs/fevp_final_claim_v4.md",
    "docs/fevp_formal_spec_v3.md",
    "docs/fevp_housemind_positioning_v2.md",
    "docs/fevp_metric_contracts_v2.md",
    "docs/fp_identity_metric_definition_v1.md",
    "docs/fp_oracle_control_protocol_status_v1.md",
    "docs/fp_room_graph_metric_protocol_v1.md",
    "docs/fp_strict_locality_protocol_v2.md",
    "docs/predicted_control_plausibility_v2.md",
    "docs/predicted_control_reasonableness_results_v1.md",
    "docs/predicted_control_zero_path_audit_v1.md",
    "paper/cvpr_claim_sections_v3.md",
    "paper/cvpr_experiment_structure_v1.md",
    "paper/cvpr_manuscript_skeleton_v1.md",
    "paper/cvpr.sty",
    "paper/ieeenat_fullname.bst",
    "paper/main.tex",
    "paper/supplement.tex",
    "paper/references.bib",
    "paper/writing_source_policy_v1.md",
    "paper/tables/main_*.md",
    "paper/tables/main_*.tex",
    "paper/tables/oracle_control_cvpr_dossier_v1.md",
    "paper/tables/predicted_control_reasonableness_dossier_v1.md",
    "paper/figures/floorplan_*.pdf",
    "paper/figures/floorplan_*.svg",
    "paper/figures/predicted_control_operation_panels_v1.pdf",
    "paper/figures/predicted_control_operation_panels_v1.svg",
    "paper/main.pdf",
    "paper/supplement.pdf",
)
EVIDENCE_FILES = (
    "docs/floorplan_submission_source_registry_v2.json",
    "outputs/reports/floorplan_source_provenance_audit_v2.json",
    "outputs/reports/floorplan_evidence_freeze_audit_v2.json",
    "outputs/reports/floorplan_heldout_availability_v1.json",
    "outputs/reports/floorplan_figure_case_manifest_v1.json",
    "outputs/reports/oracle_submission_statistics_v2.json",
    "outputs/reports/oracle_metric_trace_audit_20261102.json",
    "outputs/reports/predicted_operation_evidence_v2.json",
    "outputs/reports/matched_local_editing_baselines_v2.json",
    "outputs/reports/predicted_edit_object_delete_audit_v1.json",
    "outputs/reports/predicted_edit_ghost_state_audit_v1.json",
    "outputs/reports/fevp_budget_matched_control_v2.json",
    "outputs/reports/fevp_protocol_attacks_v2.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _registry_sources() -> list[Path]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [ROOT / entry["source_path"] for entry in registry["entries"]]


def _candidate_files() -> list[Path]:
    files = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for pattern in SOURCE_GLOBS:
        files.extend(ROOT.glob(pattern))
    files.extend(ROOT / name for name in EVIDENCE_FILES if (ROOT / name).is_file())
    files.extend(path for path in _registry_sources() if path.is_file())
    return sorted(
        {
            path
            for path in files
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo", ".npy", ".npz"}
            and not path.is_symlink()
        }
    )


def _submission_checklist(files: list[Path]) -> dict:
    registry_audit = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    freeze_audit = json.loads(FREEZE_AUDIT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    checks = {
        "official_cvpr_template": (ROOT / "paper/cvpr.sty").is_file(),
        "anonymous_author_block": "Anonymous CVPR submission" in (ROOT / "paper/main.tex").read_text(encoding="utf-8"),
        "anonymous_supplement": "Anonymous CVPR submission"
        in (ROOT / "paper/supplement.tex").read_text(encoding="utf-8"),
        "references_bib_present": (ROOT / "paper/references.bib").is_file(),
        "main_pdf_present": (ROOT / "paper/main.pdf").is_file(),
        "supplement_pdf_present": (ROOT / "paper/supplement.pdf").is_file(),
        "source_provenance_passed": registry_audit.get("status") == "passed",
        "evidence_freeze_passed": freeze_audit.get("status") == "passed_scoped_release",
        "optional_heldout_not_claimed": freeze.get("heldout", {}).get("status")
        == "optional_generalization_not_evaluated",
        "artifact_files_present": len(files) > 50,
        "no_heldout_result_fabricated": not (
            ROOT / "outputs/reports/residential_predicted_edit_heldout_v1.json"
        ).exists(),
    }
    return {
        "schema_version": "floorplan-submission-checklist-v1",
        "status": "ready_for_user_submission" if all(checks.values()) else "blocked_local_checks",
        "created_at": datetime.now(UTC).isoformat(),
        "checks": checks,
        "official_facts": {
            "venue": "CVPR 2027",
            "cfp": "https://cvpr.thecvf.com/Conferences/2027/CallForPapers",
            "author_guidelines": "https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines",
            "template": "https://github.com/cvpr-org/author-kit",
            "enrollment_deadline_aoe": "2026-11-10T23:59:59-12:00",
            "paper_deadline_aoe": "2026-11-16T23:59:59-12:00",
            "supplement_deadline_aoe": "2026-11-23T23:59:59-12:00",
            "main_page_limit_including_figures_tables": 8,
            "llm_usage_statement_required": True,
        },
        "external_actions_requiring_user_authorization": [
            "Confirm author eligibility and enroll the paper in OpenReview.",
            "Review the final PDF and enter the mandatory LLM Usage Statement.",
            "Submit the paper and supplement; this local workflow cannot create a submission receipt.",
        ],
        "boundary": "Local checks establish package consistency, not venue acceptance or an external submission receipt.",
    }


def main() -> None:
    (OUTPUT / "submission_checklist.json").unlink(missing_ok=True)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    files = _candidate_files()
    checklist = _submission_checklist(files)
    rows = []
    for source in files:
        relative = Path("PROJECT_README.md") if source == ROOT / "README.md" else source.relative_to(ROOT)
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append({"path": str(relative), "bytes": target.stat().st_size, "sha256": _sha256(target)})
    checklist_path = OUTPUT / "submission_checklist.json"
    checklist_path.write_text(json.dumps(checklist, indent=2) + "\n", encoding="utf-8")
    rows.append(
        {
            "path": "submission_checklist.json",
            "bytes": checklist_path.stat().st_size,
            "sha256": _sha256(checklist_path),
        }
    )
    readme = OUTPUT / "README.md"
    readme.write_text(
        "# FloorFlow Anonymous Review Artifact\n\n"
        "This package contains the selected source, tests, configurations, manuscript sources, tables, "
        "figures, and registry-bound evidence for the scoped floorplan submission. It excludes datasets, "
        "model checkpoints, raw arrays, third-party source trees, caches, and unselected research outputs.\n\n"
        "The evidence registry is `docs/floorplan_submission_source_registry_v2.json`. The selected release "
        "is frozen by file identity in `configs/floorplan_submission_evidence_freeze_v2.json`. Oracle, "
        "predicted, safety-contract, and semantic-diagnostic populations are not pooled. Fresh held-out "
        "generalization is explicitly not evaluated.\n\n"
        "See `artifact_manifest.json` for file sizes and SHA-256 hashes and `CHECKSUMS.sha256` for a "
        "standard verification list. The local `submission_checklist.json` distinguishes completed checks "
        "from external OpenReview actions that require the authors.\n",
        encoding="utf-8",
    )
    rows.append({"path": "README.md", "bytes": readme.stat().st_size, "sha256": _sha256(readme)})
    manifest = {
        "schema_version": "floorplan-submission-artifact-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": _commit(),
        "status": checklist["status"],
        "scope": "scoped FloorFlow oracle/predicted floorplan-editing evidence release",
        "anonymous": True,
        "contains_datasets_or_checkpoints": False,
        "registry": "docs/floorplan_submission_source_registry_v2.json",
        "freeze": "configs/floorplan_submission_evidence_freeze_v2.json",
        "files": rows,
    }
    (OUTPUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "CHECKSUMS.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "files": len(rows), "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()

