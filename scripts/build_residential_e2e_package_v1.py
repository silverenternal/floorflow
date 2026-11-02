"""Build and audit the deterministic residential E2E evidence package."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "outputs/residential_e2e_evidence_v1"
ARCHIVE = ROOT / "outputs/residential_e2e_evidence_v1.tar.gz"
AUDIT = ROOT / "outputs/reports/residential_e2e_package_audit_v1.json"

ARTIFACTS = (
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "fptodo.json",
    "configs/residential_e2e_benchmark_v1.json",
    "configs/residential_constraint_flow_v1.json",
    "configs/residential_graph_refinement_v1.json",
    "configs/residential_e2e_frontier_v1.json",
    "configs/residential_predicted_edit_v1.json",
    "configs/residential_external_comparison_v1.json",
    "configs/residential_e2e_multiseed_v1.json",
    "src/floorflow/data/model_candidate_dataset.py",
    "src/floorflow/data/interaction_candidate_dataset.py",
    "src/floorflow/data/interaction_context.py",
    "src/floorflow/data/constraint_action_dataset.py",
    "src/floorflow/data/constraint_candidates.py",
    "src/floorflow/inference/strict_local_projection.py",
    "src/floorflow/inference/local_delta.py",
    "src/floorflow/inference/constraint_action_runtime.py",
    "src/floorflow/inference/room_graph_refinement.py",
    "src/floorflow/inference/predicted_control_edit.py",
    "src/floorflow/inference/predicted_refiner_runtime.py",
    "src/floorflow/inference/cached_contour_controls.py",
    "src/floorflow/model/constraint_aware_gate.py",
    "src/floorflow/model/constraint_state_encoder.py",
    "src/floorflow/model/interaction_constraint_gate.py",
    "src/floorflow/training/interaction_policy_data.py",
    "src/floorflow/objectives/constraint_policy.py",
    "src/floorflow/objectives/constraint_fusion.py",
    "src/floorflow/evaluation/e2e_benchmark_contract.py",
    "src/floorflow/evaluation/residential_graph_frontier.py",
    "src/floorflow/evaluation/external_claim_matrix.py",
    "scripts/audit_residential_e2e_benchmarks.py",
    "scripts/build_model_candidate_corpus_v1.py",
    "scripts/audit_model_candidate_corpus_v1.py",
    "scripts/build_residential_interaction_corpus.py",
    "scripts/enrich_residential_interaction_context.py",
    "scripts/audit_residential_interaction_corpus.py",
    "scripts/train_residential_interaction_policy.py",
    "scripts/train_residential_constraint_flow.py",
    "scripts/build_residential_constraint_flow_summary.py",
    "scripts/evaluate_residential_graph_refinement.py",
    "scripts/confirm_residential_graph_ged_exact.py",
    "scripts/evaluate_residential_e2e_frontier.py",
    "scripts/build_predicted_instruction_controls.py",
    "scripts/train_residential_predicted_refiner.py",
    "scripts/materialize_residential_predicted_refiner.py",
    "scripts/evaluate_residential_predicted_edit_v1.py",
    "scripts/run_fp_raster2seq_identity.py",
    "scripts/build_residential_external_comparison_v1.py",
    "scripts/build_residential_e2e_multiseed_report.py",
    "scripts/benchmark_residential_e2e_v1.py",
    "scripts/freeze_residential_e2e_v1.py",
    "scripts/build_residential_e2e_test_final.py",
    "tests/test_residential_e2e_benchmark_contract.py",
    "tests/test_model_candidate_dataset.py",
    "tests/test_strict_local_projection.py",
    "tests/test_constraint_action_runtime.py",
    "tests/test_constraint_policy.py",
    "tests/test_interaction_context.py",
    "tests/test_interaction_constraint_gate.py",
    "tests/test_constraint_fusion.py",
    "tests/test_room_graph_refinement.py",
    "tests/test_residential_e2e_frontier.py",
    "tests/test_predicted_control_edit.py",
    "tests/test_residential_predicted_refiner.py",
    "tests/test_external_claim_matrix.py",
    "tests/test_residential_e2e_statistics.py",
    "outputs/reports/residential_e2e_baseline_audit_v1.json",
    "outputs/reports/model_candidate_corpus_audit_v1.json",
    "outputs/reports/zero_tolerance_locality_audit_v2.json",
    "outputs/reports/residential_interaction_corpus_audit_v1.json",
    "outputs/reports/residential_interaction_policy_v11.json",
    "outputs/reports/residential_constraint_flow_summary_v1.json",
    "outputs/reports/residential_graph_refinement_exact_v1.json",
    "outputs/reports/residential_graph_refinement_exact_rows_v1.json",
    "outputs/reports/residential_e2e_frontier_v1.json",
    "outputs/reports/residential_predicted_refiner_v1.json",
    "outputs/reports/residential_predicted_edit_v1.json",
    "outputs/reports/residential_instruction_robustness_v1.json",
    "outputs/reports/residential_external_comparison_v1.json",
    "outputs/reports/residential_external_comparison_rows_v1.json",
    "outputs/reports/residential_e2e_runtime_benchmark_v1.json",
    "outputs/reports/residential_e2e_multiseed_v1.json",
    "outputs/reports/residential_e2e_validation_freeze_v1.json",
    "outputs/reports/residential_external_comparison_test503_final_v1.json",
    "outputs/reports/residential_external_comparison_test503_rows_v1.json",
    "outputs/reports/residential_predicted_edit_test503_final_v1.json",
    "outputs/reports/residential_instruction_robustness_test503_final_v1.json",
    "outputs/reports/residential_e2e_test503_final_v1.json",
    "outputs/reports/residential_e2e_claim_gate_v1.json",
    "paper/figures/residential_e2e_frontier_v1.pdf",
    "paper/figures/residential_e2e_frontier_v1.svg",
    "paper/figures/residential_e2e_frontier_v1.caption.txt",
    "paper/tables/residential_graph_refinement_final_val_v2.md",
    "paper/tables/residential_external_claims_v1.md",
    "paper/tables/residential_external_claims_test503_final_v1.md",
    "paper/tables/residential_e2e_multiseed_v1.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_files() -> list[str]:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    files = []
    for relative in ARTIFACTS:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(relative)
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files.append(relative)
    return sorted(files)


def _manifest(files: list[str]) -> None:
    payload = {
        "schema_version": "residential-e2e-package-manifest-v1",
        "files": [
            {
                "path": relative,
                "sha256": _sha256(STAGE / relative),
                "bytes": (STAGE / relative).stat().st_size,
            }
            for relative in files
        ],
    }
    (STAGE / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tar(files: list[str]) -> None:
    names = sorted(files + ["manifest.json"])
    with io.BytesIO() as raw:
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name in names:
                path = STAGE / name
                info = tar.gettarinfo(str(path), arcname=name)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mode = 0o644
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
        raw.seek(0)
        with (
            ARCHIVE.open("wb") as output,
            gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed,
        ):
            shutil.copyfileobj(raw, compressed)


def main() -> None:
    files = _stage_files()
    _manifest(files)
    _tar(files)
    python_files = [path for path in files if path.endswith(".py")]
    line_counts = {
        path: len((STAGE / path).read_text(encoding="utf-8").splitlines())
        for path in python_files
    }
    forbidden = [
        path
        for path in files
        if "chip" in path.lower() or "floorset" in path.lower()
    ]
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        members = sorted(
            member.name
            for member in archive.getmembers()
            if member.isfile()
        )
    expected = sorted(files + ["manifest.json"])
    checks = {
        "archive_members_match_manifest": members == expected,
        "all_python_under_600_lines": all(
            count < 600 for count in line_counts.values()
        ),
        "no_chip_or_floorset_artifacts": not forbidden,
        "test_and_claim_gate_present": (
            "outputs/reports/residential_e2e_test503_final_v1.json" in files
            and "outputs/reports/residential_e2e_claim_gate_v1.json" in files
        ),
    }
    audit = {
        "schema_version": "residential-e2e-package-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256": _sha256(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "file_count": len(files),
        "tar_member_count": len(members),
        "line_limit": 600,
        "line_counts": line_counts,
        "forbidden_paths": forbidden,
        "checks": checks,
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
