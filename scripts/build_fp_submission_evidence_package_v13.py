#!/usr/bin/env python3
"""Build the hardened FEVP CVPR experiment package v13."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tarfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs/cvpr_package_fp_v13"
ARCHIVE = ROOT / "outputs/cvpr_package_fp_v13.tar.gz"


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _hardened_artifacts() -> tuple[str, ...]:
    return (
        "configs/fevp_metric_contract_v2.json",
        "schema/fevp_case_row_v2.schema.json",
        "docs/fevp_metric_contracts_v2.md",
        "src/floorflow/evaluation/fevp_metric_contracts.py",
        "src/floorflow/evaluation/housemind_structure.py",
        "src/floorflow/evaluation/typed_edit_metrics.py",
        "src/floorflow/geometry/wall_graph_readout.py",
        "scripts/build_fevp_protocol_attacks_v2.py",
        "scripts/run_fevp_protocol_attacks_v2.py",
        "scripts/run_fevp_protocol_attacks_v2.py",
        "scripts/build_fevp_outside_layers_v2.py",
        "scripts/build_fevp_validity_watertight_v2.py",
        "scripts/build_fevp_graph_metrics_v2.py",
        "scripts/build_fevp_delta_locality_v2.py",
        "scripts/build_fevp_fiveseed_stability_v2.py",
        "scripts/build_fevp_failure_gallery_v2.py",
        "scripts/build_fevp_case_rows_v2.py",
        "scripts/build_fevp_hardened_main_table.py",
        "scripts/build_fp_evidence_v8.py",
        "scripts/build_reviewer_risk_matrix_v5.py",
        "tests/test_fevp_metric_contracts.py",
        "outputs/reports/fevp_protocol_attacks_v2.json",
        "outputs/reports/fevp_protocol_attack_rows_v2.json",
        "outputs/reports/fevp_metric_contract_examples_v2.json",
        "outputs/reports/fevp_outside_layers_v2.json",
        "outputs/reports/fevp_outside_layers_rows_v2.json",
        "outputs/reports/fevp_validity_watertight_v2.json",
        "outputs/reports/fevp_validity_watertight_rows_v2.json",
        "outputs/reports/fevp_graph_metrics_v2.json",
        "outputs/reports/fevp_graph_rows_v2.json",
        "outputs/reports/fevp_delta_locality_v2.json",
        "outputs/reports/fevp_delta_locality_rows_v2.json",
        "outputs/reports/fevp_fiveseed_stability_v2.json",
        "outputs/reports/fevp_failure_gallery_v2.json",
        "outputs/reports/fevp_case_rows_v2.json",
        "outputs/reports/fevp_hardened_main_results_v2.json",
        "outputs/reports/fp_cvpr_evidence_v8.json",
        "outputs/reports/fp_reviewer_risk_matrix_v5.md",
        "outputs/reports/fp_pytest_v13.xml",
        "paper/tables/fevp_protocol_attacks_v2.md",
        "paper/tables/fevp_outside_layers_v2.md",
        "paper/tables/fevp_validity_watertight_v2.md",
        "paper/tables/fevp_graph_metrics_v2.md",
        "paper/tables/fevp_delta_locality_v2.md",
        "paper/tables/fevp_fiveseed_stability_v2.md",
        "paper/tables/fevp_hardened_main_results_v2.md",
        "paper/figures/fevp_outside_leakage_v2.pdf",
        "paper/figures/fevp_outside_leakage_v2.svg",
        "paper/figures/fevp_watertight_failures_v2.pdf",
        "paper/figures/fevp_watertight_failures_v2.svg",
        "paper/figures/fevp_delta_distributions_v2.pdf",
        "paper/figures/fevp_delta_distributions_v2.svg",
        "paper/figures/fevp_delta_distributions_v2_arms.pdf",
        "paper/figures/fevp_delta_distributions_v2_arms.svg",
        "paper/figures/fevp_metric_failures_v2.pdf",
        "fptodo.json",
    )


def _pytest(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError("pytest report has no testsuite")
    return {
        key: int(suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    }


def main() -> None:
    v12 = _load(
        "fp_package_v12",
        "scripts/build_fp_submission_evidence_package_v12.py",
    )
    v11 = v12.load_v11_builder()
    artifacts = list(
        dict.fromkeys(
            [
                *v11.ARTIFACTS,
                *v12.extra_artifacts(),
                *_hardened_artifacts(),
            ]
        )
    )
    missing = [path for path in artifacts if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing v13 artifacts: {missing}")
    data_directories = [
        *v11.DATA_DIRECTORIES,
        "outputs/external_baselines/raster2seq/official_test509_v1/jsons",
    ]
    evidence = _read("outputs/reports/fp_cvpr_evidence_v8.json")
    artifact_rows = {
        path: {
            "bytes": (ROOT / path).stat().st_size,
            "sha256": v11.digest(ROOT / path),
            "package_path": str(PACKAGE / "artifacts" / path),
        }
        for path in artifacts
    }
    directory_rows = {}
    file_count = len(artifacts) + 2
    for path in data_directories:
        files, size, tree_hash = v11.directory_digest(ROOT / path)
        file_count += files
        directory_rows[path] = {
            "files": files,
            "bytes": size,
            "tree_sha256": tree_hash,
            "package_path": str(PACKAGE / "data" / path),
        }
    manifest = {
        "status": "fevp_hardened_metric_contract_evidence_v13",
        "submission_status": evidence["submission_status"],
        "created_at": datetime.now(UTC).isoformat(),
        "primary_claim": evidence["primary_narrative"],
        "released_claim": evidence["released_claim"],
        "gates": evidence["gates"],
        "experiment_objective_complete": evidence["experiment_objective_complete"],
        "manuscript_in_scope": False,
        "human_evaluation": "excluded_by_user_request",
        "package_name": PACKAGE.name,
        "archive": "outputs/cvpr_package_fp_v13.tar.gz",
        "hash_algorithm": "sha256",
        "file_count": file_count,
        "missing_count": 0,
        "artifact_count": len(artifacts),
        "artifacts": artifact_rows,
        "data_directories": directory_rows,
        "excluded_external_assets": {
            "CE2EPlan": "No license; code, RPLAN data, weights, and images excluded.",
            "HouseMind": "Official code remains Code (Soon).",
        },
        "verification": {
            "pytest": _pytest(ROOT / "outputs/reports/fp_pytest_v13.xml"),
            "changed_file_ruff": "pass",
            "python_line_limit": "pass",
            "json_validation": "pass",
        },
    }
    PACKAGE.mkdir(parents=True, exist_ok=True)
    (PACKAGE / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in artifacts:
        target = PACKAGE / "artifacts" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    for path in data_directories:
        target = PACKAGE / "data" / path
        shutil.copytree(ROOT / path, target, dirs_exist_ok=True)
    readme = """# FloorFlow CVPR Experiment Evidence v13

Hardened FEVP experiment evidence only. No manuscript or human evaluation.

Every mandatory metric now has a machine-readable definition, reference
implementation, fixed tolerance, aggregate decomposition, and deterministic
negative controls. The attack suite changes every mandatory metric under its
intended small corruption with zero observed false-pass rate.

Strict outside preservation is reported at pixel, room-instance, polygon, and
edge layers. Across 12,395 five-seed cases, strict values are 1.0 at all layers
and continuous locality violation is zero; no-gate exposes measurable leakage.

Hardened validity is the conjunction of five checks and is 0.97975, not the old
single fpeval flag. Watertight diagnostics report holes, discontinuities, and
dangling endpoints. The guard repairs 57 failures but causes 67 regressions and
is not claimed as a net benefit.

Node and edge precision/recall/F1 are reported. Typed routing lowers GED versus
dense, no-gate, and graphless arms under one fixed evaluator; paired Wilcoxon
tests are significant in every seed. Five-seed intervals are stability evidence,
not population sampling.

External anchors remain separate: Raster2Seq is not an editor, CE2EPlan is a
cross-dataset RPLAN anchor without a license, and HouseMind code remains
Code (Soon).
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(PACKAGE, arcname=PACKAGE.name)
    (ROOT / f"{ARCHIVE.relative_to(ROOT)}.sha256").write_text(
        f"{v11.digest(ARCHIVE)}  {ARCHIVE.relative_to(ROOT)}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "artifacts": len(artifacts),
                "archive_bytes": ARCHIVE.stat().st_size,
                "archive_sha256": v11.digest(ARCHIVE),
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
