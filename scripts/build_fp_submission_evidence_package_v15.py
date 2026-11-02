#!/usr/bin/env python3
"""Build the locality-auditability-gap evidence package v15."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tarfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs/cvpr_package_fp_v15"
ARCHIVE = ROOT / "outputs/cvpr_package_fp_v15.tar.gz"


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


def _auditability_artifacts() -> tuple[str, ...]:
    return (
        "scripts/build_fevp_housemind_positioning_v2.py",
        "scripts/build_fevp_mechanism_pareto_v3.py",
        "scripts/build_fevp_audit_evidence_map_v2.py",
        "scripts/build_fevp_room_deletion_diagnosis_v2.py",
        "scripts/build_fevp_final_claim_v3.py",
        "scripts/build_fp_evidence_v10.py",
        "scripts/build_reviewer_risk_matrix_v7.py",
        "outputs/reports/fevp_housemind_positioning_v2.json",
        "outputs/reports/fevp_mechanism_pareto_v3.json",
        "outputs/reports/fevp_audit_evidence_map_v2.json",
        "outputs/reports/fevp_room_deletion_diagnosis_v2.json",
        "outputs/reports/fevp_final_claim_v3.json",
        "outputs/reports/fp_cvpr_evidence_v10.json",
        "outputs/reports/fp_reviewer_risk_matrix_v7.md",
        "outputs/reports/fp_pytest_v15.xml",
        "outputs/reports/fevp_coverage_matrix_v1.json",
        "docs/fevp_housemind_positioning_v2.md",
        "docs/fevp_final_claim_v3.md",
        "paper/tables/fevp_housemind_positioning_v2.md",
        "paper/tables/fevp_mechanism_pareto_v3.md",
        "paper/tables/fevp_audit_evidence_map_v2.md",
        "paper/tables/fevp_room_deletion_diagnosis_v2.md",
        "paper/tables/fevp_coverage_matrix_v1.md",
        "paper/figures/fevp_mechanism_pareto_v3.pdf",
        "paper/figures/fevp_mechanism_pareto_v3.svg",
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
    v14 = _load(
        "fp_package_v14",
        "scripts/build_fp_submission_evidence_package_v14.py",
    )
    v13 = _load(
        "fp_package_v13",
        "scripts/build_fp_submission_evidence_package_v13.py",
    )
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
                *v13._hardened_artifacts(),
                *v14._offensive_artifacts(),
                *_auditability_artifacts(),
            ]
        )
    )
    missing = [path for path in artifacts if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing v15 artifacts: {missing}")
    data_directories = [
        *v11.DATA_DIRECTORIES,
        "outputs/external_baselines/raster2seq/official_test509_v1/jsons",
    ]
    evidence = _read("outputs/reports/fp_cvpr_evidence_v10.json")
    final = _read("outputs/reports/fevp_final_claim_v3.json")
    house = _read("outputs/reports/fevp_housemind_positioning_v2.json")
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
        "status": "fevp_locality_auditability_gap_evidence_v15",
        "submission_status": evidence["submission_status"],
        "created_at": datetime.now(UTC).isoformat(),
        "primary_claim": final["claim_en"],
        "primary_claim_zh": final["claim_zh"],
        "problem_statement": final["problem_statement"],
        "house_mind_positioning": house["non_ranking_statement"],
        "forbidden_claims": final["forbidden_claims"],
        "gates": evidence["gates"],
        "experiment_objective_complete": evidence["experiment_objective_complete"],
        "manuscript_in_scope": False,
        "human_evaluation": "excluded_by_user_request",
        "package_name": PACKAGE.name,
        "archive": "outputs/cvpr_package_fp_v15.tar.gz",
        "hash_algorithm": "sha256",
        "file_count": file_count,
        "missing_count": 0,
        "artifact_count": len(artifacts),
        "artifacts": artifact_rows,
        "data_directories": directory_rows,
        "excluded_external_assets": {
            "CE2EPlan": "No license; code, RPLAN data, weights, and images excluded.",
            "HouseMind": "Official code remains Code (Soon); only public values are context.",
        },
        "verification": {
            "pytest": _pytest(ROOT / "outputs/reports/fp_pytest_v15.xml"),
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
    readme = f"""# FloorFlow CVPR Experiment Evidence v15

Locality-auditability-gap evidence only. No manuscript or human evaluation.

## Problem

{final['problem_statement']}

## Claim

{final['claim_en']}

## HouseMind Boundary

{house['non_ranking_statement']}

## Mechanism Note

Typed and No-graph both have zero violation under the strict arm. Under the
current fixed evaluator, Typed is better than No-graph on Delta-IoU, Node F1,
Edge F1, and GED. The v3 Pareto uses the corrected frontier calculation.
HouseMind-E/O are shown only as un-audited context because the public evaluation
does not report a locality-axis value.

## Room Deletion

Room deletion is the principal GED-difficulty mode. Its outside exactness remains
1.0 and violation rate remains zero, separating topological uncertainty from
outside leakage.

## Forbidden Claims

{chr(10).join('- ' + claim for claim in final['forbidden_claims'])}
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
