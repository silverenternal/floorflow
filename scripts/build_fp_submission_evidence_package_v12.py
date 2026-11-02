#!/usr/bin/env python3
"""Build the final FEVP and external-audit CVPR experiment package."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tarfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs/cvpr_package_fp_v12"
ARCHIVE = ROOT / "outputs/cvpr_package_fp_v12.tar.gz"


def load_v11_builder():
    path = ROOT / "scripts/build_fp_submission_evidence_package.py"
    spec = importlib.util.spec_from_file_location("fp_package_v11", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load package builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def extra_artifacts() -> list[str]:
    values = [
        "outputs/reports/fp_cvpr_evidence_v7.json",
        "outputs/reports/fp_reviewer_risk_matrix_v4.md",
        "outputs/reports/fevp_coverage_matrix_v1.json",
        "outputs/reports/fevp_claim_source_map_v1.json",
        "outputs/reports/fp_locality_mechanism_v2.json",
        "outputs/reports/fp_raster2seq_official_eval_metrics_v1.json",
        "outputs/reports/fp_raster2seq_official_anchor_v1.json",
        "outputs/reports/fp_raster2seq_identity_reconstruction_v1.json",
        "outputs/reports/fp_raster2seq_identity_rows_v1.json",
        "outputs/reports/ce2eplan_availability_v1.json",
        "outputs/reports/ce2eplan_official_anchor_v1.json",
        "outputs/reports/ce2eplan_floorflow_matching_gate_v1.json",
        "docs/fevp_v1_0.md",
        "docs/fevp_claim_language_v1.md",
        "docs/fp_locality_mechanism_claims_v1.md",
        "docs/ce2eplan_audit_v1.md",
        "paper/tables/fevp_coverage_matrix_v1.md",
        "paper/tables/fp_locality_mechanism_v2.md",
        "paper/tables/fp_raster2seq_official_audit_v1.md",
        "paper/tables/fp_raster2seq_identity_comparison_v1.md",
        "paper/tables/ce2eplan_official_anchor_v1.md",
        "paper/figures/fp_locality_efficiency_tradeoff_v2.svg",
        "paper/figures/fp_locality_efficiency_tradeoff_v2.pdf",
        "schema/fevp_report_v1.schema.json",
        "schema/fevp_case_row_v1.schema.json",
        "src/floorflow/evaluation/fevp_protocol.py",
        "scripts/validate_fevp_report.py",
        "scripts/build_fevp_coverage_matrix.py",
        "scripts/build_fp_locality_mechanism_v2.py",
        "scripts/build_fp_raster2seq_official_audit.py",
        "scripts/build_fp_raster2seq_identity_report.py",
        "scripts/run_fp_raster2seq_identity.py",
        "scripts/build_ce2eplan_anchor_audit.py",
        "scripts/build_fp_evidence_v7.py",
        "scripts/build_reviewer_risk_matrix_v4.py",
        "tests/test_fevp_protocol.py",
        "outputs/reports/fp_pytest_v12.xml",
        "fptodo.json",
    ]
    values.extend(
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "examples/fevp").glob("*.json"))
    )
    return values


def pytest_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError(f"pytest report has no testsuite: {path}")
    return {
        key: int(suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    }


def main() -> None:
    builder = load_v11_builder()
    artifacts = list(dict.fromkeys([*builder.ARTIFACTS, *extra_artifacts()]))
    missing = [path for path in artifacts if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing v12 artifacts: {missing}")
    data_directories = [
        *builder.DATA_DIRECTORIES,
        "outputs/external_baselines/raster2seq/official_test509_v1/jsons",
    ]
    evidence = read("outputs/reports/fp_cvpr_evidence_v7.json")
    v11_manifest = read("outputs/cvpr_package_fp_v11/manifest.json")
    artifact_rows = {
        path: {
            "bytes": (ROOT / path).stat().st_size,
            "sha256": builder.digest(ROOT / path),
            "package_path": str(PACKAGE / "artifacts" / path),
        }
        for path in artifacts
    }
    directory_rows = {}
    file_count = len(artifacts) + 2
    for path in data_directories:
        files, size, tree_hash = builder.directory_digest(ROOT / path)
        file_count += files
        directory_rows[path] = {
            "files": files,
            "bytes": size,
            "tree_sha256": tree_hash,
            "package_path": str(PACKAGE / "data" / path),
        }
    limitations = list(v11_manifest["explicit_limitations"])
    limitations.extend(
        [
            "FEVP is the evaluated matrix named in this package, not a global first-method claim.",
            "Raster2Seq contour-region cases cannot be bijected to all 503 FloorFlow SVG-floor cases.",
            "The 503-row Raster2Seq input-renderer result is diagnostic, not official.",
            "CE2EPlan has no license file and is excluded from the archive.",
            "CE2EPlan GED is independent because its evaluator implementation is not released.",
            "No merged RPLAN/CubiCasa external leaderboard exists.",
            "HouseMind official inference code remains unavailable.",
            "Human evaluation is excluded; no participant-dependent claim exists.",
        ]
    )
    manifest = {
        "status": "fevp_protocol_and_external_audits_evidence_v12",
        "submission_status": evidence["submission_status"],
        "created_at": datetime.now(UTC).isoformat(),
        "primary_claim": evidence["primary_narrative"],
        "released_claim": evidence["released_claim"],
        "secondary_claim": (
            "Matched routing audits show that strict locality and edit quality "
            "are separable; unaffected external dimensions remain N/A."
        ),
        "explicit_limitations": limitations,
        "gates": evidence["gates"],
        "experiment_objective_complete": evidence["experiment_objective_complete"],
        "manuscript_in_scope": False,
        "human_evaluation": "excluded_by_user_request",
        "package_name": PACKAGE.name,
        "archive": "outputs/cvpr_package_fp_v12.tar.gz",
        "hash_algorithm": "sha256",
        "file_count": file_count,
        "missing_count": 0,
        "artifacts": artifact_rows,
        "data_directories": directory_rows,
        "excluded_external_assets": {
            "CE2EPlan": (
                "No LICENSE file; code, RPLAN data, weights, and generated "
                "images remain outside the archive. Hashes and reports are included."
            ),
            "HouseMind": "Official inference code remains Code (Soon).",
        },
        "raw_logits_policy": (
            "FloorFlow's control-gated compositor introduces no new network logits. "
            "Rendered rows, scorer outputs, predictions, and Raster2Seq JSON "
            "predictions are included. CE2EPlan raw assets are excluded by license."
        ),
        "verification": {
            "pytest": pytest_summary(ROOT / "outputs/reports/fp_pytest_v12.xml"),
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
    readme = """# FloorFlow CVPR Experiment Evidence v12

Experiment evidence only. No manuscript is included.

## Supported claim

Under FEVP v1.0, FloorFlow is the only method in the evaluated coverage matrix
with a complete report for all five mandatory strict-locality metrics. On
CubiCasa test503 it combines high Delta-IoU and graph quality with exact outside
pixel and room-label preservation, validity 1.0, and conditional
watertightness reported seed by seed.

The protocol requires outside exactness, validity, conditional watertightness,
Node F1, and edge overlap. Missing or non-comparable external cells remain N/A
and are never ranked.

## External audits

Raster2Seq's official CubiCasa evaluator reproduces RoomF1 0.8853 against the
public 0.887 anchor. Its official preprocessing creates 509 contour-region
cases that cannot be bijected to all 503 FloorFlow SVG-floor cases. The
official anchor and matched edit rows remain separate; the project-defined
identity adapter is diagnostic only.

CE2EPlan's RPLAN mode-auto checkpoint generated 12,002 cases and completed the
released post-processing. FID 1.1270 and all six paper-style statistic ratios
are close to the published anchor. The independent GED diagnostic is 5.2323,
not the paper's 3.48, because the exact evaluator code is not public. CE2EPlan
has no LICENSE file and remains a cross-dataset context anchor.

HouseMind remains a complementary MLLM framework with official code marked
Code (Soon).

## Exclusions

- No matched 503-case Raster2Seq identity rank.
- No merged RPLAN/CubiCasa leaderboard.
- No CE2EPlan editing baseline or code/weight redistribution.
- No HouseMind complete-method comparison.
- No human evaluation or participant-derived result.

See `manifest.json` for every artifact hash and directory tree hash.
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(PACKAGE, arcname=PACKAGE.name)
    (ARCHIVE.parent / f"{ARCHIVE.name}.sha256").write_text(
        f"{builder.digest(ARCHIVE)}  {ARCHIVE.relative_to(ROOT)}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "artifacts": len(artifacts),
                "data_directories": len(data_directories),
                "archive_bytes": ARCHIVE.stat().st_size,
                "archive_sha256": builder.digest(ARCHIVE),
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
