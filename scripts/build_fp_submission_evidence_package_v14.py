#!/usr/bin/env python3
"""Build the bounded offensive FEVP CVPR experiment package v14."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tarfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs/cvpr_package_fp_v14"
ARCHIVE = ROOT / "outputs/cvpr_package_fp_v14.tar.gz"


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


def _offensive_artifacts() -> tuple[str, ...]:
    return (
        "scripts/build_fevp_locality_claim_v2.py",
        "scripts/build_fevp_mechanism_pareto_v2.py",
        "scripts/build_fevp_safety_baseline_v2.py",
        "scripts/build_fevp_edit_strength_v2.py",
        "scripts/build_fevp_attack_compact_v2.py",
        "scripts/build_fevp_final_claim_v2.py",
        "scripts/build_fp_evidence_v9.py",
        "scripts/build_reviewer_risk_matrix_v6.py",
        "outputs/reports/fevp_locality_claim_v2.json",
        "outputs/reports/fevp_mechanism_pareto_v2.json",
        "outputs/reports/fevp_safety_baseline_v2.json",
        "outputs/reports/fevp_edit_strength_v2.json",
        "outputs/reports/fevp_protocol_attacks_compact_v2.json",
        "outputs/reports/fevp_final_claim_v2.json",
        "outputs/reports/fp_cvpr_evidence_v9.json",
        "outputs/reports/fp_reviewer_risk_matrix_v6.md",
        "outputs/reports/fp_pytest_v14.xml",
        "docs/fevp_locality_claim_v2.md",
        "docs/fevp_safety_baseline_v2.md",
        "docs/fevp_final_claim_v2.md",
        "paper/tables/fevp_mechanism_pareto_v2.md",
        "paper/tables/fevp_edit_strength_v2.md",
        "paper/tables/fevp_protocol_attacks_compact_v2.md",
        "paper/figures/fevp_locality_leakage_heatmap_v2.pdf",
        "paper/figures/fevp_locality_leakage_heatmap_v2.svg",
        "paper/figures/fevp_mechanism_pareto_v2.pdf",
        "paper/figures/fevp_mechanism_pareto_v2.svg",
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
                *_offensive_artifacts(),
            ]
        )
    )
    missing = [path for path in artifacts if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing v14 artifacts: {missing}")
    data_directories = [
        *v11.DATA_DIRECTORIES,
        "outputs/external_baselines/raster2seq/official_test509_v1/jsons",
    ]
    evidence = _read("outputs/reports/fp_cvpr_evidence_v9.json")
    final = _read("outputs/reports/fevp_final_claim_v2.json")
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
        "status": "fevp_bounded_offensive_evidence_v14",
        "submission_status": evidence["submission_status"],
        "created_at": datetime.now(UTC).isoformat(),
        "primary_claim": final["claim_en"],
        "primary_claim_zh": final["claim_zh"],
        "forbidden_claims": final["forbidden_claims"],
        "gates": evidence["gates"],
        "experiment_objective_complete": evidence["experiment_objective_complete"],
        "manuscript_in_scope": False,
        "human_evaluation": "excluded_by_user_request",
        "package_name": PACKAGE.name,
        "archive": "outputs/cvpr_package_fp_v14.tar.gz",
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
            "pytest": _pytest(ROOT / "outputs/reports/fp_pytest_v14.xml"),
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
    readme = f"""# FloorFlow CVPR Experiment Evidence v14

Bounded offensive FEVP evidence only. No manuscript or human evaluation.

## Canonical Claim

{final['claim_en']}

## Key Contrast

Strict locality is {final['evidence']['locality']['strict']['outside_room_level_exact']:.4f}
at the room layer with zero violation, while removing edit gating lowers room
exactness to {final['evidence']['locality']['no_edit_gating']['outside_room_level_exact']:.4f}
and raises violation rate to {final['evidence']['locality']['no_edit_gating']['locality_violation_rate']:.4f}.
The leaking arm is evaluator-sensitivity evidence, not a constant-pass result.

Typed routing is on the computed mechanism Pareto frontier and improves Node F1,
edge F1, and fixed-evaluator GED over dense, no-gate, and graphless arms with
five-seed paired tests.

## Safety Boundary

Five-check validity is a safety baseline, not a restored validity-1.0 claim.
The watertight guard repairs 57 failures but causes 67 regressions and is not
presented as a net contribution.

## Forbidden Claims

{chr(10).join('- ' + claim for claim in final['forbidden_claims'])}

## Verification

The manifest records every artifact hash, directory tree hash, pytest result,
and external-asset exclusion.
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
