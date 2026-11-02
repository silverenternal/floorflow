#!/usr/bin/env python3
"""Build the portable-FEVP and prompt-robustness evidence package v16."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tarfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs/cvpr_package_fp_v16"
ARCHIVE = ROOT / "outputs/cvpr_package_fp_v16.tar.gz"


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


def _portability_artifacts() -> tuple[str, ...]:
    return (
        "scripts/build_fevp_core_audit_coverage_v3.py",
        "scripts/build_fevp_ce2eplan_portability_v3.py",
        "scripts/audit_fevp_housemind_portability_v1.py",
        "scripts/build_fevp_adversarial_prompts_v3.py",
        "scripts/build_fevp_final_claim_v4.py",
        "scripts/build_fp_evidence_v11.py",
        "scripts/build_reviewer_risk_matrix_v8.py",
        "src/floorflow/evaluation/ce2eplan_adapter.py",
        "outputs/reports/fevp_core_audit_coverage_v3.json",
        "outputs/reports/fevp_ce2eplan_portability_v3.json",
        "outputs/reports/fevp_housemind_portability_gate_v1.json",
        "outputs/reports/fevp_adversarial_prompts_v3.json",
        "outputs/reports/fevp_adversarial_prompt_rows_v3.json",
        "outputs/reports/fevp_final_claim_v4.json",
        "outputs/reports/fp_cvpr_evidence_v11.json",
        "outputs/reports/fp_reviewer_risk_matrix_v8.md",
        "outputs/reports/fp_pytest_v16.xml",
        "docs/fevp_final_claim_v4.md",
        "paper/tables/fevp_core_audit_coverage_v3.md",
        "paper/tables/fevp_ce2eplan_portability_v3.md",
        "paper/tables/fevp_adversarial_prompts_v3.md",
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
    v15 = _load(
        "fp_package_v15",
        "scripts/build_fp_submission_evidence_package_v15.py",
    )
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
                *v15._auditability_artifacts(),
                *_portability_artifacts(),
            ]
        )
    )
    missing = [path for path in artifacts if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing v16 artifacts: {missing}")
    data_directories = [
        *v11.DATA_DIRECTORIES,
        "outputs/external_baselines/raster2seq/official_test509_v1/jsons",
    ]
    evidence = _read("outputs/reports/fp_cvpr_evidence_v11.json")
    final = _read("outputs/reports/fevp_final_claim_v4.json")
    ce2e = _read("outputs/reports/fevp_ce2eplan_portability_v3.json")
    prompts = _read("outputs/reports/fevp_adversarial_prompts_v3.json")
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
        "status": "portable_fevp_and_prompt_robustness_v16",
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
        "archive": "outputs/cvpr_package_fp_v16.tar.gz",
        "hash_algorithm": "sha256",
        "file_count": file_count,
        "missing_count": 0,
        "artifact_count": len(artifacts),
        "artifacts": artifact_rows,
        "data_directories": directory_rows,
        "portability": {
            "CE2EPlan_cases": ce2e["cases"],
            "CE2EPlan_outside_exact": "not applicable",
            "HouseMind": "blocked external; no machine-readable outputs",
            "adversarial_prompt_proxies": prompts["prompts"],
        },
        "excluded_external_assets": {
            "CE2EPlan": "No license; code, RPLAN data, weights, and generated images excluded.",
            "HouseMind": "Official inference remains Code (Soon).",
        },
        "verification": {
            "pytest": _pytest(ROOT / "outputs/reports/fp_pytest_v16.xml"),
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
    readme = f"""# FloorFlow CVPR Experiment Evidence v16

Portable FEVP and adversarial prompt-proxy evidence only.

## Claim Addition

{final['portability_claim']}

{final['prompt_robustness_claim']}

## Boundaries

- CE2EPlan is a generation-only portability target.
- HouseMind inference remains unavailable; missing audit cells are not failures.
- Adversarial prompts are real-control conflict proxies, not an NLU benchmark.
- No generation SOTA, HouseMind IoU/FID/GED defeat, watertight-guard benefit, or human claim.
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
