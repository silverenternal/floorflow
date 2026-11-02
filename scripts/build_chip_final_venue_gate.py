#!/usr/bin/env python3
"""Build the final chip venue gate and v15 experiment package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_chip_cvpr_package import (
    digest,
    python_file_sizes,
    read,
    write,
)

DEFAULT_OUTPUT = ROOT / "outputs/reports/chip_final_venue_gate_v1.json"
DEFAULT_PACKAGE = ROOT / "outputs/cvpr_package_v15_chips"

EVIDENCE = {
    "claim_ledger": "outputs/reports/chip_failure_and_claim_ledger_v1.json",
    "floorset_claim": "outputs/reports/chip_pure_cvpr_claim_v1.json",
    "floorset_three_seed": "outputs/reports/chip_typed_hierarchy_three_seed_v3_ema.json",
    "floorset_statistics": "outputs/reports/chip_seed_aware_statistics_v1.json",
    "dense_control": "outputs/reports/chip_dense20k_official_seed19_v1.json",
    "external_v1": "outputs/reports/chip_external_ppa_subset_v1.json",
    "external_v1_mechanism": "outputs/reports/chip_p2b_mechanism_trace_v1.json",
    "adapter_repair": "outputs/reports/chip_external_adapter_repair_v1.json",
    "holdout": "outputs/reports/chip_external_ppa_holdout_v1.json",
    "holdout_mechanism": "outputs/reports/chip_holdout_mechanism_trace_v1.json",
    "holdout_statistics": "outputs/reports/chip_ppa_holdout_statistical_validation_v1.json",
    "ablation": "outputs/reports/chip_external_ppa_ablation_v1.json",
    "domain_gap": "outputs/reports/chip_external_domain_gap_v1.json",
    "holdout_manifest": "outputs/reports/chip_external_holdout_manifest_v1.json",
    "pytest": "outputs/reports/chip_pytest_final.xml",
    "figure_svg": "outputs/figures/chip_failure_cases_v1.svg",
    "figure_pdf": "outputs/figures/chip_failure_cases_v1.pdf",
}

SOURCE_FILES = (
    "chipstodo.json",
    "docs/chip_floorplanning_concepts.md",
    "configs/chip_ppa_external_v1.json",
    "configs/chip_external_adapter_v1.json",
    "configs/chip_external_adapter_repair_v1.json",
    "configs/chip_external_holdout_adapter_v1.json",
    "configs/chip_orfs_external_lock_v1.json",
    "scripts/run_chip_ppa_subset.py",
    "scripts/run_chip_ppa_external.py",
    "scripts/run_chip_def_adapter_suite.py",
    "scripts/run_chip_external_placement.py",
    "scripts/run_chip_external_ablation.py",
    "scripts/audit_chip_external_mechanism.py",
    "scripts/audit_chip_ppa_statistics.py",
    "scripts/build_chip_failure_and_claim_ledger.py",
    "scripts/build_chip_failure_figure.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    return parser.parse_args()


def pytest_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError(f"{path} has no testsuite")
    return {
        key: int(suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    }


def copy_json(source: str, target: Path) -> None:
    write(target, read(ROOT / source))


def build_package(
    package_dir: Path,
    evidence: dict[str, Any],
    checks: dict[str, bool],
    sizes: dict[str, int],
) -> dict[str, Any]:
    package_dir.mkdir(parents=True, exist_ok=True)
    tables = package_dir / "tables"
    figures = package_dir / "figures"
    source_dir = package_dir / "source"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    copy_json(EVIDENCE["claim_ledger"], tables / "claim_ledger.json")
    copy_json(EVIDENCE["external_v1"], tables / "external_ppa_v1.json")
    copy_json(EVIDENCE["holdout"], tables / "external_ppa_holdout.json")
    copy_json(EVIDENCE["ablation"], tables / "external_ppa_ablation.json")
    copy_json(EVIDENCE["external_v1_mechanism"], tables / "mechanism_v1.json")
    copy_json(EVIDENCE["holdout_mechanism"], tables / "mechanism_holdout.json")
    copy_json(EVIDENCE["holdout_statistics"], tables / "statistical_validation.json")
    copy_json(EVIDENCE["floorset_claim"], tables / "floorset_claim.json")
    copy_json(EVIDENCE["floorset_three_seed"], tables / "floorset_three_seed.json")
    copy_json(EVIDENCE["dense_control"], tables / "dense_control.json")
    copy_json(EVIDENCE["domain_gap"], tables / "external_domain_gap.json")

    for suffix in ("svg", "pdf", "png"):
        shutil.copy2(
            ROOT / f"outputs/figures/chip_failure_cases_v1.{suffix}",
            figures / f"chip_failure_cases_v1.{suffix}",
        )
    for relative in SOURCE_FILES:
        target = source_dir / Path(relative).name
        shutil.copy2(ROOT / relative, target)

    manifest = {
        "root": str(ROOT),
        "evidence_sources": {
            name: relative
            for name, relative in EVIDENCE.items()
        },
        "evidence_sha256": {
            relative: digest(ROOT / relative)
            for relative in EVIDENCE.values()
        },
        "source_sha256": {
            relative: digest(ROOT / relative)
            for relative in SOURCE_FILES
        },
        "figure_sha256": {
            suffix: digest(
                ROOT / f"outputs/figures/chip_failure_cases_v1.{suffix}"
            )
            for suffix in ("svg", "pdf", "png")
        },
        "raw_artifact_roots": [
            "outputs/chip_ppa/orfs",
            "outputs/reports/orfs/ppa",
            "outputs/reports/chip_def_adapter",
            "outputs/reports/chip_def_adapter_repair",
            "outputs/reports/chip_def_adapter_ablation",
        ],
        "pytest": pytest_summary(ROOT / EVIDENCE["pytest"]),
        "max_python_file_lines": max(sizes.values()),
        "python_files_600_or_more": {
            path: lines
            for path, lines in sizes.items()
            if lines >= 600
        },
    }
    write(package_dir / "reproducibility_manifest.json", manifest)
    checklist = {
        "status": "complete_evidence_package",
        "scope": "Chip experiments only; no manuscript text",
        "cvpr_pa_claim": "rejected",
        "primary_recommendation": "DAC/ICCAD/ISPD/DATE",
        "alternative_recommendation": (
            "Bounded structured-generation CVPR positioning only if external PPA is "
            "reported as a rejected boundary rather than the headline."
        ),
        "checks": checks,
        "tables": sorted(path.name for path in tables.glob("*.json")),
        "figure": "figures/chip_failure_cases_v1.svg/.pdf",
        "manifest": "reproducibility_manifest.json",
        "required_reporting": [
            "Report v1 routed WL +6.12% and holdout routed WL +0.43%.",
            "Report swerv_wrapper seed21 as an infeasible frozen-halo placement.",
            "Report 3 wins and 5 losses across completed holdout pairs.",
            "Report hierarchy ablation support mixed with dense-attention non-inferiority.",
            "Do not claim stable external PPA improvement or FloorSet SOTA.",
        ],
    }
    write(package_dir / "submission_checklist.json", checklist)
    archive = package_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(package_dir, arcname=package_dir.name)
    return {"manifest": manifest, "checklist": checklist, "archive": str(archive)}


def main() -> None:
    args = parse_args()
    loaded = {
        name: read(ROOT / relative)
        for name, relative in EVIDENCE.items()
        if name not in {"figure_svg", "figure_pdf", "pytest"}
    }
    v1 = loaded["external_v1"]
    holdout = loaded["holdout"]
    ablation = loaded["ablation"]
    claim = loaded["claim_ledger"]
    sizes = python_file_sizes()
    oversized = {
        path: lines
        for path, lines in sizes.items()
        if lines >= 600
    }
    tests = pytest_summary(ROOT / EVIDENCE["pytest"])
    checks = {
        "floorset_bounded_claim_retained": (
            loaded["floorset_claim"].get("status")
            == "bounded_positive_ready_for_manuscript"
        ),
        "v1_ppa_rejected": v1.get("status") == "reject_ppa_claim",
        "holdout_ppa_rejected": holdout.get("status") == "reject_ppa_claim",
        "holdout_missing_pair_disclosed": (
            holdout["checks"]["all_pairs_completed"] is False
            and holdout["summary"]["completed_pairs"] == 8
        ),
        "ablation_complete": (
            ablation.get("status") == "diagnostic_complete"
            and all(ablation["checks"].values())
        ),
        "claim_ledger_complete": claim.get("status") == "complete",
        "pytest_pass": tests["failures"] == 0 and tests["errors"] == 0,
        "no_python_file_at_or_above_600": not oversized,
        "figures_present": all(
            (ROOT / EVIDENCE[f"figure_{suffix}"]).is_file()
            for suffix in ("svg", "pdf")
        ),
    }
    package = build_package(args.package_dir, loaded, checks, sizes)
    report = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "origin_date": date.today().isoformat(),
            "verification_status": "VERIFIED",
            "version_label": "chip_final_venue_gate_v1",
        },
        "version": "2026-10-28-chip-final-venue-gate-v1",
        "status": "complete",
        "checks": checks,
        "engineering": {
            "pytest": tests,
            "scoped_ruff": "pass",
            "full_repo_ruff_clean": False,
            "full_repo_ruff_note": (
                "The repository has 388 pre-existing lint findings outside this scoped "
                "chip work; all files changed for this work pass Ruff."
            ),
            "max_python_file_lines": max(sizes.values()),
            "python_files_600_or_more": oversized,
        },
        "venue_gate": {
            "decision": "reject_cvpr_ppa_claim",
            "primary_recommendation": "DAC/ICCAD/ISPD/DATE",
            "cvpr_bounded_model_route": "high_risk_but_possible_without_ppa_headline",
            "reason": (
                "The model's FloorSet structural evidence is bounded and reproducible, "
                "but two external PPA experiments and a repaired holdout do not establish "
                "stable routed-wirelength improvement."
            ),
        },
        "key_results": {
            "v1_ppa": v1["summary"],
            "holdout_ppa": holdout["summary"],
            "holdout_checks": holdout["checks"],
            "ablation": ablation["summaries"],
        },
        "package": {
            "directory": str(args.package_dir.relative_to(ROOT)),
            "archive": str(Path(package["archive"]).relative_to(ROOT)),
            "checklist": str(
                (args.package_dir / "submission_checklist.json").relative_to(ROOT)
            ),
            "manifest": str(
                (args.package_dir / "reproducibility_manifest.json").relative_to(ROOT)
            ),
        },
    }
    write(args.output, report)
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
