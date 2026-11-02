#!/usr/bin/env python3
"""Build an immutable manifest for the chip CVPR experiment evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    "chipstodo.json",
    "docs/chip_sota_experiment_protocol_v44.md",
    "configs/chip_external_holdout_adapter_v44.json",
    "configs/chip_orfs_external_lock_v1.json",
    "outputs/reports/chip_canonical_anchor_audit_v44.json",
    "outputs/reports/chip_refiner_official_1600_three_seed_v44.json",
    "outputs/reports/chip_graph_token_official_1600_three_seed_v44.json",
    "outputs/reports/chip_matched_budget_1600_detglop_seed19_v44.json",
    "outputs/reports/chip_model_promotion_v44.json",
    "outputs/reports/chip_seed_rootcause_v44.json",
    "outputs/reports/chip_external_holdout_inventory_v44.json",
    "outputs/reports/chip_external_ppa_holdout_v44_all_seed19.json",
    "outputs/reports/chip_external_ppa_holdout_v44_seeds20_21.json",
    "outputs/reports/chip_external_ppa_holdout_v44_all.json",
    "outputs/reports/chip_ppa_endpoint_certification_v44.json",
    "outputs/reports/chip_p2b_routing_rootcause_v44_holdout.json",
    "outputs/reports/chip_final_sota_gate_v44.json",
)
SOURCE_GLOBS = (
    "scripts/*chip*v44*.py",
    "scripts/evaluate_chip_refiner.py",
    "scripts/evaluate_chip_official.py",
    "scripts/eval_trinity_baseline.py",
    "scripts/run_chip_external_placement.py",
    "scripts/run_chip_ppa_subset.py",
    "scripts/run_chip_def_adapter_suite.py",
    "src/floorflow/chip_flow/batched.py",
    "src/floorflow/chip_flow/legalization.py",
    "src/floorflow/chip_flow/reproducibility.py",
    "src/floorflow/chip_flow/refiner_config.py",
    "src/floorflow/chip_flow/train_cli.py",
    "src/floorflow/chip_flow/train_objective.py",
)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_reproducibility_manifest_v44.json",
    )
    args = parser.parse_args()
    evidence = {
        path: sha256(ROOT / path)
        for path in EVIDENCE
        if (ROOT / path).is_file()
    }
    sources = sorted(
        {
            str(path.relative_to(ROOT))
            for pattern in SOURCE_GLOBS
            for path in ROOT.glob(pattern)
            if path.is_file()
        }
    )
    source_hashes = {path: sha256(ROOT / path) for path in sources}
    python_files = sorted(
        path
        for root in ("src", "scripts", "tests")
        for path in (ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    )
    line_counts = {
        str(path.relative_to(ROOT)): len(path.read_text(encoding="utf-8").splitlines())
        for path in python_files
    }
    oversized = {
        path: count for path, count in line_counts.items() if count >= 600
    }
    pytest_path = ROOT / "outputs/reports/chip_pytest_final_v44.xml"
    pytest_summary = None
    if pytest_path.is_file():
        suite = ET.parse(pytest_path).find(".//testsuite")
        if suite is not None:
            pytest_summary = {
                "tests": int(suite.attrib["tests"]),
                "failures": int(suite.attrib["failures"]),
                "errors": int(suite.attrib["errors"]),
                "skipped": int(suite.attrib["skipped"]),
                "seconds": float(suite.attrib["time"]),
            }
    report = {
        "version": "2026-10-28-chip-reproducibility-manifest-v44",
        "root": str(ROOT),
        "evidence_sha256": evidence,
        "source_sha256": source_hashes,
        "missing_evidence": [path for path in EVIDENCE if path not in evidence],
        "engineering": {
            "python_files_checked": len(line_counts),
            "max_python_file_lines": max(line_counts.values()),
            "files_at_or_above_600_lines": oversized,
            "pytest": pytest_summary,
        },
        "canonical_commands": {
            "three_seed_floorplan": (
                "uv run python scripts/build_chip_refiner_official_report.py "
                "--steps 1600 "
                "--report-template 'candidate_boundary2_detglop_{steps}_seed{seed}.json'"
            ),
            "external_seed19": (
                "uv run python scripts/run_chip_ppa_subset.py "
                "--config configs/chip_external_holdout_adapter_v44.json "
                "--manifest outputs/reports/chip_external_holdout_inventory_v44.json "
                "--seeds 19 --jobs 2 --num-cores 16 --collect-only"
            ),
            "external_replication": (
                "uv run python scripts/run_chip_ppa_subset.py "
                "--config configs/chip_external_holdout_adapter_v44.json "
                "--manifest outputs/reports/chip_external_holdout_inventory_v44.json "
                "--cases nangate45/ariane136 nangate45/black_parrot "
                "nangate45/bp_be_top nangate45/bp_fe_top nangate45/bp_multi_top "
                "--seeds 20 21 --jobs 8 --num-cores 3"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "evidence_files": len(evidence),
                "missing_evidence": len(report["missing_evidence"]),
                "max_python_file_lines": report["engineering"][
                    "max_python_file_lines"
                ],
                "oversized_python_files": oversized,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
