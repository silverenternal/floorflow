#!/usr/bin/env python3
"""Freeze the adapter-certified v45 external holdout manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_external_holdout_v45.json",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_inventory_v45_expanded_audit.json",
    )
    parser.add_argument(
        "--adapter-report",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_def_adapter_holdout_v45_seed19_legalizer_v3.json",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "configs/chip_orfs_external_lock_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_holdout_v45_manifest_v2.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = load(args.config)
    inventory = load(args.inventory)
    adapter = load(args.adapter_report)
    lock = load(args.lock)
    audit_cases = {
        row["design_key"]: row for row in inventory["inventory"]
    }
    config_cases = set(config["cases"])
    selected_cases = set(inventory["selection"]["selected_cases"])
    adapter_cases = set(adapter["cases"])
    expected_cases = config_cases & selected_cases
    cases = []
    for design_key in sorted(expected_cases):
        row = audit_cases[design_key]
        adapter_case = config["cases"][design_key]
        odb = ROOT / adapter_case["input_odb"]
        cases.append(
            {
                "design_key": design_key,
                "platform": row["platform"],
                "design_nickname": row["design_nickname"],
                "source_family": adapter_case["source_family"],
                "input_odb": adapter_case["input_odb"],
                "input_odb_sha256": sha256(odb),
                "macro_halo_um": adapter_case["macro_halo_um"],
                "coordinate_roundtrip_tolerance_um": adapter_case[
                    "coordinate_roundtrip_tolerance_um"
                ],
            }
        )
    placement_runs = adapter["placement_runs"]
    roundtrips = adapter["roundtrips"]
    checks = {
        "config_has_15_cases": len(config_cases) == 15,
        "inventory_selection_matches_config": selected_cases == config_cases,
        "adapter_has_15_cases": adapter_cases == config_cases,
        "adapter_status_pass": adapter["status"] == "pass",
        "placements_pass": len(placement_runs) == 30
        and all(run["exit_code"] == 0 for run in placement_runs),
        "roundtrips_pass": len(roundtrips) == 30
        and all(row["status"] == "pass" for row in roundtrips),
        "nine_source_families": len({row["source_family"] for row in cases}) == 9,
        "image_pinned": bool(lock["image"].get("linux_amd64_manifest_digest")),
    }
    report = {
        "version": "chip-external-holdout-v45-manifest-v2",
        "status": (
            "adapter_certified_before_new_ppa_outcomes"
            if all(checks.values())
            else "fail"
        ),
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": sha256(args.config),
        "inventory": str(args.inventory.relative_to(ROOT)),
        "inventory_sha256": sha256(args.inventory),
        "adapter_report": str(args.adapter_report.relative_to(ROOT)),
        "adapter_report_sha256": sha256(args.adapter_report),
        "lock": str(args.lock.relative_to(ROOT)),
        "lock_sha256": sha256(args.lock),
        "case_count": len(cases),
        "cases": cases,
        "checks": checks,
        "policy": (
            "All candidate/control placements and ORFS roundtrips passed before "
            "opening new routed-PPA outcomes. No design may be added or removed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
