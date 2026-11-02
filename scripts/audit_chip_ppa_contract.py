#!/usr/bin/env python3
"""Audit the frozen external PPA contract before data acquisition."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.cache import PackedCaseStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_ppa_external_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_ppa_contract_v1.json",
    )
    return parser.parse_args()


def docker_version() -> str | None:
    executable = shutil.which("docker")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    todo = json.loads((ROOT / "chipstodo.json").read_text(encoding="utf-8"))
    floorstore = PackedCaseStore(
        ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/dev"
    )
    local_ppa_files = sorted(
        str(path.relative_to(ROOT))
        for root in (ROOT / ".data", ROOT / "third_party/Trinity")
        if root.exists()
        for suffix in ("*.def", "*.lef", "*.spef")
        for path in root.rglob(suffix)
    )
    primary = config["metrics"]["primary"]
    case_selection = config["case_selection"]
    checks = {
        "contract_status_frozen": config["status"] == "contract_frozen",
        "case_count_valid": (
            5
            >= case_selection["target_count"]
            >= case_selection["minimum_count"]
            >= 3
        ),
        "case_selection_precedes_candidate": case_selection[
            "freeze_before_candidate_evaluation"
        ],
        "three_fixed_seeds": config["arms"]["candidate_seeds"]
        == [19, 20, 21],
        "primary_endpoint_frozen": (
            primary["metric"] == "mean relative detailed-routed wirelength"
            and primary["minimum_effect"] <= -0.02
            and primary["direction"] == "lower_is_better"
        ),
        "drc_regression_blocks_claim": (
            "legality_or_drc_regression"
            in config["metrics"]["regression_guards"]
            and config["metrics"]["regression_guards"][
                "legality_or_drc_regression"
            ]
            == "reject the PPA claim"
        ),
        "na_cannot_fallback_to_surrogate": "surrogate" in config["metrics"][
            "na_policy"
        ],
        "same_infrastructure_required": all(
            config["same_infrastructure"].get(key)
            for key in (
                "orfs_image_or_binary_digest",
                "openroad_version",
                "opensta_version",
                "klayout_version",
                "platform",
                "sdc",
                "route_effort",
                "endpoint_slack",
                "macro_legalizer",
                "def_writer",
            )
        ),
        "flowplace_license_not_assumed": (
            config["source_policy"]["optional_benchmark_cache"][
                "license_status"
            ]
            == "unverified_no_repository_license_returned"
        ),
        "orfs_is_primary_path": (
            config["source_policy"]["preferred"]["name"]
            == "OpenROAD-flow-scripts"
        ),
        "floorset_cache_has_no_ppa_arrays": not any(
            key in floorstore.arrays
            for key in ("def", "lef", "spef", "congestion", "wns", "tns")
        ),
        "external_ppa_tooling_not_installed": all(
            shutil.which(name) is None
            for name in ("openroad", "sta", "yosys", "klayout")
        ),
        "docker_available": docker_version() is not None,
        "todo_has_active_contract": any(
            task["id"] == "PPA-CONTRACT" and task["status"] == "active"
            for task in todo["tasks"]
        ),
        "todo_has_no_completed_tasks": not any(
            task.get("status") == "complete" for task in todo["tasks"]
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "PPA-CONTRACT audit; no data downloaded and no model evaluated",
        "contract": str(args.config.relative_to(ROOT)),
        "environment": {
            "docker": docker_version(),
            "openroad_on_path": shutil.which("openroad"),
            "opensta_on_path": shutil.which("sta"),
            "yosys_on_path": shutil.which("yosys"),
            "klayout_on_path": shutil.which("klayout"),
            "local_ppa_files_found": local_ppa_files,
            "floorset_dev_cases": len(floorstore),
            "floorset_cache_arrays": sorted(floorstore.arrays),
        },
        "source_policy": config["source_policy"],
        "case_selection": case_selection,
        "primary_endpoint": primary,
        "statistics": config["statistics"],
        "promotion_gate": config["promotion_gate"],
        "checks": checks,
        "decision": {
            "contract_frozen": all(checks.values()),
            "proceed_to": (
                "DATA-INGEST using ORFS official designs first; use FlowPlace "
                "cache only after a separate license/provenance audit."
            ),
            "network_download_performed": False,
            "gpu_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
