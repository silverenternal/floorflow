#!/usr/bin/env python3
"""Regenerate collapsed v45 pairs with position-aware shelf packing."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.refiner_config import append_refiner_protocol

AUTO_TAG = "v45_collapse_sensitivity"
SHELF_TAG = "v45_collapse_sensitivity_shelf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_external_holdout_v45.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_external_pair_integrity.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_collapse_sensitivity.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(design_key: str, nickname: str) -> str:
    platform = design_key.split("/", maxsplit=1)[0]
    return f"{platform}__{nickname}"


def report_path(design_key: str, arm: str, tag: str = AUTO_TAG) -> Path:
    case_name = slug(design_key, design_key.rsplit("/", maxsplit=1)[-1])
    return (
        ROOT
        / "outputs/reports"
        / tag
        / case_name
        / f"{arm}_seed19.json"
    )


def command(
    args: argparse.Namespace,
    config: dict[str, Any],
    design_key: str,
    arm: str,
    *,
    tag: str,
    force_shelf: bool,
) -> list[str]:
    case = config["cases"][design_key]
    policy = config["primary_protocol"]
    checkpoint = ROOT / config[arm]["seeds"]["19"]
    case_name = slug(design_key, design_key.rsplit("/", maxsplit=1)[-1])
    output = report_path(design_key, arm, tag)
    result = [
        sys.executable,
        str(ROOT / "scripts/run_chip_external_placement.py"),
        "--case",
        case_name,
        "--checkpoint",
        str(checkpoint),
        "--arm",
        arm,
        "--seed",
        "19",
        "--odb",
        str(ROOT / case["input_odb"]),
        "--device",
        args.device,
        "--nfe",
        str(policy["nfe"]),
        "--refiner-steps",
        str(policy["refiner_steps"]),
        "--projector-iterations",
        str(policy["candidate_projector_iterations"]),
        "--macro-halo-um",
        str(case["macro_halo_um"]),
        "--legalizer-relation",
        str(policy["legalizer_relation"]),
        "--protocol-tag",
        tag,
        "--position-aware-shelf",
        "--output",
        str(output),
    ]
    if force_shelf:
        result.append("--force-shelf")
    append_refiner_protocol(result, policy)
    if policy.get("instance_hierarchy"):
        result.append("--instance-hierarchy")
    if policy.get("preserve_reference_orientation"):
        result.append("--preserve-reference-orientation")
    return result


def run_row(
    args: argparse.Namespace,
    config: dict[str, Any],
    design_key: str,
    original: dict[str, Any],
    *,
    tag: str,
    force_shelf: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    arms = {}
    for arm in ("candidate", "control"):
        result = subprocess.run(
            command(
                args,
                config,
                design_key,
                arm,
                tag=tag,
                force_shelf=force_shelf,
            ),
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        current_report_path = report_path(design_key, arm, tag)
        report = load(current_report_path) if current_report_path.is_file() else {}
        tcl = ROOT / report.get("outputs", {}).get("tcl", "")
        arms[arm] = {
            "exit_code": result.returncode,
            "report": (
                str(current_report_path.relative_to(ROOT))
                if current_report_path.is_file()
                else None
            ),
            "tcl": str(tcl.relative_to(ROOT)) if tcl.is_file() else None,
            "tcl_sha256": digest(tcl) if tcl.is_file() else None,
            "legalizer_route": report.get("legalizer_route"),
            "p2b_hpwl_um": report.get("connectivity", {})
            .get("legalized_output", {})
            .get("p2b_hpwl_um"),
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
    hashes = [arms[arm]["tcl_sha256"] for arm in ("candidate", "control")]
    return {
        "case": design_key,
        "strategy": "forced_position_shelf" if force_shelf else "automatic_core_fit",
        "original_candidate_sha256": original["candidate"]["tcl_sha256"],
        "original_control_sha256": original["control"]["tcl_sha256"],
        "arms": arms,
        "distinct_after_position_aware_shelf": bool(
            all(hashes) and hashes[0] != hashes[1]
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    config = load(args.config)
    audit = load(args.audit)
    by_case = {row["case"]: row for row in audit["cases"]}
    selected = args.case or audit["collapsed_cases"]
    rows = []
    for design_key in selected:
        if design_key not in audit["collapsed_cases"]:
            raise ValueError(f"{design_key} is not a collapsed v45 case")
        row = run_row(
            args,
            config,
            design_key,
            by_case[design_key],
            tag=AUTO_TAG,
            force_shelf=False,
        )
        rows.append(row)
        print(
            design_key,
            "distinct" if row["distinct_after_position_aware_shelf"] else "collapsed",
            flush=True,
        )
    for index, row in enumerate(rows):
        routes = {
            row["arms"][arm]["legalizer_route"]
            for arm in ("candidate", "control")
        }
        if len(routes) == 1 or not row["distinct_after_position_aware_shelf"]:
            continue
        forced = run_row(
            args,
            config,
            row["case"],
            by_case[row["case"]],
            tag=SHELF_TAG,
            force_shelf=True,
        )
        forced_routes = {
            forced["arms"][arm]["legalizer_route"]
            for arm in ("candidate", "control")
        }
        if forced["distinct_after_position_aware_shelf"] and len(forced_routes) == 1:
            forced["automatic_attempt"] = row
            rows[index] = forced
            print(f"{row['case']} common forced shelf", flush=True)
        else:
            row["forced_shelf_attempt"] = forced
            print(f"{row['case']} forced shelf rejected", flush=True)
    checks = {
        "all_commands_succeeded": all(
            row["arms"][arm]["exit_code"] == 0
            for row in rows
            for arm in ("candidate", "control")
        ),
        "all_pairs_distinct": all(
            row["distinct_after_position_aware_shelf"] for row in rows
        ),
        "common_route_within_case": all(
            len(
                {
                    row["arms"][arm]["legalizer_route"]
                    for arm in ("candidate", "control")
                }
            )
            == 1
            for row in rows
        ),
    }
    report = {
        "version": "chip-v45-collapse-sensitivity-v1",
        "status": "pass" if all(checks.values()) else "reject_or_redesign",
        "source_audit": str(args.audit.relative_to(ROOT)),
        "policy": (
            "Sensitivity only. The original v45 endpoints remain archived and "
            "this corrected cohort cannot replace the observed 15-case evidence."
        ),
        "cases": rows,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
