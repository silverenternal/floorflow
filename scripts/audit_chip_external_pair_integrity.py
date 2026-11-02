#!/usr/bin/env python3
"""Audit whether candidate/control external PPA pairs use distinct placements."""

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
        "--adapter-report",
        type=Path,
        default=ROOT / "outputs/reports/chip_def_adapter_holdout_v45_seed19_legalizer_v3.json",
    )
    parser.add_argument(
        "--ppa-report",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_v45_partial.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_external_pair_integrity.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    args.adapter_report = args.adapter_report.resolve()
    args.ppa_report = args.ppa_report.resolve()
    args.output = args.output.resolve()
    adapter = load(args.adapter_report)
    ppa = load(args.ppa_report)
    statuses = {
        row["case"]: row["status"]
        for row in ppa.get("comparisons", [])
    }
    pairs: dict[str, dict[str, Any]] = {}
    if "placement_runs" in adapter:
        for run in adapter["placement_runs"]:
            report_path = ROOT / run["report"]
            report = load(report_path)
            tcl_path = ROOT / report["outputs"]["tcl"]
            pairs.setdefault(run["design_key"], {})[run["arm"]] = {
                "report": str(report_path.relative_to(ROOT)),
                "tcl": str(tcl_path.relative_to(ROOT)),
                "tcl_sha256": digest(tcl_path) if tcl_path.is_file() else None,
                "legalizer_route": report.get("legalizer_route"),
                "p2b_hpwl_um": report.get("connectivity", {})
                .get("legalized_output", {})
                .get("p2b_hpwl_um"),
            }
    else:
        for run in adapter.get("runs", []):
            tcl_path = ROOT / run["macro_placement_tcl"]
            pairs.setdefault(run["case"], {})[run["arm"]] = {
                "report": str(args.ppa_report.relative_to(ROOT)),
                "tcl": run["macro_placement_tcl"],
                "tcl_sha256": (
                    digest(tcl_path)
                    if tcl_path.is_file()
                    else run.get("macro_placement_tcl_sha256")
                ),
                "legalizer_route": run.get("legalizer_route"),
                "p2b_hpwl_um": None,
            }
    cases = []
    for case, arms in sorted(pairs.items()):
        candidate = arms.get("candidate")
        control = arms.get("control")
        distinct = bool(
            candidate
            and control
            and candidate["tcl_sha256"]
            and control["tcl_sha256"]
            and candidate["tcl_sha256"] != control["tcl_sha256"]
        )
        cases.append(
            {
                "case": case,
                "ppa_status": statuses.get(case, "missing"),
                "candidate": candidate,
                "control": control,
                "distinct_legalized_placements": distinct,
                "pair_usable_for_model_comparison": (
                    distinct and statuses.get(case) == "completed"
                ),
            }
        )
    collapsed = [
        row["case"] for row in cases if not row["distinct_legalized_placements"]
    ]
    completed = [row for row in cases if row["ppa_status"] == "completed"]
    valid_completed = [
        row["case"] for row in completed if row["pair_usable_for_model_comparison"]
    ]
    invalid_completed = [
        row["case"] for row in completed if not row["pair_usable_for_model_comparison"]
    ]
    checks = {
        "all_cases_have_both_arms": all(
            row["candidate"] is not None and row["control"] is not None
            for row in cases
        ),
        "all_completed_pairs_are_distinct": not invalid_completed,
    }
    report = {
        "version": "chip-v45-external-pair-integrity-v1",
        "status": (
            "pass"
            if all(checks.values())
            else "reject_legalized_candidate_control_collapse"
        ),
        "adapter_report": str(args.adapter_report.relative_to(ROOT)),
        "ppa_report": str(args.ppa_report.relative_to(ROOT)),
        "case_count": len(cases),
        "completed_case_count": len(completed),
        "valid_completed_case_count": len(valid_completed),
        "collapsed_pair_count": len(collapsed),
        "collapsed_cases": collapsed,
        "valid_completed_cases": valid_completed,
        "invalid_completed_cases": invalid_completed,
        "cases": cases,
        "checks": checks,
        "decision": (
            "The original endpoints remain archived. A collapsed pair cannot "
            "support a candidate/control PPA comparison and cannot be silently "
            "replaced by a corrected rerun."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_case_count": len(completed),
                "valid_completed_case_count": len(valid_completed),
                "collapsed_pair_count": len(collapsed),
            },
            indent=2,
        )
    )
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
