#!/usr/bin/env python3
"""Validate matched-flow proposal budgets and common legalizer routes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-n1",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_search_candidate_n1_positioned_all.json",
    )
    parser.add_argument(
        "--candidate-n2",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_search_candidate_n2_positioned_all.json",
    )
    parser.add_argument(
        "--control-n1",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_search_control_n1_positioned_all.json",
    )
    parser.add_argument(
        "--control-n2",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_search_control_n2_positioned_all.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_matched_flow_preflight.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    inputs = {
        "candidate_n1": args.candidate_n1,
        "candidate_n2": args.candidate_n2,
        "control_n1": args.control_n1,
        "control_n2": args.control_n2,
    }
    reports = {name: load(path) for name, path in inputs.items()}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    budget_checks = {}
    for arm, report in reports.items():
        expected_count = 1 if arm.endswith("n1") else 2
        expected_nfe = 4 // expected_count
        expected_refiner = 1600 // expected_count
        rows = [
            proposal
            for case in report["results"]
            for proposal in case["proposals"]
        ]
        budget_checks[arm] = {
            "report_status": report["status"],
            "expected_proposals_per_case": expected_count,
            "proposal_count": sum(
                len(case["proposals"]) for case in report["results"]
            ),
            "nfe_each_is_expected": all(
                row["nfe"] == expected_nfe for row in rows
            ),
            "refiner_steps_each_is_expected": all(
                row["refiner_steps"] == expected_refiner for row in rows
            ),
        }
        for case in report["results"]:
            for proposal in case["proposals"]:
                by_case[case["case"]].append(
                    {
                        "arm": arm,
                        **proposal,
                    }
                )
    cases = []
    for case, rows in sorted(by_case.items()):
        p2b_values = [float(row["p2b_hpwl_um"]) for row in rows]
        routes = sorted({str(row["legalizer_route"]) for row in rows})
        tcl_hashes = sorted(
            {
                file_digest(ROOT / str(row["placement_tcl"]))
                for row in rows
            }
        )
        cases.append(
            {
                "case": case,
                "proposals": len(rows),
                "legalizer_routes": routes,
                "p2b_hpwl_um": p2b_values,
                "p2b_range": max(p2b_values) - min(p2b_values),
                "p2b_has_discrimination": max(p2b_values) - min(p2b_values)
                > 1.0e-9,
                "unique_tcl_hashes": len(tcl_hashes),
                "tcl_has_diversity": len(tcl_hashes) > 1,
            }
        )
    checks = {
        "all_reports_generated": all(
            report["status"] == "generated" for report in reports.values()
        ),
        "budget_contract": all(
            row["report_status"] == "generated"
            and row["nfe_each_is_expected"]
            and row["refiner_steps_each_is_expected"]
            for row in budget_checks.values()
        ),
        "common_route_per_case": all(
            len(case["legalizer_routes"]) == 1 for case in cases
        ),
    }
    p2b_signal = {
        "discriminative_cases": [
            case["case"] for case in cases if case["p2b_has_discrimination"]
        ],
        "tied_cases": [
            case["case"] for case in cases if not case["p2b_has_discrimination"]
        ],
        "rule": "P2b discrimination is reported per case; it is not a pass criterion.",
    }
    report = {
        "version": "chip-v46-matched-flow-preflight-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "scope": (
            "Legalization and budget audit only. No global-place response or "
            "routed quality metric is included."
        ),
        "inputs": {name: str(path) for name, path in inputs.items()},
        "budget_checks": budget_checks,
        "cases": cases,
        "p2b_signal": p2b_signal,
        "checks": checks,
        "decision": (
            "Proceed to analytical response scoring with each case on its common "
            "legalizer route; compare against p2b separately on tied and "
            "discriminative cases."
            if all(checks.values())
            else "Do not score until budget or legalizer mismatches are repaired."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
