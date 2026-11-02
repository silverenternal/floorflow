#!/usr/bin/env python3
"""Build the frozen candidate/control placement manifest for the 15-design cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "outputs/reports/chipbench_holdout_ppa_v1/candidate_placements.json",
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=ROOT / "outputs/reports/chipbench_holdout_ppa_v1/control_placements.json",
    )
    parser.add_argument(
        "--candidate-forced",
        type=Path,
        default=ROOT / "outputs/reports/chipbench_holdout_ppa_v1/candidate_forced_shelf.json",
    )
    parser.add_argument(
        "--control-forced",
        type=Path,
        default=ROOT / "outputs/reports/chipbench_holdout_ppa_v1/control_forced_shelf.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chipbench_holdout_pair_manifest_v1.json",
    )
    parser.add_argument(
        "--skip-forced",
        action="store_true",
        help="Build from the primary arms only.",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reports_by_case(path: Path) -> dict[str, dict]:
    report = load(path)
    return {
        row["case"]: {
            **load(ROOT / row["placement_report"]),
            "_report_path": row["placement_report"],
        }
        for row in report["runs"]
        if row["status"] == "pass"
    }


def main() -> None:
    args = parse_args()
    candidate = reports_by_case(args.candidate)
    control = reports_by_case(args.control)
    candidate_forced = (
        {}
        if args.skip_forced
        else reports_by_case(args.candidate_forced)
    )
    control_forced = (
        {}
        if args.skip_forced
        else reports_by_case(args.control_forced)
    )
    rows = []
    for case in sorted(candidate):
        if case not in control:
            raise ValueError(f"missing control placement for {case}")
        candidate_report = candidate[case]
        control_report = control[case]
        forced_cases = {
            key
            for key in candidate_forced
            if key in control_forced
        }
        if case in forced_cases:
            candidate_report = candidate_forced[case]
            control_report = control_forced[case]
        candidate_route = candidate_report.get("legalizer_route")
        control_route = control_report.get("legalizer_route")
        if candidate_route != control_route:
            raise ValueError(
                f"unmatched legalizer routes for {case}: "
                f"{candidate_route} != {control_route}"
            )
        candidate_geometry = candidate_report["geometry"]["legalized_output"]
        control_geometry = control_report["geometry"]["legalized_output"]
        if not (
            candidate_geometry["checker_feasible"]
            and control_geometry["checker_feasible"]
            and candidate_geometry["overlap_pair_count"] == 0
            and control_geometry["overlap_pair_count"] == 0
        ):
            raise ValueError(f"illegal placement pair for {case}")
        candidate_tcl = ROOT / candidate_report["outputs"]["tcl"]
        control_tcl = ROOT / control_report["outputs"]["tcl"]
        candidate_hash = digest(candidate_tcl)
        control_hash = digest(control_tcl)
        distinct = candidate_hash != control_hash
        rows.append(
            {
                "case": case,
                "legalizer_route": candidate_route,
                "candidate": {
                    "placement_report": candidate_report["_report_path"],
                    "macro_tcl": candidate_report["outputs"]["tcl"],
                    "tcl_sha256": candidate_hash,
                    "boxes": candidate_report["outputs"]["boxes"],
                },
                "control": {
                    "placement_report": control_report["_report_path"],
                    "macro_tcl": control_report["outputs"]["tcl"],
                    "tcl_sha256": control_hash,
                    "boxes": control_report["outputs"]["boxes"],
                },
                "checks": {
                    "candidate_feasible": True,
                    "control_feasible": True,
                    "candidate_zero_overlap": True,
                    "control_zero_overlap": True,
                    "routes_match": True,
                    "distinct_tcl": distinct,
                },
            }
        )
    all_distinct = all(row["checks"]["distinct_tcl"] for row in rows)
    report = {
        "version": "chipbench-holdout-pair-manifest-v1",
        "status": (
            "frozen_pair_manifest_ready"
            if all_distinct
            else "reject_collapsed_candidate_control_pairs"
        ),
        "cases": len(rows),
        "forced_shelf_cases": sorted(
            set(candidate_forced) & set(control_forced)
        ),
        "all_routes_matched": all(
            row["checks"]["routes_match"] for row in rows
        ),
        "all_legalized_feasible": all(
            row["checks"]["candidate_feasible"]
            and row["checks"]["control_feasible"]
            for row in rows
        ),
        "all_zero_overlap": all(
            row["checks"]["candidate_zero_overlap"]
            and row["checks"]["control_zero_overlap"]
            for row in rows
        ),
        "all_candidate_control_tcl_distinct": all_distinct,
        "collapsed_cases": [
            row["case"]
            for row in rows
            if not row["checks"]["distinct_tcl"]
        ],
        "routed_ppa_opened": False,
        "rows": rows,
        "claim_boundary": (
            "Pair placement manifest only. ORFS routed PPA remains to be run."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": report["cases"],
                "forced_shelf_cases": report["forced_shelf_cases"],
                "collapsed_cases": report["collapsed_cases"],
            },
            indent=2,
        )
    )
    if not all_distinct:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
