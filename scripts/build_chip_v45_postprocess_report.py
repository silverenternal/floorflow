#!/usr/bin/env python3
"""Compare the corrected v2 postprocess smoke with its source placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(candidate: float, control: float) -> float | None:
    return candidate / control - 1.0 if control != 0.0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_holdout_v44_seed19.json",
    )
    parser.add_argument(
        "--postprocess",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_v45_postprocess_ppa_v2_bp_blackparrot.json",
    )
    parser.add_argument(
        "--placement-report",
        type=Path,
        default=ROOT
        / "outputs/chip_ppa/placements_v45_postprocess_v2/bp/candidate_seed19_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_postprocess_v2_bp_smoke.json",
    )
    return parser.parse_args()


def find_candidate(ppa: dict[str, Any], case: str) -> dict[str, Any]:
    for run in ppa["runs"]:
        if run["case"] == case and run["arm"] == "candidate":
            return run["metrics"]["values"]
    raise ValueError(f"candidate metrics missing for {case}")


def main() -> None:
    args = parse_args()
    baseline_report = load(args.baseline)
    postprocess_report = load(args.postprocess)
    placement = load(args.placement_report)
    case = postprocess_report["case"]
    baseline = find_candidate(baseline_report, case)
    postprocess = postprocess_report["metric_audit"]["values"]
    metrics = {
        "routed_wirelength": "detailedroute__route__wirelength",
        "congestion": "globalplace__gpl__routability__congestion",
        "setup_ws": "finish__timing__setup__ws",
        "hold_ws": "finish__timing__hold__ws",
        "power": "finish__power__total",
        "instance_area": "finish__design__instance__area",
        "drc_errors": "detailedroute__route__drc_errors",
    }
    comparisons = {}
    for name, key in metrics.items():
        before = baseline.get(key)
        after = postprocess.get(key)
        comparisons[name] = {
            "before": before,
            "after": after,
            "relative": (
                relative(float(after), float(before))
                if before is not None and after is not None
                else None
            ),
        }
    checks = {
        "proxy_improves": placement["proxy_relative"] < 0.0,
        "legal_after": bool(placement["legal_after"]),
        "routed_wirelength_improves": (
            comparisons["routed_wirelength"]["relative"] is not None
            and comparisons["routed_wirelength"]["relative"] < 0.0
        ),
        "congestion_nonregression": (
            comparisons["congestion"]["relative"] is not None
            and comparisons["congestion"]["relative"] <= 0.0
        ),
        "setup_ws_nonregression": (
            comparisons["setup_ws"]["after"] >= comparisons["setup_ws"]["before"]
        ),
        "drc_zero": comparisons["drc_errors"]["after"] == 0,
        "power_regression_below_5pct": (
            comparisons["power"]["relative"] is not None
            and comparisons["power"]["relative"] <= 0.05
        ),
        "area_regression_below_5pct": (
            comparisons["instance_area"]["relative"] is not None
            and comparisons["instance_area"]["relative"] <= 0.05
        ),
    }
    report = {
        "version": "2026-10-29-chip-v45-postprocess-v2-smoke-v1",
        "status": (
            "smoke_pass_pending_multicase"
            if all(checks.values())
            else "rejected_smoke"
        ),
        "case": case,
        "source_placement": placement["source"],
        "postprocess": {
            "proxy_before": placement["proxy_before"],
            "proxy_after": placement["proxy_after"],
            "proxy_relative": placement["proxy_relative"],
            "moved_macros": placement["moved_macros"],
            "displacement_um": placement["displacement_um"],
        },
        "metrics": comparisons,
        "checks": checks,
        "claim_boundary": (
            "This single-case routed smoke supports advancing to a frozen "
            "multi-case postprocess screen. It is not holdout evidence and does "
            "not establish routed-PPA improvement."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
