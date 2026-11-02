#!/usr/bin/env python3
"""Freeze the E1 external routed-PPA decision from completed ORFS evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = "outputs/reports/chip_external_ppa_holdout_v44_all.json"
PIN_ALIGN = "outputs/reports/orfs/chipbench_m49_p32_pinalign_or1200_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_e1_external_holdout_v1.json",
    )
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    holdout = load(HOLDOUT)
    pin_align = load(PIN_ALIGN)
    checks = {
        "at_least_15_completed_pairs": int(holdout["completed_pairs"]) >= 15,
        "at_least_8_designs": bool(holdout["checks"]["at_least_8_designs"]),
        "drc_zero_both_arms": bool(holdout["checks"]["drc_zero_both_arms"]),
        "routed_wirelength_nonregression": bool(
            holdout["checks"]["routed_wirelength_nonregression"]
        ),
        "congestion_mean_within_2pct": bool(
            holdout["checks"]["congestion_mean_at_most_2pct"]
        ),
        "timing_nonregression": bool(
            holdout["checks"]["timing_mean_nonregression_20ps"]
        ),
        "power_within_5pct": bool(holdout["checks"]["power_max_within_5pct"]),
        "stdcell_area_within_5pct": bool(
            holdout["checks"]["stdcell_area_max_within_5pct"]
        ),
    }
    ppa_promotable = (
        checks["at_least_15_completed_pairs"]
        and checks["at_least_8_designs"]
        and checks["drc_zero_both_arms"]
        and checks["routed_wirelength_nonregression"]
        and checks["congestion_mean_within_2pct"]
        and checks["timing_nonregression"]
        and checks["power_within_5pct"]
        and checks["stdcell_area_within_5pct"]
    )
    report = {
        "version": "chip-e1-external-holdout-v1",
        "status": "pass" if ppa_promotable else "reject_ppa_claim",
        "source": HOLDOUT,
        "designs": len(holdout["cases"]),
        "completed_pairs": int(holdout["completed_pairs"]),
        "wirelength_win_tie_loss": holdout["wirelength_win_tie_loss"],
        "summary": holdout["summary"],
        "checks": checks,
        "ppa_promotable": ppa_promotable,
        "routeability_sensitivity": {
            "source": PIN_ALIGN,
            "status": pin_align["status"],
            "drc_errors": pin_align["metric_audit"]["values"][
                "detailedroute__route__drc_errors"
            ],
            "hold_metrics_present": False,
            "decision": (
                "Do not use this or1200 sensitivity run as primary external "
                "evidence or retry route iterations."
            ),
        },
        "claim_decision": (
            "The frozen external cohort is complete and DRC-clean, but routed "
            "wirelength and congestion regress. Do not claim routed-PPA "
            "transfer or external SOTA."
        ),
        "paper_boundary": (
            "Report the external cohort as a negative transfer result under a "
            "pinned ORFS protocol. Keep internal structural gains independent."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
