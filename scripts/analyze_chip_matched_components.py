#!/usr/bin/env python3
"""Decompose the matched-compute candidate/anchor gap by structural metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "cost": "legalized_hard_cost",
    "v_rel": "legalized_v_rel",
    "grouping": "legalized_grouping",
    "boundary": "legalized_boundary",
    "b2b_hpwl_gap": "legalized_b2b_hpwl_gap",
    "p2b_hpwl_gap": "legalized_p2b_hpwl_gap",
    "area_gap": "legalized_area_gap",
    "rho_mae": "legalized_rho_mae",
    "mib": "legalized_mib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_matched_compute_three_seed.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_matched_component_gap_v1.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(report: dict) -> dict:
    rows = {}
    for seed, values in report["seeds"].items():
        seed_row = {}
        for name, field in METRICS.items():
            candidate = float(values["candidate"][field])
            anchor = float(values["anchor"][field])
            seed_row[name] = {
                "candidate": candidate,
                "anchor": anchor,
                "relative": (
                    candidate / anchor - 1.0
                    if abs(anchor) > 1.0e-15
                    else None
                ),
            }
        rows[seed] = seed_row
    summary = {}
    for name in METRICS:
        relative = [
            row[name]["relative"]
            for row in rows.values()
            if row[name]["relative"] is not None
        ]
        values = [row[name]["candidate"] for row in rows.values()]
        anchors = [row[name]["anchor"] for row in rows.values()]
        summary[name] = {
            "mean_candidate": float(np.mean(values)),
            "mean_anchor": float(np.mean(anchors)),
            "mean_relative": float(np.mean(relative)) if relative else None,
            "seed_improvements": int(sum(value < 0.0 for value in relative)),
            "seed_regressions": int(sum(value > 0.0 for value in relative)),
            "seed_unchanged": int(sum(value == 0.0 for value in relative)),
            "no_discrimination": bool(
                all(value == 0.0 for value in values + anchors)
            ),
        }
    return {
        "version": "chip-matched-component-gap-v1",
        "status": "descriptive_negative_composite",
        "source": str(report.get("version")),
        "source_status": report["status"],
        "inference_seed": report["inference_seed"],
        "per_seed": rows,
        "summary": summary,
        "claim_boundary": (
            "These are matched-protocol structural differences, not evidence "
            "that any component causes the gap or transfers to routed PPA."
        ),
    }


def main() -> None:
    args = parse_args()
    result = analyze(load(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
