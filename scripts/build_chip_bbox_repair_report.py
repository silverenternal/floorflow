#!/usr/bin/env python3
"""Compare the 5k L0 bbox-alignment repair with its matched controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_hpwl_gap",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_mib",
    "legalized_boundary",
    "legalized_rho_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        default=(
            ROOT
            / "outputs/chip_hierarchy/v5_bbox_repair_5k/"
            "reproducible_v1"
        ),
    )
    parser.add_argument(
        "--no-bbox",
        type=Path,
        default=ROOT / "outputs/chip_hierarchy/v3_screen_5k/reproducible_v1",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "outputs/chip_hierarchy/v3_screen_5k/reproducible_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/chip_typed_hierarchy_bbox_repair_5k_v1.json"
        ),
    )
    parser.add_argument("--candidate-stem", default="candidate_primary")
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def primary(path: Path, stem: str) -> tuple[dict, list[dict]]:
    summary = read(path / f"{stem}.json")["results"]["nfe4_refine400"]
    rows = read(path / f"{stem}_cases.json")["nfe4_refine400"]
    return summary, rows


def compare(
    candidate_rows: list[dict],
    control_rows: list[dict],
    bootstrap: int,
    offset: int,
) -> dict:
    output = {}
    for index, metric in enumerate(METRICS):
        current = np.asarray([float(row[metric]) for row in candidate_rows])
        control = np.asarray([float(row[metric]) for row in control_rows])
        current_mean = float(current.mean())
        control_mean = float(control.mean())
        output[metric] = {
            "candidate_mean": current_mean,
            "control_mean": control_mean,
            "relative_delta": (
                current_mean / control_mean - 1.0
                if control_mean != 0.0
                else None
            ),
            "paired": paired_delta(
                candidate_rows,
                control_rows,
                metric,
                iterations=bootstrap,
                seed=20261200 + offset + index,
                lower_is_better=True,
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    candidate, candidate_rows = primary(args.candidate, args.candidate_stem)
    no_bbox, no_bbox_rows = primary(
        args.no_bbox,
        "candidate_primary",
    )
    baseline, baseline_rows = primary(
        args.baseline,
        "baseline_primary",
    )
    vs_no_bbox = compare(
        candidate_rows,
        no_bbox_rows,
        args.bootstrap,
        0,
    )
    vs_baseline = compare(
        candidate_rows,
        baseline_rows,
        args.bootstrap,
        100,
    )
    checks = {
        "legalized_feasible_fraction_one": (
            candidate["legalized_feasible_fraction"] == 1.0
        ),
        "area_improves_5pct_vs_no_bbox": (
            vs_no_bbox["legalized_area_gap"]["relative_delta"] <= -0.05
        ),
        "cost_regression_within_2pct_vs_no_bbox": (
            vs_no_bbox["legalized_hard_cost"]["relative_delta"] <= 0.02
        ),
        "v_rel_regression_within_2pct_vs_no_bbox": (
            vs_no_bbox["legalized_v_rel"]["relative_delta"] <= 0.02
        ),
        "b2b_regression_within_2pct_vs_no_bbox": (
            vs_no_bbox["legalized_b2b_hpwl_gap"]["relative_delta"] <= 0.02
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "reject_bbox_repair",
        "scope": "seed-19 5k single-variable L0-to-L2 bbox alignment screen",
        "candidate": candidate,
        "no_bbox_typed_hierarchy": no_bbox,
        "graph_token_control": baseline,
        "candidate_vs_no_bbox": vs_no_bbox,
        "candidate_vs_graph_token": vs_baseline,
        "checks": checks,
        "decision": (
            "promote hierarchy-bbox-weight 0.5 to the 20k gate"
            if all(checks.values())
            else "reject the bbox alignment objective"
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
                "checks": checks,
                "vs_no_bbox": {
                    metric: vs_no_bbox[metric]["relative_delta"]
                    for metric in METRICS[:6]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
