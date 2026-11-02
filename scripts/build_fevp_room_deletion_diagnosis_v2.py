#!/usr/bin/env python3
"""Diagnose the room-deletion GED tail without conflating it with leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edit-strength",
        type=Path,
        default=ROOT / "outputs/reports/fevp_edit_strength_v2.json",
    )
    parser.add_argument(
        "--graph-rows",
        type=Path,
        default=ROOT / "outputs/reports/fevp_graph_rows_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_room_deletion_diagnosis_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_room_deletion_diagnosis_v2.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strength = json.loads(args.edit_strength.read_text(encoding="utf-8"))
    graph = json.loads(args.graph_rows.read_text(encoding="utf-8"))["rows"]
    values = [
        float(row["arms"]["typed_edit_graph"]["ged"])
        for row in graph
    ]
    threshold = float(np.quantile(values, 0.90))
    tail = [
        row
        for row in graph
        if float(row["arms"]["typed_edit_graph"]["ged"]) >= threshold
    ]
    tail_by_operation = {}
    for operation in sorted({row["operation"] for row in graph}):
        operation_tail = [
            row for row in tail if row["operation"] == operation
        ]
        tail_by_operation[operation] = {
            "cases": len(operation_tail),
            "tail_share": len(operation_tail) / len(tail),
        }
    deletion_group = next(
        group for group in strength["groups"] if group["name"] == "room deletion"
    )
    deletion_graph = strength["graph_by_operation"]["room_remove"]
    deletion_ged_values = [
        float(row["arms"]["typed_edit_graph"]["ged"])
        for row in graph
        if row["operation"] == "room_remove"
    ]
    deletion_ged_mean = float(np.mean(deletion_ged_values))
    deletion_ged_std = float(np.std(deletion_ged_values, ddof=1))
    deletion_half = float(
        t.ppf(0.975, len(deletion_ged_values) - 1)
        * deletion_ged_std
        / np.sqrt(len(deletion_ged_values))
    )
    report = {
        "status": "complete",
        "room_deletion": {
            **deletion_graph,
            "delta_iou": deletion_group["mean"],
            "delta_mse": deletion_group["delta_mse"]["mean"],
            "outside_exact": deletion_group["outside_exact_mean"],
            "locality_violation_rate": deletion_group[
                "locality_violation_rate_mean"
            ],
            "ged_cases": len(deletion_ged_values),
            "ged_std": deletion_ged_std,
            "ged_ci95": [
                deletion_ged_mean - deletion_half,
                deletion_ged_mean + deletion_half,
            ],
            "ged_ci_method": "student_t_interval_over_cases",

        },
        "ged_tail": {
            "threshold_p90": threshold,
            "cases": len(tail),
            "by_operation": tail_by_operation,
        },
        "claim": (
            "Room deletion is the principal graph-distance difficulty: GED 5.418, "
            "Node F1 0.864, and Edge F1 0.717. It is not a locality failure: "
            "outside exactness remains 1.0 and violation rate remains zero on "
            "the same subset. FEVP therefore separates outside leakage from "
            "topological uncertainty intrinsic to deletion."
        ),
        "checks": {
            "room_deletion_is_ged_tail": deletion_graph["ged"] > 5.0,
            "room_deletion_strict_locality": (
                deletion_group["outside_exact_mean"] == 1.0
                and deletion_group["locality_violation_rate_mean"] == 0.0
            ),
            "room_deletion_dominates_tail": tail_by_operation["room_remove"][
                "tail_share"
            ]
            > 0.5,
            "deletion_sample_not_135": len(deletion_ged_values) == 2500,
        },
        "boundary": (
            "This diagnosis explains a hard mode; it does not claim room deletion "
            "is solved or that GED is comparable across evaluators."
        ),
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# Room-Deletion Diagnosis",
        "",
        "| Metric | Room deletion |",
        "|---|---:|",
        f"| Delta-IoU | {report['room_deletion']['delta_iou']:.6f} |",
        f"| Delta-MSE | {report['room_deletion']['delta_mse']:.9f} |",
        f"| Node F1 | {report['room_deletion']['node_f1']:.6f} |",
        f"| Edge F1 | {report['room_deletion']['edge_f1']:.6f} |",
        f"| GED | {report['room_deletion']['ged']:.6f} |",
        (
            f"| GED 95% Student-t CI | "
            f"[{report['room_deletion']['ged_ci95'][0]:.6f}, "
            f"{report['room_deletion']['ged_ci95'][1]:.6f}] |"
        ),
        f"| GED cases | {report['room_deletion']['ged_cases']} |",
        f"| Outside exact | {report['room_deletion']['outside_exact']:.6f} |",
        f"| Locality violation | {report['room_deletion']['locality_violation_rate']:.9f} |",
        "",
        "| Tail metric | Value |",
        "|---|---:|",
        f"| Overall GED P90 threshold | {threshold:.6f} |",
        f"| Room-removal share of top-decile GED | {tail_by_operation['room_remove']['tail_share']:.6f} |",
        "",
        report["claim"],
        "",
        report["boundary"],
        "The 135-case figure belongs to the 4+ room edit-strength bin, not room deletion.",
        "",
    ]
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
