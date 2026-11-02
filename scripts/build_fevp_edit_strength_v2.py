#!/usr/bin/env python3
"""Build edit-strength and edit-type fidelity evidence."""

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
        "--delta-rows",
        type=Path,
        default=ROOT / "outputs/reports/fevp_delta_locality_rows_v2.json",
    )
    parser.add_argument(
        "--graph-rows",
        type=Path,
        default=ROOT / "outputs/reports/fevp_graph_rows_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_edit_strength_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_edit_strength_v2.md",
    )
    return parser.parse_args()


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * std / np.sqrt(len(array)))
    return {
        "cases": len(array),
        "mean": mean,
        "std": std,
        "ci95_student_t_case": [mean - half, mean + half],
        "ci_method": "student_t_interval_over_cases",
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _group(rows: list[dict], predicate, name: str) -> dict:
    selected = [row for row in rows if predicate(row)]
    if not selected:
        raise ValueError(f"empty edit-strength group: {name}")
    return {
        "name": name,
        **_distribution([row["delta_iou"] for row in selected]),
        "delta_mse": _distribution([row["delta_mse"] for row in selected]),
        "outside_exact_mean": float(
            np.mean([row["outside_exact"] for row in selected])
        ),
        "locality_violation_rate_mean": float(
            np.mean(
                [row["strict_locality_violation_rate"] for row in selected]
            )
        ),
    }


def main() -> None:
    args = parse_args()
    rows = json.loads(args.delta_rows.read_text(encoding="utf-8"))["rows"]
    graph = json.loads(args.graph_rows.read_text(encoding="utf-8"))["rows"]
    groups = [
        _group(rows, lambda row: row["change_bin"] == "changed_rooms_1", "1 room"),
        _group(
            rows,
            lambda row: row["change_bin"] == "changed_rooms_2_3",
            "2-3 rooms",
        ),
        _group(
            rows,
            lambda row: row["change_bin"] == "changed_rooms_4_plus",
            "4+ rooms",
        ),
        _group(rows, lambda row: row["operation"] == "room_remove", "room deletion"),
        _group(
            rows,
            lambda row: row["operation"] in {"opening_delete", "furniture_delete"},
            "opening/furniture edit",
        ),
    ]
    graph_by_operation = {}
    for operation in sorted({row["operation"] for row in graph}):
        selected = [row for row in graph if row["operation"] == operation]
        graph_by_operation[operation] = {
            "cases": len(selected),
            "node_f1": float(
                np.mean(
                    [row["arms"]["typed_edit_graph"]["node_f1"] for row in selected]
                )
            ),
            "edge_f1": float(
                np.mean(
                    [row["arms"]["typed_edit_graph"]["edge_f1"] for row in selected]
                )
            ),
            "ged": float(
                np.mean(
                    [row["arms"]["typed_edit_graph"]["ged"] for row in selected]
                )
            ),
        }
    report = {
        "status": "complete",
        "cases": len(rows),
        "groups": groups,
        "graph_by_operation": graph_by_operation,
        "checks": {
            "groups_reconcile_subset": all(group["cases"] > 0 for group in groups),
            "strict_locality_preserved": all(
                group["outside_exact_mean"] == 1.0
                and group["locality_violation_rate_mean"] == 0.0
                for group in groups
            ),
            "room_deletion_ged_tail_available": (
                graph_by_operation["room_remove"]["ged"]
                >= max(
                    graph_by_operation[operation]["ged"]
                    for operation in graph_by_operation
                )
            ),
        },
        "interpretation": (
            "Delta fidelity is reported under strict locality. Room deletion is "
            "retained as the principal graph-distance tail rather than hidden "
            "inside the aggregate GED."
        ),
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# Delta Fidelity by Edit Strength",
        "",
        "| Group | Cases | Delta-IoU | IoU 95% interval | IoU P10 | Delta-MSE | Outside exact | Violation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        table.append(
            f"| {group['name']} | {group['cases']} | {group['mean']:.6f} | "
            f"[{group['ci95_student_t_case'][0]:.6f}, "
            f"{group['ci95_student_t_case'][1]:.6f}] | "
            f"{group['p10']:.6f} | {group['delta_mse']['mean']:.9f} | "
            f"{group['outside_exact_mean']:.6f} | "
            f"{group['locality_violation_rate_mean']:.9f} |"
        )
    table.extend(
        [
            "",
            "| Operation | Node F1 | Edge F1 | GED |",
            "|---|---:|---:|---:|",
        ]
    )
    for operation, values in graph_by_operation.items():
        table.append(
            f"| {operation} | {values['node_f1']:.6f} | "
            f"{values['edge_f1']:.6f} | {values['ged']:.6f} |"
        )
    table.extend(
        [
            "",
            report["interpretation"],
            "",
            "The intervals are descriptive case-level Student-t intervals, not five-seed population inference.",
            "",
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
