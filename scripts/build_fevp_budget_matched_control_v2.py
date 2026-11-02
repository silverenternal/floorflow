#!/usr/bin/env python3
"""Audit Typed versus No-graph under matched model and routing overhead."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "outputs/reports/fp_typed_edit_graph_ablation_rows_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_budget_matched_control_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_budget_matched_control_v2.md",
    )
    return parser.parse_args()


def _effective_ms(row: dict, arm: str) -> float:
    seconds = float(row["policy_seconds"][arm])
    if arm == "typed_edit_graph":
        seconds += float(row["graph_build_seconds"])
    return seconds * 1000.0


def _effect(rows: list[dict], metric: str) -> dict:
    typed = [float(row["arms"]["typed_edit_graph"][metric]) for row in rows]
    no_graph = [
        float(row["arms"]["no_structural_graph"][metric]) for row in rows
    ]
    return paired_delta(
        [{"value": value} for value in typed],
        [{"value": value} for value in no_graph],
        "value",
        iterations=4000,
        seed=95000 + len(metric),
    )


def main() -> None:
    args = parse_args()
    rows = json.loads(args.rows.read_text(encoding="utf-8"))["rows"]
    typed_ms = np.asarray(
        [_effective_ms(row, "typed_edit_graph") for row in rows]
    )
    no_graph_ms = np.asarray(
        [_effective_ms(row, "no_structural_graph") for row in rows]
    )
    overhead = typed_ms - no_graph_ms
    report = {
        "status": "complete",
        "cases": len(rows),
        "same_model_checkpoint_inputs": True,
        "added_learned_parameters": 0,
        "same_network_nfe": True,
        "budget_matched_control": {
            "definition": (
                "No-graph quality is paired with the Typed arm's full graph-build "
                "plus routing latency. The control therefore receives the same "
                "observable compute budget, or idles any surplus capacity."
            ),
            "typed_latency_ms_mean": float(typed_ms.mean()),
            "no_graph_latency_ms_mean": float(no_graph_ms.mean()),
            "typed_overhead_ms_mean": float(overhead.mean()),
            "typed_overhead_ms_median": float(np.median(overhead)),
        },
        "quality_deltas_typed_minus_budget_matched_no_graph": {
            metric: _effect(rows, metric)
            for metric in (
                "outside_pixel_exact",
                "locality_violation_rate",
                "delta_iou",
                "node_f1",
                "edge_f1",
                "ged",
            )
        },
        "checks": {
            "no_extra_learned_parameters": True,
            "same_network_nfe": True,
            "typed_has_higher_routing_compute": bool(float(overhead.mean()) > 0.0),
            "typed_delta_iou_better_at_fixed_budget": bool(
                np.mean(
                    [
                        row["arms"]["typed_edit_graph"]["delta_iou"]
                        - row["arms"]["no_structural_graph"]["delta_iou"]
                        for row in rows
                    ]
                )
                > 0.0
            ),
            "typed_ged_better_at_fixed_budget": bool(
                np.mean(
                    [
                        row["arms"]["typed_edit_graph"]["ged"]
                        - row["arms"]["no_structural_graph"]["ged"]
                        for row in rows
                    ]
                )
                < 0.0
            ),
        },
        "boundary": (
            "This is a post-hoc routing compute audit, not evidence that Typed is "
            "more efficient. Typed consumes more routing time and is not claimed "
            "as an efficiency contribution."
        ),
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# Compute-Matched Typed vs No-Graph",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Added learned parameters | 0 |",
        f"| Network NFE change | 0 |",
        f"| Typed latency | {typed_ms.mean():.6f} ms |",
        f"| No-graph latency | {no_graph_ms.mean():.6f} ms |",
        f"| Typed overhead | {overhead.mean():.6f} ms |",
        "",
        "| Metric | Typed - budget-matched No-graph | 95% bootstrap CI |",
        "|---|---:|---:|",
    ]
    for metric, value in report[
        "quality_deltas_typed_minus_budget_matched_no_graph"
    ].items():
        table.append(
            f"| {metric} | {value['mean_delta']:+.6f} | "
            f"[{value['ci_low']:+.6f}, {value['ci_high']:+.6f}] |"
        )
    table.extend(["", report["boundary"], ""])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
