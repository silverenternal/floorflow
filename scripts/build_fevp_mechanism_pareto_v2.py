#!/usr/bin/env python3
"""Build mechanism Pareto fronts for locality, edit quality, and graph quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "typed_edit_graph",
    "dense_cross_attention",
    "no_edit_gating",
    "no_structural_graph",
)


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
        default=ROOT / "outputs/reports/fevp_mechanism_pareto_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_mechanism_pareto_v2.md",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "paper/figures/fevp_mechanism_pareto_v2.pdf",
    )
    return parser.parse_args()


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    half = float(t.ppf(0.975, 4) * std / np.sqrt(5))
    return {
        "mean": mean,
        "std": std,
        "ci95": [mean - half, mean + half],
        "values": [float(value) for value in array],
    }


def _dominates(left: dict, right: dict) -> bool:
    objectives = (
        ("locality_violation_rate", -1),
        ("delta_iou", 1),
        ("node_f1", 1),
        ("edge_f1", 1),
        ("ged", -1),
    )
    no_worse = all(
        direction * left[key]["mean"] >= direction * right[key]["mean"]
        for key, direction in objectives
    )
    better = any(
        direction * left[key]["mean"] > direction * right[key]["mean"] + 1.0e-12
        for key, direction in objectives
    )
    return no_worse and better


def main() -> None:
    args = parse_args()
    rows = json.loads(args.rows.read_text(encoding="utf-8"))["rows"]
    arm_metrics = {}
    for arm in ARMS:
        arm_metrics[arm] = {
            metric: _summary(
                [float(row["arms"][arm][metric]) for row in rows]
            )
            for metric in (
                "outside_pixel_exact",
                "outside_room_exact",
                "locality_violation_rate",
                "delta_iou",
                "delta_mse",
                "node_f1",
                "edge_f1",
                "ged",
            )
        }
    frontier = {}
    for arm, target_values in arm_metrics.items():
        frontier[arm] = not any(
            _dominates(candidate_values, target_values)
            for other, candidate_values in arm_metrics.items()
            if other != arm
        )
    report = {
        "status": "complete",
        "cases": len(rows),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "objectives": {
            "locality_violation_rate": "minimize",
            "delta_iou": "maximize",
            "node_f1": "maximize",
            "edge_f1": "maximize",
            "ged": "minimize",
        },
        "arms": {
            arm: {
                **values,
                "pareto_frontier": frontier[arm],
            }
            for arm, values in arm_metrics.items()
        },
        "claim": (
            "Typed routing forms a joint locality-quality and locality-graph "
            "frontier. Graphless routing can preserve locality but loses edit "
            "and graph quality; no-gate trades visible quality for measurable "
            "outside leakage."
        ),
        "checks": {
            "typed_on_frontier": frontier["typed_edit_graph"],
            "typed_violation_zero": (
                arm_metrics["typed_edit_graph"]["locality_violation_rate"]["mean"]
                == 0.0
            ),
            "typed_better_delta_than_graphless": (
                arm_metrics["typed_edit_graph"]["delta_iou"]["mean"]
                > arm_metrics["no_structural_graph"]["delta_iou"]["mean"]
            ),
            "typed_better_node_than_graphless": (
                arm_metrics["typed_edit_graph"]["node_f1"]["mean"]
                > arm_metrics["no_structural_graph"]["node_f1"]["mean"]
            ),
            "no_gate_has_violation": (
                arm_metrics["no_edit_gating"]["locality_violation_rate"]["mean"]
                > 0.0
            ),
        },
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# Typed Mechanism Pareto",
        "",
        "| Arm | Violation rate | Delta-IoU | Node F1 | Edge F1 | GED | Frontier |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    labels = {
        "typed_edit_graph": "Typed",
        "dense_cross_attention": "Dense",
        "no_edit_gating": "No gate",
        "no_structural_graph": "No graph",
    }
    for arm in ARMS:
        values = arm_metrics[arm]
        table.append(
            f"| {labels[arm]} | "
            f"{values['locality_violation_rate']['mean']:.9f} | "
            f"{values['delta_iou']['mean']:.6f} | "
            f"{values['node_f1']['mean']:.6f} | "
            f"{values['edge_f1']['mean']:.6f} | "
            f"{values['ged']['mean']:.6f} | "
            f"{'yes' if frontier[arm] else 'no'} |"
        )
    table.extend(["", report["claim"], ""])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    colors = {
        "typed_edit_graph": "#16837a",
        "dense_cross_attention": "#2f6fb6",
        "no_edit_gating": "#b84a5a",
        "no_structural_graph": "#b77a00",
    }
    for panel, quality_key, title in (
        (axes[0], "delta_iou", "Locality vs Delta-IoU"),
        (axes[1], "node_f1", "Locality vs Node F1"),
    ):
        for arm in ARMS:
            values = arm_metrics[arm]
            x = values["locality_violation_rate"]["mean"]
            y = values[quality_key]["mean"]
            panel.scatter(
                x,
                y,
                s=50,
                color=colors[arm],
                edgecolor="black",
                linewidth=0.4,
            )
            panel.annotate(
                labels[arm],
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
            )
        panel.set_xscale("symlog", linthresh=1.0e-6)
        panel.set_xlabel("Locality violation rate (lower is better)")
        panel.set_ylabel(title.split(" vs ")[1] + " (higher is better)")
        panel.set_title(title, loc="left", fontsize=8, fontweight="bold")
        panel.grid(True, alpha=0.25)
        panel.spines[["top", "right"]].set_visible(False)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    figure.savefig(args.figure.with_suffix(".svg"))
    plt.close(figure)
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
