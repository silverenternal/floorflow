#!/usr/bin/env python3
"""Redraw mechanism Pareto with audited and un-audited method contexts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "typed_edit_graph",
    "dense_cross_attention",
    "no_edit_gating",
    "no_structural_graph",
)
LABELS = {
    "typed_edit_graph": "Typed",
    "dense_cross_attention": "Dense",
    "no_edit_gating": "No gate",
    "no_structural_graph": "No graph",
}
COLORS = {
    "typed_edit_graph": "#16837a",
    "dense_cross_attention": "#2f6fb6",
    "no_edit_gating": "#b84a5a",
    "no_structural_graph": "#b77a00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pareto",
        type=Path,
        default=ROOT / "outputs/reports/fevp_mechanism_pareto_v2.json",
    )
    parser.add_argument(
        "--housemind",
        type=Path,
        default=ROOT / "outputs/reports/fevp_housemind_positioning_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_mechanism_pareto_v3.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_mechanism_pareto_v3.md",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "paper/figures/fevp_mechanism_pareto_v3.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pareto = json.loads(args.pareto.read_text(encoding="utf-8"))
    housemind = json.loads(args.housemind.read_text(encoding="utf-8"))
    arms = pareto["arms"]
    typed = arms["typed_edit_graph"]
    graphless = arms["no_structural_graph"]
    correction = {
        "checked_against_fixed_evaluator_rows": True,
        "typed_better_than_graphless": {
            metric: typed[metric]["mean"] > graphless[metric]["mean"]
            if metric != "ged"
            else typed[metric]["mean"] < graphless[metric]["mean"]
            for metric in (
                "delta_iou",
                "node_f1",
                "edge_f1",
                "ged",
            )
        },
        "note": (
            "The current fixed-evaluator rows show Typed better than No graph "
            "on Delta-IoU, Node F1, Edge F1, and GED. The older visual reading "
            "that Typed only wins locality was caused by an incorrect frontier "
            "membership calculation; the corrected table and figure are authoritative."
        ),
    }
    context_methods = {
        name: {
            "delta_iou": values["delta_iou"],
            "node_f1": values["node_f1"],
            "edge_overlap": values["edge_overlap"],
            "ged": values["ged"],
            "locality_violation_rate": values["locality_violation_rate"],
            "plot_eligible": False,
            "reason": "Outside exactness and violation rate are not publicly reported.",
        }
        for name, values in housemind["methods"].items()
    }
    report = {
        "status": "complete",
        "cases": pareto["cases"],
        "seeds": pareto["seeds"],
        "audited_arms": arms,
        "house_mind_context": context_methods,
        "data_correction": correction,
        "claim": (
            "Among audited arms, Typed routing is the joint locality-quality "
            "frontier and dominates No graph on Delta-IoU, Node F1, Edge F1, "
            "and GED. HouseMind-E/O cannot be plotted on the locality axis "
            "because the public evaluation does not report outside exactness "
            "or violation rate; they appear only as un-audited context."
        ),
        "checks": {
            "typed_frontier": arms["typed_edit_graph"]["pareto_frontier"],
            "typed_dominates_graphless_on_all_reported_quality_metrics": all(
                correction["typed_better_than_graphless"].values()
            ),
            "housemind_not_assigned_fake_locality_point": all(
                values["locality_violation_rate"] is None
                for values in context_methods.values()
            ),
            "housemind_context_has_published_editing_metrics": (
                context_methods["HouseMind-E"]["delta_iou"] is not None
                and context_methods["HouseMind-O"]["delta_iou"] is not None
            ),
            "generation_only_not_ranked": (
                context_methods["HouseMind-G"]["delta_iou"] is None
            ),
        },
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    table = [
        "# Audited Mechanism Pareto v3",
        "",
        "| Audited arm | Violation | Delta-IoU | Node F1 | Edge F1 | GED | Frontier |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        values = arms[arm]
        table.append(
            f"| {LABELS[arm]} | "
            f"{values['locality_violation_rate']['mean']:.9f} | "
            f"{values['delta_iou']['mean']:.6f} | "
            f"{values['node_f1']['mean']:.6f} | "
            f"{values['edge_f1']['mean']:.6f} | "
            f"{values['ged']['mean']:.6f} | "
            f"{'yes' if values['pareto_frontier'] else 'no'} |"
        )
    table.extend(
        [
            "",
            "## Un-audited Context",
            "",
            "| Method | Delta-IoU | Node F1 | Edge overlap | GED | Locality axis |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name, values in context_methods.items():
        def value(key, row=values):
            item = row[key]
            return "—" if item is None else f"{item:.3f}"

        table.append(
            f"| {name} | {value('delta_iou')} | {value('node_f1')} | "
            f"{value('edge_overlap')} | {value('ged')} | not reported |"
        )
    table.extend(
        [
            "",
            report["claim"],
            "",
            correction["note"],
            "",
            "HouseMind-G is generation-only and cannot enter an editing Pareto.",
            "",
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")

    figure = plt.figure(figsize=(10.8, 3.5), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.05))
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    context_axis = figure.add_subplot(grid[0, 2])
    for axis, quality, title in (
        (axes[0], "delta_iou", "Locality vs Delta-IoU"),
        (axes[1], "node_f1", "Locality vs Node F1"),
    ):
        for arm in ARMS:
            values = arms[arm]
            x = values["locality_violation_rate"]["mean"]
            y = values[quality]["mean"]
            axis.scatter(
                x,
                y,
                s=55,
                color=COLORS[arm],
                edgecolor="black",
                linewidth=0.4,
            )
            axis.annotate(
                LABELS[arm],
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
            )
        axis.set_xscale("symlog", linthresh=1.0e-6)
        axis.set_xlabel("Locality violation rate (lower is better)")
        axis.set_ylabel(title.split(" vs ")[1] + " (higher is better)")
        axis.set_title(title, loc="left", fontsize=8, fontweight="bold")
        axis.grid(True, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    context_axis.axis("off")
    context_lines = [
        "HouseMind public editing context",
        "",
        "E: ΔIoU 0.608 | Node .998",
        "    Edge .934 | GED .467",
        "O: ΔIoU 0.598 | Node .997",
        "    Edge .908 | GED .653",
        "G: generation-only",
        "",
        "Locality audit: not reported",
        "Not plotted on the x-axis.",
    ]
    context_axis.text(
        0.02,
        0.98,
        "\n".join(context_lines),
        va="top",
        ha="left",
        fontsize=7,
        bbox={"facecolor": "#f2f4f7", "edgecolor": "#8c96a3"},
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    figure.savefig(args.figure.with_suffix(".svg"))
    plt.close(figure)
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
