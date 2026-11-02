"""Build validation-only fidelity, tail, distribution, and Pareto evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.residential_graph_frontier import (
    build_frontier_summary,
)

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "outputs/reports/residential_graph_refinement_exact_rows_v1.json"
REPORT = ROOT / "outputs/reports/residential_e2e_frontier_v1.json"
FIGURE = ROOT / "paper/figures/residential_e2e_frontier_v1.pdf"
CAPTION = ROOT / "paper/figures/residential_e2e_frontier_v1.caption.txt"
COLORS = {
    "baseline": "#526173",
    "node": "#2f6fb6",
    "edge": "#16837a",
    "candidate": "#3f8f5f",
    "risk": "#b84a5a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    parser.add_argument("--caption", type=Path, default=CAPTION)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def _paired(rows: list[dict], metric: str, bootstrap: int) -> dict:
    return paired_delta(
        [{"value": row["metrics"]["direct_tiled"][metric]} for row in rows],
        [
            {"value": row["metrics"]["wall_wall_primary"][metric]}
            for row in rows
        ],
        "value",
        iterations=bootstrap,
        seed=19001,
    )


def _plot(summary: dict, path: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    operations = list(summary["strata"]["operations"])
    display = [
        operation.replace("_", " ").replace("delete", "delete").title()
        for operation in operations
    ]
    y = np.arange(len(operations))
    node_delta = [
        summary["strata"]["operations"][operation]["paired"]["node_f1"]["delta"]
        for operation in operations
    ]
    edge_delta = [
        summary["strata"]["operations"][operation]["paired"]["edge_overlap"]["delta"]
        for operation in operations
    ]
    node_bars = axes[0].barh(
        y - 0.18,
        node_delta,
        height=0.32,
        color=COLORS["node"],
        label="Node F1",
    )
    edge_bars = axes[0].barh(
        y + 0.18,
        edge_delta,
        height=0.32,
        color=COLORS["edge"],
        label="Edge overlap",
    )
    axes[0].set_yticks(y, display)
    axes[0].set_xlabel("Paired gain vs wall-face graph")
    axes[0].set_title("Per-operation fidelity")
    axes[0].set_xlim(0.0, 0.55)
    axes[0].grid(axis="x", alpha=0.25)
    for index, operation in enumerate(operations):
        ged = summary["strata"]["operations"][operation]["paired"]["ged"][
            "relative_delta"
        ]
        axes[0].text(
            0.41,
            index,
            f"GED {100 * ged:+.0f}%",
            va="center",
            fontsize=6.5,
            color=COLORS["baseline"],
        )

    for operation in operations:
        baseline = summary["operation_points"][operation]["baseline"]
        candidate = summary["operation_points"][operation]["candidate"]
        x_values = [
            baseline["distribution"]["mmd"],
            candidate["distribution"]["mmd"],
        ]
        y_values = [baseline["node_f1"], candidate["node_f1"]]
        axes[1].plot(
            x_values,
            y_values,
            color=COLORS["baseline"],
            linewidth=1.2,
            alpha=0.55,
        )
        axes[1].scatter(
            x_values[0],
            y_values[0],
            color=COLORS["baseline"],
            s=24,
            marker="o",
        )
        axes[1].scatter(
            x_values[1],
            y_values[1],
            color=COLORS["candidate"],
            s=30,
            marker="D",
        )
        short_name = {
            "room_type": "type",
            "room_remove": "remove",
            "room_add": "add",
            "opening_delete": "opening",
            "furniture_delete": "furniture",
        }[operation]
        label_offset = {
            "furniture": (6, -8),
            "opening": (6, 6),
            "type": (5, 2),
            "add": (-25, -4),
            "remove": (5, 4),
        }[short_name]
        axes[1].annotate(
            short_name,
            (x_values[1], y_values[1]),
            xytext=label_offset,
            textcoords="offset points",
            fontsize=6,
        )
    axes[1].set_xlabel("Graph MMD (lower is better)")
    axes[1].set_ylabel("Node F1")
    axes[1].set_title("Fidelity-distribution Pareto")
    axes[1].grid(alpha=0.25)

    room_bins = list(summary["strata"]["room_count"])
    x = np.arange(len(room_bins))
    room_node = [
        summary["strata"]["room_count"][name]["paired"]["node_f1"]["delta"]
        for name in room_bins
    ]
    room_edge = [
        summary["strata"]["room_count"][name]["paired"]["edge_overlap"]["delta"]
        for name in room_bins
    ]
    axes[2].bar(
        x - 0.18,
        room_node,
        width=0.32,
        color=COLORS["node"],
        label="Node F1",
    )
    axes[2].bar(
        x + 0.18,
        room_edge,
        width=0.32,
        color=COLORS["edge"],
        label="Edge overlap",
    )
    axes[2].set_xticks(x, ["0-4", "5-7", "8+"])
    axes[2].set_xlabel("Target room count")
    axes[2].set_title("Complexity strata")
    axes[2].set_ylim(0.0, 0.46)
    axes[2].grid(axis="y", alpha=0.25)
    for index, name in enumerate(room_bins):
        ged = summary["strata"]["room_count"][name]["paired"]["ged"][
            "relative_delta"
        ]
        axes[2].text(
            index,
            max(room_node[index], room_edge[index]) + 0.035,
            f"{100 * ged:+.0f}% GED",
            ha="center",
            fontsize=6.2,
            color=COLORS["baseline"],
        )
    figure.tight_layout(w_pad=1.2, rect=(0.0, 0.06, 1.0, 1.0))
    figure.legend(
        [node_bars, edge_bars],
        ["Node F1", "Edge overlap"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = json.loads(args.rows.read_text(encoding="utf-8"))["rows"]
    summary = build_frontier_summary(rows)
    paired = {
        metric: _paired(rows, metric, args.bootstrap)
        for metric in ("node_f1", "edge_overlap", "ged")
    }
    distribution = summary["distribution"]
    checks = {
        "node_f1_paired_ci_positive": paired["node_f1"]["ci_low"] > 0.0,
        "edge_overlap_paired_ci_positive": (
            paired["edge_overlap"]["ci_low"] > 0.0
        ),
        "ged_paired_ci_negative": paired["ged"]["ci_high"] < 0.0,
        "all_operations_quality_improve": all(
            values["paired"]["node_f1"]["delta"] > 0.0
            and values["paired"]["edge_overlap"]["delta"] > 0.0
            and values["paired"]["ged"]["delta"] < 0.0
            for values in summary["strata"]["operations"].values()
        ),
        "all_room_bins_quality_improve": all(
            values["paired"]["node_f1"]["delta"] > 0.0
            and values["paired"]["edge_overlap"]["delta"] > 0.0
            and values["paired"]["ged"]["delta"] < 0.0
            for values in summary["strata"]["room_count"].values()
        ),
        "distribution_mmd_improves": (
            distribution["candidate"]["mmd"] < distribution["baseline"]["mmd"]
        ),
        "coverage_not_reduced": (
            distribution["candidate"]["coverage"]
            >= distribution["baseline"]["coverage"]
        ),
        "diversity_not_collapsed": (
            distribution["candidate"]["diversity"]
            >= 0.5 * distribution["baseline"]["diversity"]
        ),
        "worst_decile_ged_improves": (
            summary["tail"]["candidate_ged_highest"]["paired"]["ged"]["delta"]
            < 0.0
        ),
    }
    report = {
        "schema_version": "residential-e2e-frontier-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "residential_floorplan_only",
        "split": "val191",
        "test_opened": False,
        "source_rows": str(args.rows.relative_to(ROOT)),
        "summary": summary,
        "paired_bootstrap": paired,
        "checks": checks,
        "tail_interpretation": {
            "candidate_node_tail_status": (
                "retained_limitation"
                if summary["tail"]["candidate_node_f1_lowest"]["paired"][
                    "node_f1"
                ]["delta"]
                < 0.0
                else "improved"
            ),
            "note": (
                "The candidate-node worst decile can regress while overall "
                "fidelity, GED, operation strata, and room-count strata improve."
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _plot(summary, args.figure)
    args.caption.parent.mkdir(parents=True, exist_ok=True)
    args.caption.write_text(
        "Direct-tiled graph refinement improves room identity and adjacency "
        "across edit operations and room-count strata while moving the "
        "predicted graph distribution closer to the fixed target. The worst "
        "candidate-node decile remains a reported limitation.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.report.relative_to(ROOT)),
                "figure": str(args.figure.relative_to(ROOT)),
                "checks": checks,
            },
            indent=2,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
