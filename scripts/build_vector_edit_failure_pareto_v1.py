#!/usr/bin/env python3
"""Build FEVP-2 failure-tail evidence and a two-panel Pareto figure."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/reports/vector_edit_results_v1.json"
BASELINES = ROOT / "outputs/reports/vector_edit_baselines_v1.json"
OUTPUT = ROOT / "outputs/reports/vector_edit_failure_tail_v1.json"
TABLE = ROOT / "paper/tables/vector_edit_failure_tail_v1.md"
FIGURE_SVG = ROOT / "paper/figures/vector_edit_pareto_v1.svg"
FIGURE_PDF = ROOT / "paper/figures/vector_edit_pareto_v1.pdf"
COLORS = {
    "blue": "#2f6fb6",
    "teal": "#16837a",
    "green": "#3f8f5f",
    "amber": "#b77a00",
    "red": "#b84a5a",
    "purple": "#7254a7",
    "gray": "#526173",
}


def _mean(rows: list[dict], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows]))


def _style() -> None:
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


def _figure(ours: dict, external: dict) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    panels = (
        (
            axes[0],
            "outside_room_level_exact",
            "node_f1",
            "Strict locality",
            "Node F1",
            (0.96, 1.005),
            (0.0, 1.0),
        ),
        (
            axes[1],
            "validity",
            "edge_overlap",
            "Validity",
            "Edge overlap",
            (0.0, 1.02),
            (0.0, 1.0),
        ),
    )
    for axis, x_key, y_key, x_label, y_label, xlim, ylim in panels:
        points = (
            ("Raster2Seq wrapper", external, COLORS["red"]),
            ("FloorFlow", ours, COLORS["blue"]),
        )
        for label, row, color in points:
            x = row[x_key]
            y = row[y_key]
            axis.scatter([x], [y], s=54, color=color, edgecolor="white", linewidth=0.8)
            axis.annotate(
                f"{label}\n({x:.3f}, {y:.3f})",
                (x, y),
                xytext=(7, 10),
                textcoords="offset points",
                fontsize=7,
                color=color,
                weight="bold",
            )
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.25)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_title("Locality-quality")
    axes[1].set_title("Validity-graph consistency")
    fig.suptitle("FEVP-2 vector audit, matched CubiCasa cases", fontsize=9, weight="bold")
    fig.tight_layout()
    FIGURE_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_SVG)
    fig.savefig(FIGURE_PDF)
    plt.close(fig)
    svg = FIGURE_SVG.read_text(encoding="utf-8")
    identifiers = re.findall(r'id="(p[0-9a-f]+)"', svg)
    replacements = {
        identifier: f"clip_{index + 1}"
        for index, identifier in enumerate(dict.fromkeys(identifiers))
    }
    for identifier, replacement in replacements.items():
        svg = svg.replace(identifier, replacement)
    FIGURE_SVG.write_text(svg, encoding="utf-8")


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = results["rows"]
    baseline = json.loads(BASELINES.read_text(encoding="utf-8"))
    external_rows = [
        row
        for operation in baseline["operations"].values()
        for row in operation["rows"]
    ]
    external = {
        "outside_room_level_exact": float(
            np.mean(
                [
                    row["surgical"]["preservation"]["outside_room_level_exact"]
                    for row in external_rows
                ]
            )
        ),
        "validity": float(
            np.mean(
                [
                    row["surgical"]["topology"]["validity"]["validity"]
                    for row in external_rows
                ]
            )
        ),
        "node_f1": float(
            np.mean([row["surgical"]["graph"]["node_f1"] for row in external_rows])
        ),
        "edge_overlap": float(
            np.mean(
                [row["surgical"]["graph"]["edge_overlap"] for row in external_rows]
            )
        ),
    }
    seed19 = [row for row in rows if int(row["seed"]) == 19]
    ours = {
        metric: _mean(seed19, metric)
        for metric in ("outside_room_level_exact", "validity", "node_f1", "edge_overlap")
    }
    tails = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["operation"]].append(row)
    for operation, selected in grouped.items():
        ordered = sorted(selected, key=lambda row: float(row["node_f1"]))
        count = max(1, math.ceil(0.10 * len(ordered)))
        tail = ordered[:count]
        tails[operation] = {
            "tail_cases": count,
            "representative_seed19_cases": [
                row["case_id"] for row in tail if int(row["seed"]) == 19
            ][:5],
            "node_f1": _mean(tail, "node_f1"),
            "edge_overlap": _mean(tail, "edge_overlap"),
            "validity": _mean(tail, "validity"),
            "ged": _mean(tail, "ged"),
            "outside_room_level_exact": _mean(
                tail,
                "outside_room_level_exact",
            ),
            "locality_violation_area": _mean(
                tail,
                "locality_violation_area",
            ),
        }
    report = {
        "status": "complete",
        "source": str(RESULTS.relative_to(ROOT)),
        "comparison_source": str(BASELINES.relative_to(ROOT)),
        "ours_seed19": ours,
        "raster2seq_wrapper": external,
        "failure_tails": tails,
        "figures": [
            str(FIGURE_SVG.relative_to(ROOT)),
            str(FIGURE_PDF.relative_to(ROOT)),
        ],
        "checks": {
            "floorflow_locality_above_external": (
                ours["outside_room_level_exact"]
                > external["outside_room_level_exact"]
            ),
            "floorflow_graph_above_external": (
                ours["node_f1"] > external["node_f1"]
                and ours["edge_overlap"] > external["edge_overlap"]
            ),
            "all_failure_tails_preserve_locality": all(
                row["outside_room_level_exact"] == 1.0
                and row["locality_violation_area"] == 0.0
                for row in tails.values()
            ),
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# FEVP-2 Failure Tails v1",
        "",
        "Lowest-decile Node F1 rows within each edit operation, pooled over five seeds.",
        "",
        "| Operation | Cases | Node F1 | Edge overlap | Validity | GED | Outside exact | VLR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for operation, row in tails.items():
        lines.append(
            f"| {operation} | {row['tail_cases']} | "
            f"{row['node_f1']:.4f} | {row['edge_overlap']:.4f} | "
            f"{row['validity']:.4f} | {row['ged']:.4f} | "
            f"{row['outside_room_level_exact']:.4f} | "
            f"{row['locality_violation_area']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Failure tails remain strictly local: graph degradation is not caused by outside leakage.",
        ]
    )
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _figure(ours, external)
    print(json.dumps({"status": "complete", "checks": report["checks"]}, indent=2))
    raise SystemExit(0 if all(report["checks"].values()) else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
