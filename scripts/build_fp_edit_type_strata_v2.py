#!/usr/bin/env python3
"""Build edit-type strata and GED diagnostics for the strict-locality protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
GED_PARTS = (
    "node_substitution",
    "node_deletion",
    "node_insertion",
    "edge_deletion",
    "edge_insertion",
)


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fp_edit_type_strata_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fp_edit_type_strata_v2.md",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "paper/figures/fp_edit_type_strata_v2.svg",
    )
    return parser.parse_args()


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    edit = read(
        "outputs/reports/fp_edit_detection_test503_controlgated_fiveseed_v1.json"
    )
    structural = read("outputs/reports/fp_housemind_structural_alignment_v1.json")
    typed = read("outputs/reports/fp_typed_edit_graph_ablation_v1.json")
    response = read(
        "outputs/reports/fp_local_edit_response_test503_multiseed_v1.json"
    )
    rows = {}
    for operation in OPERATIONS:
        edit_values = edit["operations"][operation]
        structural_values = {
            name: summary(
                [
                    seed["operations"][operation][
                        "after_prediction_vs_after_target"
                    ][name]
                    for seed in structural["per_seed"].values()
                ]
            )
            for name in (
                "node_f1",
                "edge_overlap",
                "ged",
                "exact",
                *GED_PARTS,
            )
        }
        response_values = response["operations"][operation]["summary"]
        typed_values = typed["operations"][operation]["typed_edit_graph"]
        decomposition_sum = sum(
            structural_values[name]["mean"] for name in GED_PARTS
        )
        rows[operation] = {
            "cases_per_seed": round(np.mean(edit_values["cases_per_seed"])),
            "delta_iou": summary(edit_values["local_delta_iou"]),
            "delta_mse": summary(edit_values["local_delta_mse"]),
            "outside_pixel_exact": typed_values["outside_pixel_exact"],
            "outside_room_exact": typed_values["outside_room_exact"],
            "conditional_watertight": response_values[
                "conditional_watertight"
            ],
            "inside_response_fraction": response_values[
                "inside_response_fraction"
            ],
            **structural_values,
            "ged_components_mean_sum": decomposition_sum,
            "ged_reconstruction_error": (
                structural_values["ged"]["mean"] - decomposition_sum
            ),
        }
    coverage = {
        "add_or_remove_room": {
            "status": "available",
            "matched_operations": ["room_add", "room_remove"],
            "boundary": (
                "Room addition is implemented as a split-to-new-type control "
                "edit; room removal is implemented as a merge-to-neighbor edit."
            ),
        },
        "move_wall": {
            "status": "unavailable",
            "matched_operations": [],
            "boundary": "No matched wall-move controls or predictions were evaluated.",
        },
        "change_room_function": {
            "status": "available",
            "matched_operations": ["room_type"],
            "boundary": "One room component is relabeled to another program type.",
        },
        "expand_or_shrink_room": {
            "status": "unavailable",
            "matched_operations": [],
            "boundary": "No isolated boundary-resize edit set was evaluated.",
        },
        "merge_or_split_room": {
            "status": "partially_available",
            "matched_operations": ["room_remove", "room_add"],
            "boundary": (
                "The frozen set contains merge-to-neighbor and split-to-new-type "
                "edits, but not a complete merge/split operator family."
            ),
        },
        "multi_step_combined_edit": {
            "status": "unavailable",
            "matched_operations": [],
            "boundary": "No sequentially composed multi-step control set was evaluated.",
        },
    }
    dominant = max(rows, key=lambda name: rows[name]["ged"]["mean"])
    report = {
        "status": "complete",
        "scope": "CubiCasa test503 available local-edit operations, five seeds",
        "seeds": edit["seeds"],
        "operations": rows,
        "coverage_map": coverage,
        "ged_diagnosis": {
            "highest_operation": dominant,
            "highest_ged": rows[dominant]["ged"],
            "highest_components": {
                name: rows[dominant][name]["mean"] for name in GED_PARTS
            },
            "explanation": (
                "Room removal dominates the aggregate GED tail because merging "
                "one room into a neighbor creates edge insertions/deletions in "
                "the unique room-type graph. Its GED decomposition is explicitly "
                "reported; opening and furniture edits are near zero at the node "
                "level."
            ),
        },
        "checks": {
            "five_numeric_operations_present": set(rows) == set(OPERATIONS),
            "all_available_outside_exact_one": all(
                row["outside_pixel_exact"] == 1.0
                and row["outside_room_exact"] == 1.0
                for row in rows.values()
            ),
            "ged_components_reconcile": all(
                abs(row["ged_reconstruction_error"]) < 1.0e-9
                for row in rows.values()
            ),
            "room_remove_is_ged_tail": dominant == "room_remove",
            "unavailable_operations_have_no_numeric_claim": all(
                not item["matched_operations"]
                for item in coverage.values()
                if item["status"] == "unavailable"
            ),
        },
        "boundaries": [
            "No wall-move, resize, or sequential multi-step numeric claim.",
            "Merge/split coverage is partial and follows the frozen edit definitions.",
            "This report does not change the primary protocol claim or add SOTA language.",
        ],
    }
    failures = [name for name, passed in report["checks"].items() if not passed]
    if failures:
        raise ValueError(f"edit-type checks failed: {failures}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_table(report, args.table)
    _write_figure(report, args.figure)
    print(
        json.dumps(
            {
                "status": report["status"],
                **report["checks"],
                "highest_ged_operation": dominant,
            },
            indent=2,
        )
    )


def _write_table(report: dict, path: Path) -> None:
    lines = [
        "# Edit-Type Strata v2",
        "",
        "| Operation | Cases/seed | Delta-IoU | Delta-MSE | Outside exact | Watertight | Node F1 | Edge | GED |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for operation, row in report["operations"].items():
        lines.append(
            f"| {operation} | {row['cases_per_seed']} | "
            f"{row['delta_iou']['mean']:.6f} "
            f"({row['delta_iou']['min']:.4f}-{row['delta_iou']['max']:.4f}) | "
            f"{row['delta_mse']['mean']:.9f} | "
            f"{row['outside_pixel_exact']:.6f} | "
            f"{row['conditional_watertight']['mean']:.6f} | "
            f"{row['node_f1']['mean']:.6f} | "
            f"{row['edge_overlap']['mean']:.6f} | "
            f"{row['ged']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Requested Operation Coverage",
            "",
            "| Requested type | Status | Matched evidence |",
            "|---|---|---|",
        ]
    )
    for name, item in report["coverage_map"].items():
        evidence = ", ".join(item["matched_operations"]) or "none"
        lines.append(f"| {name} | {item['status']} | {evidence} |")
    lines.extend(
        [
            "",
            "## GED Diagnosis",
            "",
            (
                f"`{report['ged_diagnosis']['highest_operation']}` has the highest "
                f"mean GED ({report['ged_diagnosis']['highest_ged']['mean']:.6f}). "
                "The decomposition separates node substitutions/deletions/insertions "
                "from edge deletions/insertions; edge edits dominate the tail."
            ),
            "",
            (
                "No numeric claim is made for wall movement, room resizing, or "
                "sequential multi-step editing because matched rows are unavailable."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figure(report: dict, path: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    operations = list(report["operations"])
    labels = [
        operation.replace("_", "\n")
        for operation in operations
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    colors = {
        "node_substitution": "#2f6fb6",
        "node_deletion": "#7254a7",
        "node_insertion": "#16837a",
        "edge_deletion": "#b77a00",
        "edge_insertion": "#b84a5a",
    }
    bottom = np.zeros(len(operations))
    for component in GED_PARTS:
        values = np.asarray(
            [
                report["operations"][operation][component]["mean"]
                for operation in operations
            ]
        )
        axes[0].bar(
            labels,
            values,
            bottom=bottom,
            color=colors[component],
            label=component.replace("_", " "),
        )
        bottom += values
    axes[0].set_ylabel("Mean GED components")
    axes[0].set_title("A  GED decomposition by edit type", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")

    x = np.arange(len(operations))
    iou = [
        report["operations"][operation]["delta_iou"]["mean"]
        for operation in operations
    ]
    watertight = [
        report["operations"][operation]["conditional_watertight"]["mean"]
        for operation in operations
    ]
    bars = axes[1].bar(x, iou, color="#526173", label="Delta-IoU")
    axis_right = axes[1].twinx()
    axis_right.plot(
        x,
        watertight,
        color="#b77a00",
        marker="o",
        label="Watertight",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.75, 1.01)
    axis_right.set_ylim(0.90, 1.00)
    axes[1].set_ylabel("Delta-IoU")
    axis_right.set_ylabel("Conditional watertight")
    axes[1].set_title("B  Quality and topology", loc="left", fontweight="bold")
    for bar, value in zip(bars, iou, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    for axis in (axes[0], axes[1], axis_right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(True, axis="y", color="#c7ced8", alpha=0.55)
    figure.suptitle(
        "Room removal explains the GED tail; protocol quality remains stable",
        x=0.02,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


__all__ = ["main"]


if __name__ == "__main__":
    main()
