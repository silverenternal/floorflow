#!/usr/bin/env python3
"""Quantify strict-locality protocol boundaries and wall-break mechanisms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--complexity-rows",
        type=Path,
        default=Path("outputs/reports/fp_complexity_strata_rows_fiveseed_v1.json"),
    )
    parser.add_argument(
        "--factorial-rows",
        type=Path,
        default=Path("outputs/reports/fp_safety_factorial_2x2_rows_v1.json"),
    )
    parser.add_argument(
        "--failure",
        type=Path,
        default=Path("outputs/reports/fp_failure_analysis_v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_protocol_boundary_v1.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_protocol_boundary.md"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("paper/figures/fp_weak_signal_diagnostics.svg"),
    )
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(rows: list[dict]) -> dict:
    keys = (
        "delta_iou",
        "delta_mse",
        "outside_exact_ratio",
        "outside_room_exact_ratio",
        "validity",
        "conditional_watertight",
        "node_f1",
        "ged",
    )
    return {
        key: float(np.mean([row[key] for row in rows])) for key in keys
    }


def _table(report: dict) -> str:
    def number(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.6f}"

    lines = [
        "# Strict-Locality Protocol Boundary",
        "",
        "| Boundary | N | Delta-IoU | Delta-MSE | Outside exact | Room exact | Validity | Watertight | NodeF1 | GED |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["weak_boundaries"].items():
        metrics = values["metrics"]
        lines.append(
            f"| {name} | {values['cases']} | {number(metrics['delta_iou'])} | "
            f"{number(metrics['delta_mse'])} | "
            f"{number(metrics['outside_exact_ratio'])} | "
            f"{number(metrics['outside_room_exact_ratio'])} | "
            f"{number(metrics['validity'])} | "
            f"{number(metrics['conditional_watertight'])} | "
            f"{number(metrics['node_f1'])} | {number(metrics['ged'])} |"
        )
    lines.extend(
        [
            "",
            "| Arm | Wall breaks | Watertight | Outside exact |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm, values in report["wall_breaks"].items():
        lines.append(
            f"| {arm} | {values['count']} | {values['watertight']:.6f} | "
            f"{values['outside_exact']:.6f} |"
        )
    lines.extend(
        [
            "",
            report["boundary_statement"],
        ]
    )
    return "\n".join(lines) + "\n"


def _plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    names = list(report["weak_boundaries"])
    watertight = [
        report["weak_boundaries"][name]["metrics"]["conditional_watertight"]
        for name in names
    ]
    exact = [
        report["weak_boundaries"][name]["metrics"]["outside_exact_ratio"]
        for name in names
    ]
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(6.6, 2.8))
    axis.bar(x - 0.18, watertight, 0.36, color="#b84a5a", label="Watertight")
    axis.bar(x + 0.18, exact, 0.36, color="#16837a", label="Outside exact")
    axis.set_xticks(x, labels=[name.replace("_", "\n") for name in names])
    axis.set_ylim(0.8, 1.01)
    axis.set_ylabel("Score")
    axis.set_title("Weak-signal protocol boundaries")
    axis.axhline(0.95, color="#526173", linestyle="--", linewidth=1.0)
    axis.legend(frameon=False, ncol=2, loc="lower left")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    complexity = _read(args.complexity_rows)["rows"]
    complexity_index = {
        (row["seed"], row["operation"], row["name"]): row
        for row in complexity
    }
    factorial = _read(args.factorial_rows)["rows"]
    failure = _read(args.failure)
    weak_room_rows = [
        row for row in complexity if row["target_node_count"] <= 4
    ]
    weak_interface_rows = [
        row
        for row in complexity
        if row["interface_complexity"] in {"opening_only", "neither"}
    ]
    seed19 = {
        (row["operation"], row["name"]): row
        for row in factorial
        if row["seed"] == 19
    }
    signal_values = np.asarray(
        [
            row["arms"]["control_on_guard_off"]["inside_change_fraction"]
            for row in seed19.values()
        ]
    )
    signal_cut = float(np.quantile(signal_values, 1.0 / 3.0))
    weak_signal_rows = [
        seed19[(row["operation"], row["name"])]
        for row in complexity
        if row["seed"] == 19
        and seed19[(row["operation"], row["name"])]["arms"][
            "control_on_guard_off"
        ]["inside_change_fraction"]
        <= signal_cut
    ]
    weak_boundaries = {
        "low_room_count_0_4": {
            "cases": len(weak_room_rows),
            "metrics": _metrics(weak_room_rows),
        },
        "weak_interface_opening_or_none": {
            "cases": len(weak_interface_rows),
            "metrics": _metrics(weak_interface_rows),
        },
        "low_edit_signal_bottom_tercile": {
            "cases": len(weak_signal_rows),
            "metrics": {
                key: float(
                    np.mean(
                        [
                            row["arms"]["control_on_guard_off"][key]
                            for row in weak_signal_rows
                        ]
                    )
                )
                for key in (
                    "delta_iou",
                    "delta_mse",
                    "outside_pixel_exact",
                    "outside_room_exact",
                    "validity",
                    "conditional_watertight",
                    "junction_f1",
                )
            },
        },
    }
    # Normalize the seed19 dynamic-signal keys to the common naming.
    dynamic = weak_boundaries["low_edit_signal_bottom_tercile"]["metrics"]
    weak_boundaries["low_edit_signal_bottom_tercile"]["metrics"] = {
        "delta_iou": dynamic["delta_iou"],
        "delta_mse": dynamic["delta_mse"],
        "outside_exact_ratio": dynamic["outside_pixel_exact"],
        "outside_room_exact_ratio": dynamic["outside_room_exact"],
        "validity": dynamic["validity"],
        "conditional_watertight": dynamic["conditional_watertight"],
        "node_f1": None,
        "ged": None,
    }
    wall_breaks = {}
    for arm in (
        "control_off_guard_off",
        "control_on_guard_off",
        "control_off_guard_on",
        "control_on_guard_on",
    ):
        count = 0
        for row in factorial:
            complexity_row = complexity_index[
                (row["seed"], row["operation"], row["name"])
            ]
            values = row["arms"][arm]
            if (
                row["target_watertight"]
                and values["conditional_watertight"] == 0.0
                and complexity_row["interface_complexity"] in {"wall_only", "both"}
                and values["junction_f1"] < 0.9
            ):
                count += 1
        wall_breaks[arm] = {
            "count": count,
            "watertight": float(
                np.mean([row["arms"][arm]["conditional_watertight"] for row in factorial])
            ),
            "outside_exact": float(
                np.mean([row["arms"][arm]["outside_pixel_exact"] for row in factorial])
            ),
        }
    report = {
        "status": "complete",
        "signal_definition": {
            "changed_pixel_fraction_cut": signal_cut,
            "changed_pixel_fraction_source": "seed19 control-on/guard-off inside change",
            "static_features": [
                "target room count",
                "wall crossing count",
                "opening crossing count",
                "interface complexity",
            ],
        },
        "weak_boundaries": weak_boundaries,
        "wall_breaks": wall_breaks,
        "failure_reference": {
            "edit_boundary_wall_break_cases": failure["families"][
                "edit_boundary_wall_break"
            ]["cases"],
            "definition": failure["families"]["edit_boundary_wall_break"][
                "definition"
            ],
        },
        "answers": {
            "low_signal_bins_have_lower_watertightness": (
                weak_boundaries["low_edit_signal_bottom_tercile"]["metrics"][
                    "conditional_watertight"
                ]
                < float(
                    np.mean(
                        [
                            row["arms"]["control_on_guard_off"][
                                "conditional_watertight"
                            ]
                            for row in factorial
                            if row["seed"] == 19
                        ]
                    )
                )
            ),
            "weak_interfaces_have_lower_watertightness": (
                weak_boundaries["weak_interface_opening_or_none"]["metrics"][
                    "conditional_watertight"
                ]
                < 0.95
            ),
            "control_gate_preserves_outside_exact_in_weak_signal": (
                weak_boundaries["low_edit_signal_bottom_tercile"]["metrics"][
                    "outside_exact_ratio"
                ]
                == 1.0
            ),
        },
        "boundary_statement": (
            "When edit signals are sparse or weak interfaces provide few "
            "topological constraints, conditional watertightness can fall below "
            "0.95. Strict outside preservation remains intact under the control "
            "gate. This is a validity boundary of the current protocol, not a "
            "redefinition of the locality guarantee."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_table(report), encoding="utf-8")
    _plot(report, args.figure)
    print(json.dumps({
        "weak_boundaries": weak_boundaries,
        "wall_breaks": wall_breaks,
        "answers": report["answers"],
    }, indent=2))


if __name__ == "__main__":
    main()
