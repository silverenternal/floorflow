#!/usr/bin/env python3
"""Plot the external PPA boundary and mechanism evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "blue": "#2f6fb6",
    "green": "#3f8f5f",
    "red": "#b84a5a",
    "amber": "#b77a00",
    "gray": "#526173",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "outputs/figures/chip_failure_cases_v1",
    )
    return parser.parse_args()


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def bar_colors(values: list[float]) -> list[str]:
    return [
        COLORS["green"] if value < 0 else COLORS["red"]
        for value in values
    ]


def main() -> None:
    args = parse_args()
    v1 = read("outputs/reports/chip_external_ppa_subset_v1.json")
    holdout = read("outputs/reports/chip_external_ppa_holdout_v1.json")
    ablation = read("outputs/reports/chip_external_ppa_ablation_v1.json")
    v1_mech = read("outputs/reports/chip_p2b_mechanism_trace_v1.json")
    holdout_mech = read("outputs/reports/chip_holdout_mechanism_trace_v1.json")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    fig.subplots_adjust(wspace=0.58, bottom=0.20)

    case_entries = []
    for case, value in sorted(v1["case_relative_wirelength_means"].items()):
        case_entries.append((f"v1 | {case.split('/')[-1]}", value * 100, "3/3"))
    for case, value in sorted(holdout["case_relative_wirelength_means"].items()):
        nickname = case.split("/")[-1]
        count = sum(
            row["status"] == "completed"
            for row in holdout["comparisons"]
            if row["case"] == case
        )
        case_entries.append((f"holdout | {nickname}", value * 100, f"{count}/3"))

    labels = [entry[0] for entry in case_entries]
    values = [entry[1] for entry in case_entries]
    counts = [entry[2] for entry in case_entries]
    y = list(range(len(labels)))
    axes[0].barh(y, values, color=bar_colors(values), height=0.68)
    axes[0].axvline(0, color=COLORS["gray"], linewidth=1.0)
    axes[0].axvline(-2, color=COLORS["amber"], linestyle="--", linewidth=1.3)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Routed WL change vs control (%)")
    axes[0].set_title("A. Per-case seed means")
    axes[0].grid(axis="x", alpha=0.25)
    for index, (value, count) in enumerate(zip(values, counts)):
        label_x = value + 0.8 if value >= 0 else 0.5
        axes[0].text(
            label_x,
            index,
            f"{value:+.1f}%  {count}",
            va="center",
            ha="left",
            fontsize=6.5,
        )

    arms = ablation["summaries"]
    arm_labels = [row["arm"] for row in arms]
    arm_values = [row["relative_wirelength_vs_full"] * 100 for row in arms]
    y2 = list(range(len(arm_labels)))
    axes[1].barh(
        y2,
        arm_values,
        color=bar_colors(arm_values),
        height=0.62,
    )
    axes[1].axvline(0, color=COLORS["gray"], linewidth=1.0)
    axes[1].set_yticks(y2, [label.replace("_", " ") for label in arm_labels])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean routed WL vs full (%)")
    axes[1].set_title("B. Three-case ablation")
    axes[1].grid(axis="x", alpha=0.25)
    for index, value in enumerate(arm_values):
        label_x = value + 0.18 if value >= 0 else 0.12
        axes[1].text(
            label_x,
            index,
            f"{value:+.1f}%",
            va="center",
            ha="left",
            fontsize=6.5,
        )

    route_rows = [
        ("v1 fallback / scale", v1_mech["legalizer_route_strata"][
            "('shelf_pack_fallback', 'scale_pack')"
        ]["routed_wirelength_relative_mean"]),
        ("v1 scale / scale", v1_mech["legalizer_route_strata"][
            "('scale_pack', 'scale_pack')"
        ]["routed_wirelength_relative_mean"]),
        ("v1 fallback / fallback", v1_mech["legalizer_route_strata"][
            "('shelf_pack_fallback', 'shelf_pack_fallback')"
        ]["routed_wirelength_relative_mean"]),
        ("holdout scale / scale", holdout_mech["legalizer_route_strata"][
            "('scale_pack', 'scale_pack')"
        ]["routed_wirelength_relative_mean"]),
        ("holdout shelf / shelf", holdout_mech["legalizer_route_strata"][
            "('shelf_pack_fallback', 'shelf_pack_fallback')"
        ]["routed_wirelength_relative_mean"]),
    ]
    route_labels = [row[0] for row in route_rows]
    route_values = [row[1] * 100 for row in route_rows]
    y3 = list(range(len(route_labels)))
    axes[2].barh(
        y3,
        route_values,
        color=bar_colors(route_values),
        height=0.62,
    )
    axes[2].axvline(0, color=COLORS["gray"], linewidth=1.0)
    axes[2].axvline(-2, color=COLORS["amber"], linestyle="--", linewidth=1.3)
    axes[2].set_yticks(y3, route_labels)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Routed WL change (%)")
    axes[2].set_title("C. Legalizer-route strata")
    axes[2].grid(axis="x", alpha=0.25)
    for index, value in enumerate(route_values):
        label_x = value + 0.4 if value >= 0 else 0.4
        axes[2].text(
            label_x,
            index,
            f"{value:+.1f}%",
            va="center",
            ha="left",
            fontsize=6.5,
        )

    fig.text(
        0.5,
        0.01,
        "Dashed amber line: predeclared -2% WL gate. Counts under panel A show completed seeds.",
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLORS["gray"],
    )
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".svg"))
    fig.savefig(args.output_stem.with_suffix(".pdf"))
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=220)
    print(str(args.output_stem))


if __name__ == "__main__":
    main()
