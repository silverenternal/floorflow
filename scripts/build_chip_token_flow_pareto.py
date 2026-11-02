#!/usr/bin/env python3
"""Build measured NFE/refiner budget-quality Pareto evidence and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "candidate": "#16837a",
    "baseline": "#526173",
    "threshold": "#b77a00",
    "risk": "#b84a5a",
    "grid": "#c7ced8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mechanism",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_token_flow_mechanism_diagnostics_v1.json",
    )
    parser.add_argument(
        "--budget",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_budget_curve_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_pareto_v1.json",
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=ROOT / "outputs/figures",
    )
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def points(mechanism: dict, budget: dict) -> dict:
    nfe = {}
    for step in (1, 2, 4, 8):
        curve = mechanism["nfe_curves"][str(step)]
        nfe[str(step)] = {
            "candidate_cost": curve["candidate_mean"]["cost"],
            "candidate_v_rel": curve["candidate_mean"]["v_rel"],
            "baseline_cost": curve["baseline_mean"]["cost"],
            "baseline_v_rel": curve["baseline_mean"]["v_rel"],
        }
    refiner = {}
    for step, values in budget["refiner"]["curve"]["candidate"].items():
        baseline = budget["refiner"]["curve"]["baseline"][step]
        refiner[step] = {
            "candidate_cost": values["cost"],
            "candidate_v_rel": values["v_rel"],
            "baseline_cost": baseline["cost"],
            "baseline_v_rel": baseline["v_rel"],
        }
    return {"nfe": nfe, "refiner": refiner}


def style_axis(axis, *, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontweight="semibold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=3, width=0.8)


def line_pair(
    axis,
    x: list[float],
    candidate: list[float],
    baseline: list[float],
    *,
    xscale: str = "linear",
) -> None:
    axis.plot(
        x,
        baseline,
        color=COLORS["baseline"],
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="global-slot baseline",
    )
    axis.plot(
        x,
        candidate,
        color=COLORS["candidate"],
        marker="o",
        markersize=4,
        linewidth=1.9,
        label="graph-token flow",
    )
    axis.set_xscale(xscale)
    if xscale == "log":
        axis.set_xticks(x)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
        axis.minorticks_off()


def annotate_equal_quality(
    axis,
    *,
    candidate_x: float,
    candidate_y: float,
    baseline_x: float,
    baseline_y: float,
) -> None:
    axis.axhline(
        baseline_y,
        color=COLORS["threshold"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )
    axis.axvline(
        baseline_x,
        color=COLORS["threshold"],
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
    )
    axis.scatter(
        [candidate_x],
        [candidate_y],
        marker="*",
        s=85,
        color=COLORS["threshold"],
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    axis.annotate(
        f"{candidate_x:g} vs {baseline_x:g}",
        xy=(candidate_x, candidate_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=7,
        color=COLORS["threshold"],
    )


def build_figure(data: dict, budget: dict, path_stem: Path) -> None:
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
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), constrained_layout=True)
    nfe = data["nfe"]
    nfe_x = [float(step) for step in nfe]
    refiner = data["refiner"]
    refiner_x = [float(step) for step in refiner]

    specs = (
        (axes[0, 0], nfe_x, nfe, "candidate_cost", "baseline_cost", "NFE: cost (lower is better)", "NFE"),
        (axes[0, 1], nfe_x, nfe, "candidate_v_rel", "baseline_v_rel", "NFE: V_rel (lower is better)", "NFE"),
        (axes[1, 0], refiner_x, refiner, "candidate_cost", "baseline_cost", "Refiner: cost (lower is better)", "Refiner steps"),
        (axes[1, 1], refiner_x, refiner, "candidate_v_rel", "baseline_v_rel", "Refiner: V_rel (lower is better)", "Refiner steps"),
    )
    for axis, x, source, candidate_key, baseline_key, title, xlabel in specs:
        y_candidate = [source[key][candidate_key] for key in source]
        y_baseline = [source[key][baseline_key] for key in source]
        line_pair(
            axis,
            x,
            y_candidate,
            y_baseline,
            xscale="log" if xlabel == "NFE" else "linear",
        )
        style_axis(axis, title=title, xlabel=xlabel, ylabel="Pooled metric")
        low = min(min(y_candidate), min(y_baseline))
        high = max(max(y_candidate), max(y_baseline))
        margin = max((high - low) * 0.18, 1.0e-3)
        axis.set_ylim(low - margin, high + margin)
        axis.text(
            0.99,
            0.02,
            "zoomed measured range",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            color=COLORS["baseline"],
        )

    annotate_equal_quality(
        axes[0, 0],
        candidate_x=1,
        candidate_y=nfe["1"]["candidate_cost"],
        baseline_x=8,
        baseline_y=nfe["8"]["baseline_cost"],
    )
    annotate_equal_quality(
        axes[0, 1],
        candidate_x=1,
        candidate_y=nfe["1"]["candidate_v_rel"],
        baseline_x=8,
        baseline_y=nfe["8"]["baseline_v_rel"],
    )
    annotate_equal_quality(
        axes[1, 0],
        candidate_x=100,
        candidate_y=refiner["100"]["candidate_cost"],
        baseline_x=400,
        baseline_y=refiner["400"]["baseline_cost"],
    )
    annotate_equal_quality(
        axes[1, 1],
        candidate_x=100,
        candidate_y=refiner["100"]["candidate_v_rel"],
        baseline_x=400,
        baseline_y=refiner["400"]["baseline_v_rel"],
    )
    axes[1, 1].annotate(
        "50 steps:\ncost better, V_rel above baseline",
        xy=(50, refiner["50"]["candidate_v_rel"]),
        xytext=(170, 0.073),
        fontsize=6.5,
        color=COLORS["risk"],
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["risk"],
            "linewidth": 0.8,
        },
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.07),
    )
    figure.suptitle(
        "Equal-quality inference budget on FloorSet-100",
        y=1.12,
        fontsize=11,
        fontweight="bold",
    )
    figure.text(
        0.5,
        1.055,
        "Stars mark equal-quality settings; the 50-step point retains a V_rel deficit.",
        ha="center",
        fontsize=7,
        color=COLORS["baseline"],
    )
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_stem.with_suffix(".svg"))
    figure.savefig(path_stem.with_suffix(".pdf"))
    figure.savefig(path_stem.with_suffix(".png"), dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    mechanism = read(args.mechanism)
    budget = read(args.budget)
    data = points(mechanism, budget)
    report = {
        "status": "complete",
        "claim": (
            "Graph-token flow reaches declared baseline quality with fewer "
            "measured NFE or refiner steps."
        ),
        "points": data,
        "equal_quality": {
            "nfe": budget["nfe"],
            "refiner": {
                "candidate_steps": budget["refiner"]["candidate_minimum_steps"],
                "baseline_steps": 400,
            },
        },
        "scope": {
            "nfe": "three-seed pooled raw legalized means",
            "refiner": "seed-19 NFE4 controlled curve",
            "primary_endpoint_unchanged": "NFE4 + 400 refiner steps",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    build_figure(data, budget, args.figure_root / "chip_token_flow_pareto_v1")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
