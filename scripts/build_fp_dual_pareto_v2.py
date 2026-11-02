#!/usr/bin/env python3
"""Build matched quality-locality and safety-efficiency Pareti evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fp_dual_pareto_v2.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "paper/figures/fp_dual_pareto_v2.svg",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fp_dual_pareto_v2.md",
    )
    return parser.parse_args()


def pareto(
    points: dict[str, dict],
    *,
    x_key: str,
    y_key: str,
    maximize_x: bool,
    maximize_y: bool,
) -> list[str]:
    frontier = []
    for name, point in points.items():
        dominated = False
        for other_name, other in points.items():
            if other_name == name:
                continue
            x_better = (
                other[x_key] >= point[x_key]
                if maximize_x
                else other[x_key] <= point[x_key]
            )
            y_better = (
                other[y_key] >= point[y_key]
                if maximize_y
                else other[y_key] <= point[y_key]
            )
            x_strict = (
                other[x_key] > point[x_key]
                if maximize_x
                else other[x_key] < point[x_key]
            )
            y_strict = (
                other[y_key] > point[y_key]
                if maximize_y
                else other[y_key] < point[y_key]
            )
            if x_better and y_better and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return frontier


def main() -> None:
    args = parse_args()
    typed = read("outputs/reports/fp_typed_edit_graph_ablation_v1.json")
    factorial = read("outputs/reports/fp_safety_factorial_2x2_v1.json")
    matched = read("outputs/reports/fp_pareto_matched_v1.json")
    seed = "19"
    typed_summary = typed["per_seed"][seed]["summary"]
    control = factorial["summary"]["control_on_guard_off"]
    raw = factorial["summary"]["control_off_guard_off"]
    method_points = {
        "FloorFlow control-gated": {
            "delta_iou": control["delta_iou"],
            "outside_pixel_exact": control["outside_pixel_exact"],
            "outside_room_exact": control["outside_room_exact"],
            "conditional_watertight": control["conditional_watertight"],
            "validity": control["validity"],
        },
        "FloorFlow raw renderer": {
            "delta_iou": raw["delta_iou"],
            "outside_pixel_exact": raw["outside_pixel_exact"],
            "outside_room_exact": raw["outside_room_exact"],
            "conditional_watertight": raw["conditional_watertight"],
            "validity": raw["validity"],
        },
    }
    for name, row in typed_summary.items():
        method_points[name] = {
            "delta_iou": row["delta_iou"],
            "outside_pixel_exact": row["outside_pixel_exact"],
            "outside_room_exact": row["outside_room_exact"],
            "conditional_watertight": row["conditional_watertight"],
            "validity": row["validity"],
            "routing_latency_ms": row["routing_latency_ms"],
        }
    for name in ("Raster2Seq native", "Raster2Seq outside-frozen redraw"):
        row = next(
            item
            for item in matched["matched_methods"]
            if item["method"] == name
        )
        method_points[name] = {
            "delta_iou": row["delta_iou"],
            "outside_pixel_exact": row["outside_pixel_exact"],
            "outside_room_exact": row["outside_room_exact"],
            "conditional_watertight": None,
            "validity": None,
        }
    for point in method_points.values():
        factors = (
            point["outside_pixel_exact"],
            point["outside_room_exact"],
            point["validity"],
            point["conditional_watertight"],
        )
        point["verifiability"] = (
            float(factors[0] * factors[1] * factors[2] * factors[3])
            if all(value is not None for value in factors)
            else None
        )
    safety_points = {
        name: {
            "routing_latency_ms": point["routing_latency_ms"],
            "verifiability": point["verifiability"],
            "conditional_watertight": point["conditional_watertight"],
        }
        for name, point in method_points.items()
        if point.get("routing_latency_ms") is not None
        and point["verifiability"] is not None
    }
    quality_frontier = pareto(
        method_points,
        x_key="outside_pixel_exact",
        y_key="delta_iou",
        maximize_x=True,
        maximize_y=True,
    )
    safety_frontier = pareto(
        safety_points,
        x_key="routing_latency_ms",
        y_key="verifiability",
        maximize_x=False,
        maximize_y=True,
    )
    report = {
        "status": "complete",
        "scope": (
            "CubiCasa test503 seed19 matched edit rows; Raster2Seq uses the "
            "matching external baseline rows"
        ),
        "model_nfe_policy": (
            "Model NFE is fixed across typed-routing safety points; the x-axis "
            "measures deterministic post-processing routing latency only."
        ),
        "quality_locality": {
            "x": "outside pixel exact (maximize)",
            "y": "Delta-IoU (maximize)",
            "points": method_points,
            "frontier": quality_frontier,
        },
        "safety_efficiency": {
            "x": "routing latency ms/case (minimize)",
            "y": "Verifiability (maximize)",
            "points": safety_points,
            "frontier": safety_frontier,
        },
        "checks": {
            "same_seed19_rows": typed["cases_per_seed"] == 2479,
            "quality_frontier_nonempty": bool(quality_frontier),
            "safety_frontier_nonempty": bool(safety_frontier),
            "missing_verifiability_not_imputed": all(
                point["verifiability"] is None
                for name, point in method_points.items()
                if name.startswith("Raster2Seq")
            ),
            "fixed_nfe_disclosed": True,
        },
    }
    failures = [name for name, passed in report["checks"].items() if not passed]
    if failures:
        raise ValueError(f"dual Pareto checks failed: {failures}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_table(report, args.table)
    _write_figure(report, args.figure)
    print(
        json.dumps(
            {
                "status": report["status"],
                "quality_frontier": quality_frontier,
                "safety_frontier": safety_frontier,
            },
            indent=2,
        )
    )


def _write_table(report: dict, path: Path) -> None:
    lines = [
        "# Dual Pareto Evidence v2",
        "",
        (
            "Model NFE is fixed across typed-routing points. The second x-axis "
            "measures deterministic post-processing latency, not model sampling."
        ),
        "",
        "## Quality-Locality",
        "",
        "| Method | Outside exact | Delta-IoU | Frontier |",
        "|---|---:|---:|---|",
    ]
    for name, point in report["quality_locality"]["points"].items():
        lines.append(
            f"| {name} | {point['outside_pixel_exact']:.6f} | "
            f"{point['delta_iou']:.6f} | "
            f"{'yes' if name in report['quality_locality']['frontier'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Safety-Efficiency",
            "",
            "| Method | Routing ms/case | Verifiability | Frontier |",
            "|---|---:|---:|---|",
        ]
    )
    for name, point in report["safety_efficiency"]["points"].items():
        lines.append(
            f"| {name} | {point['routing_latency_ms']:.3f} | "
            f"{point['verifiability']:.6f} | "
            f"{'yes' if name in report['safety_efficiency']['frontier'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            (
                "Raster2Seq rows are reconstruction/editing-wrapper comparisons "
                "and are excluded from the safety-efficiency panel because the "
                "required verifiability factors are not jointly reported."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figure(report: dict, path: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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
    colors = {
        "FloorFlow control-gated": "#16837a",
        "FloorFlow raw renderer": "#526173",
        "typed_edit_graph": "#3f8f5f",
        "dense_cross_attention": "#2f6fb6",
        "no_edit_gating": "#b77a00",
        "no_structural_graph": "#b84a5a",
        "Raster2Seq native": "#7254a7",
        "Raster2Seq outside-frozen redraw": "#8c6d31",
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), constrained_layout=True)

    quality = report["quality_locality"]["points"]
    for name, point in quality.items():
        axes[0].scatter(
            point["outside_pixel_exact"],
            point["delta_iou"],
            s=50 if name in report["quality_locality"]["frontier"] else 32,
            color=colors[name],
            edgecolor="black" if name in report["quality_locality"]["frontier"] else "white",
            linewidth=0.8,
        )
    frontier = report["quality_locality"]["frontier"][0]
    point = quality[frontier]
    axes[0].annotate(
        "Frontier",
        (point["outside_pixel_exact"], point["delta_iou"]),
        xytext=(-45, -16),
        textcoords="offset points",
        fontsize=7,
        color="#16837a",
        arrowprops={"arrowstyle": "-", "linewidth": 0.8, "color": "#16837a"},
    )
    axes[0].set_xlabel("Outside pixel exact (higher is better)")
    axes[0].set_ylabel("Delta-IoU (higher is better)")
    axes[0].set_title("A  Quality-locality", loc="left", fontweight="bold")

    safety = report["safety_efficiency"]["points"]
    for name, point in safety.items():
        axes[1].scatter(
            point["routing_latency_ms"],
            point["verifiability"],
            s=50 if name in report["safety_efficiency"]["frontier"] else 32,
            color=colors[name],
            edgecolor="black" if name in report["safety_efficiency"]["frontier"] else "white",
            linewidth=0.8,
        )
    frontier = report["safety_efficiency"]["frontier"][0]
    point = safety[frontier]
    axes[1].annotate(
        "Frontier",
        (point["routing_latency_ms"], point["verifiability"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=7,
        color="#b84a5a",
    )
    axes[1].set_xlabel("Routing latency ms/case (lower is better; NFE fixed)")
    axes[1].set_ylabel("Verifiability (higher is better)")
    axes[1].set_title("B  Safety-efficiency", loc="left", fontweight="bold")
    for axis in axes:
        axis.grid(True, color="#c7ced8", alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=3, width=0.8)
    short_labels = {
        "FloorFlow control-gated": "FF gated",
        "FloorFlow raw renderer": "FF raw",
        "typed_edit_graph": "Typed graph",
        "dense_cross_attention": "Dense attention",
        "no_edit_gating": "No gate",
        "no_structural_graph": "No graph",
        "Raster2Seq native": "R2Seq native",
        "Raster2Seq outside-frozen redraw": "R2Seq frozen",
    }
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5,
            color=colors[name],
            label=short_labels[name],
        )
        for name in quality
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


__all__ = ["main", "pareto"]


if __name__ == "__main__":
    main()
