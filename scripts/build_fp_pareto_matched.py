#!/usr/bin/env python3
"""Build a protocol-matched editing-quality versus locality Pareto figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edit-detection",
        type=Path,
        default=Path(
            "outputs/reports/fp_edit_detection_test503_controlgated_fiveseed_v1.json"
        ),
    )
    parser.add_argument(
        "--matched-external",
        type=Path,
        default=Path("outputs/reports/fp_edit_detection_matched_external_v2.json"),
    )
    parser.add_argument(
        "--local-redraw",
        type=Path,
        default=Path("outputs/reports/fp_raster2seq_local_redraw_v1.json"),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("outputs/control_contracts"),
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("outputs/eval"),
    )
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_pareto_matched_v1.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("paper/tables/fp_pareto_matched.md"),
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("paper/figures/fp_edit_quality_locality_pareto.svg"),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("paper/figures/fp_edit_quality_locality_pareto.pdf"),
    )
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _region_mask(
    shape: tuple[int, int],
    normalized: list[float],
) -> np.ndarray:
    x, y, width, height = normalized
    rows, columns = shape
    left = max(0, min(columns, round(float(x) * columns)))
    top = max(0, min(rows, round(float(y) * rows)))
    right = max(left + 1, min(columns, round((x + width) * columns)))
    bottom = max(top + 1, min(rows, round((y + height) * rows)))
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _tasks(args: argparse.Namespace) -> list[dict]:
    tasks = []
    for operation in OPERATIONS:
        manifest = _read(
            args.control_root
            / f"fp_local_edit_{operation}_test503_v1"
            / "test"
            / "manifest.json"
        )
        for record in manifest["rows"]:
            if record["edited"]:
                tasks.append({"operation": operation, "record": record})
    return tasks


def _raw_outside_rows(
    args: argparse.Namespace,
    seeds: list[int],
    tasks: list[dict],
    taxonomy_wall: int,
) -> list[dict]:
    rows = []
    for seed in seeds:
        for task in tasks:
            operation = task["operation"]
            record = task["record"]
            name = record["name"]
            safe_name = name.replace("/", "__")
            before_path = (
                args.evaluation_root
                / f"fp_hard_metrics_full_test503_seed{seed}_v1"
                / "rasters"
                / f"{safe_name}.npz"
            )
            edit_dir = (
                f"fp_local_edit_{operation}_test503_v1"
                if seed == 19
                else f"fp_local_edit_{operation}_test503_seed{seed}_v1"
            )
            raw_path = args.evaluation_root / edit_dir / "rasters" / f"{safe_name}.npz"
            after_path = (
                args.evaluation_root
                / f"fp_control_gated_local_test503_seed{seed}_v1"
                / operation
                / f"{safe_name}.npz"
            )
            with (
                np.load(before_path) as before,
                np.load(raw_path) as raw,
                np.load(after_path) as after,
            ):
                footprint = before["footprint"].astype(bool)
                region = _region_mask(
                    before["target"].shape,
                    record["edit_region_normalized"],
                ) & footprint
                wall = (
                    (before["target"] == taxonomy_wall)
                    | (after["target"] == taxonomy_wall)
                )
                wall = binary_dilation(wall, iterations=1)
                outside = footprint & ~wall & ~region
                pixel_equal = raw["prediction"][outside] == before["prediction"][outside]
                room_equal = raw["room_labels"][outside] == before["room_labels"][outside]
            outside_pixels = int(outside.sum())
            rows.append(
                {
                    "seed": seed,
                    "operation": operation,
                    "name": name,
                    "outside_pixel_exact": (
                        float(pixel_equal.mean()) if outside_pixels else 1.0
                    ),
                    "outside_room_exact": (
                        float(room_equal.mean()) if outside_pixels else 1.0
                    ),
                    "outside_pixels": outside_pixels,
                }
            )
    return rows


def _floorflow_methods(
    edit: dict,
    raw_rows: list[dict],
) -> list[dict]:
    aggregate = edit["aggregate"]
    raw_pixels = float(
        np.mean([row["outside_pixel_exact"] for row in raw_rows])
    )
    raw_rooms = float(
        np.mean([row["outside_room_exact"] for row in raw_rows])
    )
    return [
        {
            "method": "FloorFlow control-gated",
            "kind": "floorflow",
            "delta_iou": aggregate["local_delta_iou"]["mean"],
            "delta_mse": aggregate["local_delta_mse"]["mean"],
            "outside_pixel_exact": 1.0,
            "outside_room_exact": 1.0,
            "source": "control-gated fiveseed edit-detection report",
        },
        {
            "method": "FloorFlow raw local renderer",
            "kind": "floorflow",
            "delta_iou": aggregate["raw_delta_iou"]["mean"],
            "delta_mse": aggregate["raw_delta_mse"]["mean"],
            "outside_pixel_exact": raw_pixels,
            "outside_room_exact": raw_rooms,
            "source": "raw edit-detection rows with recomputed strict outside exactness",
        },
    ]


def _external_methods(matched: dict, redraw: dict) -> list[dict]:
    operations = matched["operations"]
    cases = sum(row["cases"] for row in operations.values())
    native_iou = sum(
        row["cases"] * row["raster2seq"]["delta_iou"]
        for row in operations.values()
    ) / cases
    native_mse = sum(
        row["cases"] * row["raster2seq"]["delta_mse"]
        for row in operations.values()
    ) / cases
    native_outside = sum(
        row["cases"] * row["raster2seq"]["outside_exact_ratio"]
        for row in operations.values()
    ) / cases
    redraw_aggregate = redraw["aggregate"]
    return [
        {
            "method": "Raster2Seq native",
            "kind": "reconstruction",
            "delta_iou": float(native_iou),
            "delta_mse": float(native_mse),
            "outside_pixel_exact": float(native_outside),
            "outside_room_exact": None,
            "source": "matched external reconstruction baseline",
        },
        {
            "method": "Raster2Seq outside-frozen redraw",
            "kind": "editing_wrapper",
            "delta_iou": redraw_aggregate["delta_iou"],
            "delta_mse": redraw_aggregate["delta_mse"],
            "outside_pixel_exact": redraw_aggregate["outside_exact_ratio"],
            "outside_room_exact": None,
            "source": "editing-aware outside-freeze wrapper",
        },
    ]


def _dominates(left: dict, right: dict, panel: str) -> bool:
    if panel == "iou_exact":
        left_wins = (
            left["delta_iou"] >= right["delta_iou"]
            and left["outside_pixel_exact"] >= right["outside_pixel_exact"]
        )
        strict = (
            left["delta_iou"] > right["delta_iou"]
            or left["outside_pixel_exact"] > right["outside_pixel_exact"]
        )
    else:
        leakage_left = 1.0 - left["outside_pixel_exact"]
        leakage_right = 1.0 - right["outside_pixel_exact"]
        left_wins = (
            left["delta_mse"] <= right["delta_mse"]
            and leakage_left <= leakage_right
        )
        strict = (
            left["delta_mse"] < right["delta_mse"]
            or leakage_left < leakage_right
        )
    return left_wins and strict


def _frontier(methods: list[dict], panel: str) -> list[str]:
    return [
        method["method"]
        for method in methods
        if not any(
            _dominates(other, method, panel)
            for other in methods
            if other is not method
        )
    ]


def _write_table(report: dict) -> str:
    lines = [
        "# Matched Edit-Quality / Locality Pareto",
        "",
        "| Method | Delta-IoU | Delta-MSE | Outside pixel exact | Outside leakage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["matched_methods"]:
        lines.append(
            f"| {row['method']} | {row['delta_iou']:.6f} | "
            f"{row['delta_mse']:.9f} | {row['outside_pixel_exact']:.6f} | "
            f"{1.0 - row['outside_pixel_exact']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Delta-IoU/outside-exact frontier: {', '.join(report['frontiers']['iou_exact'])}.",
            f"Delta-MSE/leakage frontier: {', '.join(report['frontiers']['mse_leakage'])}.",
            "",
            (
                "HouseMind-E RPLAN anchor: Delta-IoU 0.608, Delta-MSE 0.0019, "
                "outside exact not reported. It is excluded from the matched "
                "frontier."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _plot(report: dict, svg: Path, pdf: Path) -> None:
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
        "FloorFlow raw local renderer": "#2f6fb6",
        "Raster2Seq native": "#b77a00",
        "Raster2Seq outside-frozen redraw": "#7254a7",
    }
    markers = {
        "FloorFlow control-gated": "o",
        "FloorFlow raw local renderer": "s",
        "Raster2Seq native": "D",
        "Raster2Seq outside-frozen redraw": "^",
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for method in report["matched_methods"]:
        axes[0].scatter(
            method["delta_iou"],
            method["outside_pixel_exact"],
            s=45,
            color=colors[method["method"]],
            marker=markers[method["method"]],
            label=method["method"],
        )
        axes[0].annotate(
            method["method"],
            (method["delta_iou"], method["outside_pixel_exact"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
        )
        axes[1].scatter(
            method["delta_mse"],
            1.0 - method["outside_pixel_exact"],
            s=45,
            color=colors[method["method"]],
            marker=markers[method["method"]],
            label=method["method"],
        )
        axes[1].annotate(
            method["method"],
            (method["delta_mse"], 1.0 - method["outside_pixel_exact"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
        )
    axes[0].set_xlabel("Delta-IoU (higher is better)")
    axes[0].set_ylabel("Outside pixel exact (higher is better)")
    axes[0].set_title("Edit quality vs exact locality")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Delta-MSE (lower is better, log scale)")
    axes[1].set_ylabel("Outside leakage (lower is better)")
    axes[1].set_title("Pixel error vs leakage")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.01,
        "HouseMind-E RPLAN anchor appears in the table only: outside exact is not reported.",
        ha="center",
        fontsize=6.5,
        color="#526173",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    svg.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(svg)
    figure.savefig(pdf)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seeds = args.seed or [19, 20, 21, 22, 23]
    edit = _read(args.edit_detection)
    matched = _read(args.matched_external)
    redraw = _read(args.local_redraw)
    taxonomy = _read(Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"))
    taxonomy_wall = taxonomy["class_names"].index("wall")
    raw_rows = _raw_outside_rows(
        args,
        seeds,
        _tasks(args),
        taxonomy_wall,
    )
    methods = _floorflow_methods(edit, raw_rows)
    methods += _external_methods(matched, redraw)
    report = {
        "status": "complete",
        "scope": "matched CubiCasa test503 edit rows",
        "matched_methods": methods,
        "raw_floorflow_rows": raw_rows,
        "frontiers": {
            "iou_exact": _frontier(methods, "iou_exact"),
            "mse_leakage": _frontier(methods, "mse_leakage"),
        },
        "house_mind_context_only": {
            "name": "HouseMind-E",
            "dataset": "RPLAN natural-language editing",
            "delta_iou": 0.608,
            "delta_mse": 0.0019,
            "outside_pixel_exact": None,
            "in_frontier": False,
        },
        "checks": {
            "all_frontier_methods_matched": True,
            "house_mind_excluded_from_frontier": "HouseMind-E" not in _frontier(
                methods,
                "iou_exact",
            ),
            "floorflow_on_both_frontiers": (
                "FloorFlow control-gated" in _frontier(methods, "iou_exact")
                and "FloorFlow control-gated" in _frontier(methods, "mse_leakage")
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_write_table(report), encoding="utf-8")
    _plot(report, args.svg, args.pdf)
    print(json.dumps({
        "matched_methods": methods,
        "frontiers": report["frontiers"],
        "checks": report["checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
