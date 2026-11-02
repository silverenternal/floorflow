#!/usr/bin/env python3
"""Evaluate an edit-aware local-redraw wrapper around Raster2Seq."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.edit_detection import edit_detection_iou_mse
from floorflow.inference.local_delta import compose_local_delta

EDITS = {
    "room_type": "fp_local_edit_room_type_test503_v1",
    "room_add": "fp_local_edit_room_add_test503_v1",
    "room_remove": "fp_local_edit_room_remove_test503_v1",
    "opening_delete": "fp_local_edit_opening_delete_test503_v1",
    "furniture_delete": "fp_local_edit_furniture_delete_test503_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "outputs/external_baselines/raster2seq/"
            "test503_matched_v1/predictions"
        ),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("outputs/control_contracts"),
    )
    parser.add_argument(
        "--raster-root",
        type=Path,
        default=Path(
            "outputs/eval/fp_hard_metrics_full_test503_seed19_v1/rasters"
        ),
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/reports/fp_raster2seq_local_redraw_v1.json"
        ),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path(
            "paper/tables/fp_control_gated_vs_raster2seq.md"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def region_mask(shape: tuple[int, int], normalized: list[float]) -> np.ndarray:
    height, width = shape
    x, y, box_width, box_height = normalized
    left = max(0, min(width, round(float(x) * width)))
    top = max(0, min(height, round(float(y) * height)))
    right = max(left + 1, min(width, round((x + box_width) * width)))
    bottom = max(top + 1, min(height, round((y + box_height) * height)))
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    results = {}
    for operation, directory in EDITS.items():
        manifest = read_json(
            args.control_root / directory / "test" / "manifest.json"
        )
        rows = []
        for record in manifest["rows"]:
            if not record["edited"]:
                continue
            name = record["name"]
            safe_name = name.replace("/", "__")
            with np.load(
                args.raster_root / f"{safe_name}.npz"
            ) as payload:
                target_before = payload["target"].astype(np.uint8, copy=True)
                footprint = payload["footprint"].astype(bool)
            with np.load(
                args.predictions
                / operation
                / f"{safe_name}.npz"
            ) as payload:
                external_before = payload["prediction_before"]
                external_after = payload["prediction_after"]
            with np.load(
                Path("outputs/cache/fp_edit_detection_v1")
                / operation
                / f"{safe_name}.npz"
            ) as payload:
                changed = payload["changed"].astype(bool)
                after_visible = payload["after_visible"].astype(np.uint8)
            target_after = target_before.copy()
            target_after[changed] = after_visible[changed]
            edit_region = region_mask(
                target_before.shape,
                record["edit_region_normalized"],
            ) & footprint
            wall = (
                (target_before == taxonomy.class_id("wall"))
                | (target_after == taxonomy.class_id("wall"))
            )
            valid = footprint & ~binary_dilation(wall, iterations=1)
            local = compose_local_delta(
                torch.from_numpy(external_before),
                torch.from_numpy(external_after),
                torch.from_numpy(edit_region),
            ).numpy()
            delta_iou, delta_mse = edit_detection_iou_mse(
                target_before,
                target_after,
                external_before,
                local,
                valid,
            )
            outside = valid & ~edit_region
            rows.append(
                {
                    "name": name,
                    "delta_iou": delta_iou,
                    "delta_mse": delta_mse,
                    "outside_exact_ratio": (
                        float(
                            (
                                local[outside]
                                == external_before[outside]
                            ).mean()
                        )
                        if outside.any()
                        else 1.0
                    ),
                }
            )
        results[operation] = {
            "cases": len(rows),
            "delta_iou": float(np.mean([row["delta_iou"] for row in rows])),
            "delta_mse": float(np.mean([row["delta_mse"] for row in rows])),
            "outside_exact_ratio": float(
                np.mean([row["outside_exact_ratio"] for row in rows])
            ),
            "rows": rows,
        }

    report = {
        "status": "complete",
        "method": "Raster2Seq local redraw",
        "description": (
            "The before prediction is frozen outside the declared edit "
            "region; only Raster2Seq's post-edit rendering changes inside it."
        ),
        "operations": results,
        "aggregate": {
            "delta_iou": float(
                np.mean(
                    [
                        row["delta_iou"]
                        for result in results.values()
                        for row in result["rows"]
                    ]
                )
            ),
            "delta_mse": float(
                np.mean(
                    [
                        row["delta_mse"]
                        for result in results.values()
                        for row in result["rows"]
                    ]
                )
            ),
            "outside_exact_ratio": float(
                np.mean(
                    [
                        row["outside_exact_ratio"]
                        for result in results.values()
                        for row in result["rows"]
                    ]
                )
            ),
        },
        "comparability": {
            "same_rows": True,
            "same_edit_region": True,
            "same_outside_freeze": True,
            "editing_aware_wrapper": True,
            "external_native_edit_model": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    floorflow = read_json(
        Path("outputs/reports/fp_edit_detection_test503_controlgated_fiveseed_v1.json")
    )
    table = [
        "# Control-Gated FloorFlow vs Raster2Seq Local Redraw",
        "",
        (
            "| Operation | Cases | FloorFlow Delta-IoU | Raster2Seq Delta-IoU | "
            "FloorFlow Delta-MSE | Raster2Seq Delta-MSE |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for operation, result in results.items():
        ours = floorflow["operations"][operation]
        table.append(
            f"| {operation} | {result['cases']} | "
            f"{np.mean(ours['local_delta_iou']):.4f} | "
            f"{result['delta_iou']:.4f} | "
            f"{np.mean(ours['local_delta_mse']):.6f} | "
            f"{result['delta_mse']:.6f} |"
        )
    table.extend(
        [
            "",
            (
                "Raster2Seq local redraw uses the same edit region and freezes "
                "outside predictions. It is an editing-aware wrapper around a "
                "reconstruction backbone, not a native editing model."
            ),
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
