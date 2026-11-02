#!/usr/bin/env python3
"""Evaluate a FloorFlow local edit over a Raster2Seq fidelity backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.edit_detection import (
    decode_visible_controls,
    edit_detection_iou_mse,
)
from floorflow.inference.local_delta import compose_typed_delta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_fp_raster2seq_local_redraw import EDITS, region_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--control-root",
        type=Path,
        default=ROOT / "outputs/control_contracts",
    )
    parser.add_argument(
        "--before-root",
        type=Path,
        default=ROOT / "outputs/eval/fp_hard_metrics_full_test503_seed19_v1/rasters",
    )
    parser.add_argument(
        "--after-root",
        type=Path,
        default=ROOT / "outputs/eval/fp_control_gated_local_test503_seed19_v1",
    )
    parser.add_argument(
        "--raster2seq-root",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/test503_matched_v1/predictions",
    )
    parser.add_argument(
        "--before-control-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/oracle_v1/test",
    )
    parser.add_argument(
        "--typed-control-cache-root",
        type=Path,
        default=ROOT / "outputs/cache/typed_edit_controls_v1",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/floorflow_raster2seq_hybrid_seed19_v1.json",
    )
    return parser.parse_args()


def room_support_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> float:
    predicted = np.isin(prediction, taxonomy.room_ids)
    expected = np.isin(target, taxonomy.room_ids)
    union = np.logical_or(predicted, expected).sum()
    return (
        1.0
        if union == 0
        else float(np.logical_and(predicted, expected).sum() / union)
    )


def typed_controls(
    before_path: Path,
    after_path: Path,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Render typed controls once and reuse them across evaluation seeds."""

    if cache_path.exists():
        with np.load(cache_path) as payload:
            return payload["before"].astype(np.uint8), payload["after"].astype(
                np.uint8
            )
    before = decode_visible_controls(before_path, footprint, taxonomy)
    after = decode_visible_controls(after_path, footprint, taxonomy)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        before=before.astype(np.uint8),
        after=after.astype(np.uint8),
    )
    return before, after


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    operations = {}
    for operation, directory in EDITS.items():
        manifest = json.loads(
            (
                args.control_root / directory / "test" / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        rows = []
        for record in manifest["rows"]:
            if not record["edited"]:
                continue
            name = str(record["name"])
            safe_name = name.replace("/", "__")
            with np.load(args.before_root / f"{safe_name}.npz") as payload:
                target_before = payload["target"].astype(np.uint8, copy=True)
                footprint = payload["footprint"].astype(bool)
                floorflow_before = payload["prediction"].astype(np.uint8)
            with np.load(
                args.after_root / operation / f"{safe_name}.npz"
            ) as payload:
                floorflow_after = payload["prediction"].astype(np.uint8)
            with np.load(
                args.raster2seq_root / operation / f"{safe_name}.npz"
            ) as payload:
                r2s_before = payload["prediction_before"].astype(np.uint8)
                r2s_after = payload["prediction_after"].astype(np.uint8)
            with np.load(
                ROOT / "outputs/cache/fp_edit_detection_v1" / operation
                / f"{safe_name}.npz"
            ) as payload:
                changed = payload["changed"].astype(bool)
                after_visible = payload["after_visible"].astype(np.uint8)
            before_control, after_control = typed_controls(
                args.before_control_root / f"{safe_name}.npz",
                args.control_root / directory / "test" / f"{safe_name}.npz",
                footprint,
                taxonomy,
                args.typed_control_cache_root / operation / f"{safe_name}.npz",
            )
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
            outside = valid & ~edit_region
            if operation == "opening_delete":
                relevant_ids = (*taxonomy.opening_ids, taxonomy.class_id("wall"))
            elif operation == "furniture_delete":
                relevant_ids = (*taxonomy.furniture_ids, taxonomy.class_id("wall"))
            else:
                relevant_ids = (*taxonomy.room_ids, taxonomy.class_id("wall"))
            relevant_delta = (
                np.isin(floorflow_before, relevant_ids)
                | np.isin(floorflow_after, relevant_ids)
            )
            typed_before, typed_after = compose_typed_delta(
                r2s_before,
                before_control,
                after_control,
                edit_region,
            )
            floorflow_typed_before, floorflow_typed_after = compose_typed_delta(
                floorflow_before,
                before_control,
                after_control,
                edit_region,
            )
            candidates = {
                "floorflow": (floorflow_before, floorflow_after),
                "raster2seq_local": (r2s_before, r2s_after),
                "hybrid": (
                    r2s_before,
                    np.where(edit_region, floorflow_after, r2s_before),
                ),
                "hybrid_delta": (
                    r2s_before,
                    np.where(
                        edit_region
                        & (floorflow_after != floorflow_before),
                        floorflow_after,
                        r2s_before,
                    ),
                ),
                "hybrid_context": (
                    r2s_before,
                    np.where(
                        edit_region
                        & (floorflow_after != floorflow_before),
                        floorflow_after,
                        np.where(edit_region, r2s_after, r2s_before),
                    ),
                ),
                "hybrid_filtered": (
                    r2s_before,
                    np.where(
                        edit_region
                        & (floorflow_after != floorflow_before)
                        & relevant_delta,
                        floorflow_after,
                        r2s_before,
                    ),
                ),
                "typed_splice": (typed_before, typed_after),
                "typed_floorflow": (
                    floorflow_typed_before,
                    floorflow_typed_after,
                ),
                "typed_before_floorflow_after": (
                    floorflow_typed_before,
                    floorflow_after,
                ),
            }
            metrics = {}
            for method, (before, after) in candidates.items():
                delta_iou, delta_mse = edit_detection_iou_mse(
                    target_before,
                    target_after,
                    before,
                    after,
                    valid,
                )
                metrics[method] = {
                    "delta_iou": delta_iou,
                    "delta_mse": delta_mse,
                    "outside_exact_ratio": (
                        float((after[outside] == before[outside]).mean())
                        if outside.any()
                        else 1.0
                    ),
                    "outside_changed_pixels": int(
                        np.count_nonzero(after[outside] != before[outside])
                    ),
                    "before_room_support_iou": room_support_iou(
                        before,
                        target_before,
                        taxonomy,
                    ),
                    "after_room_support_iou": room_support_iou(
                        after,
                        target_after,
                        taxonomy,
                    ),
                }
            rows.append({"name": name, **metrics})
        operations[operation] = {
            "cases": len(rows),
            "methods": {
                method: {
                    metric: float(
                        np.mean([row[method][metric] for row in rows])
                    )
                    for metric in (
                        "delta_iou",
                        "delta_mse",
                        "outside_exact_ratio",
                        "outside_changed_pixels",
                        "before_room_support_iou",
                        "after_room_support_iou",
                    )
                }
                for method in (
                    "floorflow",
                    "raster2seq_local",
                    "hybrid",
                    "hybrid_delta",
                    "hybrid_context",
                    "hybrid_filtered",
                    "typed_splice",
                    "typed_floorflow",
                    "typed_before_floorflow_after",
                )
            },
            "rows": rows,
        }
    report = {
        "status": "complete",
        "scope": "CubiCasa test503 strict-locality edit rows",
        "seed": args.seed,
        "description": (
            "Hybrid output uses Raster2Seq before-edit reconstruction outside "
            "the declared edit region and FloorFlow's control-gated output only "
            "inside that region."
        ),
        "operations": operations,
        "checks": {
            "hybrid_outside_exact_one": all(
                value["methods"]["hybrid"]["outside_exact_ratio"] == 1.0
                for value in operations.values()
            ),
            "hybrid_region_iou_above_0_608": all(
                value["methods"]["hybrid"]["delta_iou"] > 0.608
                for value in operations.values()
            ),
            "hybrid_delta_outside_exact_one": all(
                value["methods"]["hybrid_delta"]["outside_exact_ratio"] == 1.0
                for value in operations.values()
            ),
            "hybrid_delta_iou_above_0_608": all(
                value["methods"]["hybrid_delta"]["delta_iou"] > 0.608
                for value in operations.values()
            ),
            "hybrid_context_outside_exact_one": all(
                value["methods"]["hybrid_context"]["outside_exact_ratio"]
                == 1.0
                for value in operations.values()
            ),
            "hybrid_context_iou_above_0_608": all(
                value["methods"]["hybrid_context"]["delta_iou"] > 0.608
                for value in operations.values()
            ),
            "hybrid_filtered_outside_exact_one": all(
                value["methods"]["hybrid_filtered"]["outside_exact_ratio"]
                == 1.0
                for value in operations.values()
            ),
            "hybrid_filtered_iou_above_0_608": all(
                value["methods"]["hybrid_filtered"]["delta_iou"] > 0.608
                for value in operations.values()
            ),
            "typed_splice_outside_exact_one": all(
                value["methods"]["typed_splice"]["outside_exact_ratio"] == 1.0
                for value in operations.values()
            ),
            "typed_splice_iou_above_0_608": all(
                value["methods"]["typed_splice"]["delta_iou"] > 0.608
                for value in operations.values()
            ),
            "typed_floorflow_outside_exact_one": all(
                value["methods"]["typed_floorflow"]["outside_exact_ratio"] == 1.0
                for value in operations.values()
            ),
            "typed_floorflow_iou_above_0_608": all(
                value["methods"]["typed_floorflow"]["delta_iou"] > 0.608
                for value in operations.values()
            ),
            "typed_before_floorflow_after_outside_exact_one": all(
                value["methods"]["typed_before_floorflow_after"][
                    "outside_exact_ratio"
                ]
                == 1.0
                for value in operations.values()
            ),
            "typed_before_floorflow_after_iou_above_0_608": all(
                value["methods"]["typed_before_floorflow_after"]["delta_iou"]
                > 0.608
                for value in operations.values()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], indent=2))
    raise SystemExit(0 if all(report["checks"].values()) else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
