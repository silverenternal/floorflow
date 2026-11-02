#!/usr/bin/env python3
"""Measure HouseMind-style edit-detection IoU/MSE on paired FP edits."""

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
    "room_type": (
        "fp_local_edit_room_type_test503{variant}",
        "fp_local_edit_room_type_test503_v1",
    ),
    "room_add": (
        "fp_local_edit_room_add_test503{variant}",
        "fp_local_edit_room_add_test503_v1",
    ),
    "room_remove": (
        "fp_local_edit_room_remove_test503{variant}",
        "fp_local_edit_room_remove_test503_v1",
    ),
    "opening_delete": (
        "fp_local_edit_opening_delete_test503{variant}",
        "fp_local_edit_opening_delete_test503_v1",
    ),
    "furniture_delete": (
        "fp_local_edit_furniture_delete_test503{variant}",
        "fp_local_edit_furniture_delete_test503_v1",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".data/cubicasa5k"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--edit-detection-cache",
        type=Path,
        default=Path("outputs/cache/fp_edit_detection_v1"),
    )
    parser.add_argument(
        "--variant",
        choices=("local_delta", "control_gated", "room_control_gated"),
        default="local_delta",
    )
    parser.add_argument("--raster-output-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def region_mask(
    shape: tuple[int, int],
    normalized: list[float],
) -> np.ndarray:
    height, width = shape
    x, y, box_width, box_height = normalized
    left = max(0, min(width, round(float(x) * width)))
    top = max(0, min(height, round(float(y) * height)))
    right = max(left + 1, min(width, round((x + box_width) * width)))
    bottom = max(top + 1, min(height, round((y + box_height) * height)))
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def raster(root: Path, name: str) -> dict[str, np.ndarray]:
    path = root / "rasters" / f"{name.replace('/', '__')}.npz"
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def main() -> None:
    args = parse_args()
    variant = "_v1" if args.seed == 19 else f"_seed{args.seed}_v1"
    original_root = Path(
        f"outputs/eval/fp_hard_metrics_full_test503_seed{args.seed}_v1"
    )
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    results = {}
    for operation, (edit_directory, manifest_directory) in EDITS.items():
        edit_root = Path("outputs/eval") / edit_directory.format(variant=variant)
        control_root = Path("outputs/control_contracts") / edit_directory.format(
            variant="_v1"
        )
        manifest_path = control_root / "test" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = []
        for record in manifest["rows"]:
            if not record["edited"]:
                continue
            name = record["name"]
            before = raster(original_root, name)
            after = raster(edit_root, name)
            footprint = before["footprint"].astype(bool)
            edit_cache = (
                args.edit_detection_cache
                / operation
                / f"{name.replace('/', '__')}.npz"
            )
            with np.load(edit_cache) as payload:
                control_changed = payload["changed"].astype(bool)
                after_control_visible = payload["after_visible"]
            before_visible = before["prediction"]
            after_visible = after["prediction"]
            before_topology = before.get("topology_prediction")
            after_topology = after.get("topology_prediction")
            before_room_labels = before["room_labels"]
            after_room_labels = after["room_labels"]
            target = before["target"].copy()
            target_after = target.copy()
            target_after[control_changed] = after_control_visible[control_changed]
            edit_region = region_mask(
                target.shape,
                record["edit_region_normalized"],
            ) & footprint
            wall = (
                (target == taxonomy.class_id("wall"))
                | (target_after == taxonomy.class_id("wall"))
            )
            wall = binary_dilation(wall, iterations=1)
            valid = footprint & ~wall
            raw_iou, raw_mse = edit_detection_iou_mse(
                target,
                target_after,
                before_visible,
                after_visible,
                valid,
            )
            composed_visible = compose_local_delta(
                torch.from_numpy(before_visible),
                torch.from_numpy(after_visible),
                torch.from_numpy(edit_region),
            ).numpy()
            composed_room_labels = compose_local_delta(
                torch.from_numpy(before_room_labels),
                torch.from_numpy(after_room_labels),
                torch.from_numpy(edit_region),
            ).numpy()
            local_iou, local_mse = edit_detection_iou_mse(
                target,
                target_after,
                before_visible,
                composed_visible,
                valid,
            )
            local_mask = edit_region
            topology_mask = edit_region
            if args.variant == "control_gated":
                local_mask = edit_region & control_changed
                topology_mask = local_mask
                composed_visible = compose_local_delta(
                    torch.from_numpy(before_visible),
                    torch.from_numpy(after_visible),
                    torch.from_numpy(local_mask),
                ).numpy()
                composed_room_labels = compose_local_delta(
                    torch.from_numpy(before_room_labels),
                    torch.from_numpy(after_room_labels),
                    torch.from_numpy(local_mask),
                ).numpy()
                local_iou, local_mse = edit_detection_iou_mse(
                    target,
                    target_after,
                    before_visible,
                    composed_visible,
                    valid,
                )
            elif args.variant == "room_control_gated":
                room_ids = np.asarray(taxonomy.room_ids)
                room_mask = np.isin(before_visible, room_ids) | np.isin(
                    after_visible,
                    room_ids,
                )
                local_mask = edit_region & ~control_changed & room_mask
                topology_mask = np.zeros_like(edit_region)
                composed_visible[local_mask] = before_visible[local_mask]
                composed_room_labels[local_mask] = before_room_labels[local_mask]
                local_iou, local_mse = edit_detection_iou_mse(
                    target,
                    target_after,
                    before_visible,
                    composed_visible,
                    valid,
                )
            outside = valid & ~edit_region
            visible_equal = composed_visible[outside] == before_visible[outside]
            room_equal = (
                composed_room_labels[outside] == before_room_labels[outside]
            )
            rows.append(
                {
                    "name": name,
                    "raw_delta_iou": raw_iou,
                    "raw_delta_mse": raw_mse,
                    "local_delta_iou": local_iou,
                    "local_delta_mse": local_mse,
                    "outside_prediction_changes": int(
                        (~visible_equal).sum()
                    ),
                    "outside_room_changes": int(
                        (~room_equal).sum()
                    ),
                    "outside_valid_pixels": int(outside.sum()),
                    "outside_exact_ratio": (
                        float(visible_equal.mean())
                        if visible_equal.size
                        else 1.0
                    ),
                    "outside_room_exact_ratio": (
                        float(room_equal.mean()) if room_equal.size else 1.0
                    ),
                }
            )
            if args.raster_output_root is not None:
                output_dir = args.raster_output_root / operation
                output_dir.mkdir(parents=True, exist_ok=True)
                variant_raster = dict(before)
                variant_raster["prediction"] = composed_visible
                variant_raster["room_labels"] = composed_room_labels
                variant_raster["target"] = target_after
                if before_topology is not None and after_topology is not None:
                    variant_raster["topology_prediction"] = compose_local_delta(
                        torch.from_numpy(before_topology),
                        torch.from_numpy(after_topology),
                        torch.from_numpy(topology_mask),
                    ).numpy()
                np.savez_compressed(
                    output_dir / f"{name.replace('/', '__')}.npz",
                    **variant_raster,
                )
        summary = {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "median": float(np.median([row[key] for row in rows])),
                "p10": float(np.quantile([row[key] for row in rows], 0.1)),
            }
            for key in (
                "raw_delta_iou",
                "raw_delta_mse",
                "local_delta_iou",
                "local_delta_mse",
            )
        }
        results[operation] = {
            "cases": len(rows),
            "summary": summary,
            "outside_prediction_changes": int(
                sum(row["outside_prediction_changes"] for row in rows)
            ),
            "outside_room_changes": int(
                sum(row["outside_room_changes"] for row in rows)
            ),
            "outside_valid_pixels": int(
                sum(row["outside_valid_pixels"] for row in rows)
            ),
            "min_outside_exact_ratio": float(
                min(row["outside_exact_ratio"] for row in rows)
            ),
            "min_outside_room_exact_ratio": float(
                min(row["outside_room_exact_ratio"] for row in rows)
            ),
            "rows": rows,
        }
        print(f"evaluated {operation}: {len(rows)} cases", flush=True)
    operations = results.values()
    aggregate = {
        "raw_delta_iou": float(
            np.mean([row["raw_delta_iou"] for op in operations for row in op["rows"]])
        ),
        "raw_delta_mse": float(
            np.mean([row["raw_delta_mse"] for op in operations for row in op["rows"]])
        ),
        "local_delta_iou": float(
            np.mean([row["local_delta_iou"] for op in operations for row in op["rows"]])
        ),
        "local_delta_mse": float(
            np.mean([row["local_delta_mse"] for op in operations for row in op["rows"]])
        ),
    }
    anchor_iou = 0.608
    anchor_mse = 0.0019
    report = {
        "status": "complete",
        "seed": args.seed,
        "variant": args.variant,
        "local_mask_definition": (
            "declared edit rectangle"
            if args.variant == "local_delta"
            else (
                "declared edit rectangle intersected with control-change mask"
                if args.variant == "control_gated"
                else (
                    "declared edit rectangle, with non-changed room-label "
                    "pixels restored from the cached pre-edit prediction"
                )
            )
        ),
        "scope": "CubiCasa test503 synthetic local edits",
        "metric_protocol": "HouseMind B.3 edit-detection Delta IoU and Delta MSE",
        "external_anchor": {
            "method": "HouseMind-E",
            "delta_iou": anchor_iou,
            "delta_mse": anchor_mse,
            "strict_same_benchmark": False,
            "note": "Published anchor is from HouseMind's RPLAN edit benchmark; this report is a protocol-level comparison on CubiCasa edits.",
        },
        "operations": results,
        "aggregate": aggregate,
        "delta_iou_anchor_passed": aggregate["local_delta_iou"] > anchor_iou,
        "delta_mse_anchor_passed": aggregate["local_delta_mse"] < anchor_mse,
        "dual_metric_anchor_passed": (
            aggregate["local_delta_iou"] > anchor_iou
            and aggregate["local_delta_mse"] < anchor_mse
        ),
        "outside_measurement": (
            "Strict uint8 label equality with no tolerance on footprint "
            "pixels outside the declared edit rectangle after excluding all "
            "wall pixels and one-pixel dilation of pre/post wall unions."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "delta_iou_anchor_passed": report["delta_iou_anchor_passed"],
                "delta_mse_anchor_passed": report["delta_mse_anchor_passed"],
                "dual_metric_anchor_passed": report["dual_metric_anchor_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
