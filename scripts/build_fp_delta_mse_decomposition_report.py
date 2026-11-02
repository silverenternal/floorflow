#!/usr/bin/env python3
"""Decompose FloorFlow local-delta MSE into structural and interior regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.delta_mse_decomposition import (
    average_decompositions,
    decompose_delta_error,
)
from floorflow.inference.local_delta import compose_local_delta

EDITS = {
    "room_type": "fp_local_edit_room_type_test503",
    "room_add": "fp_local_edit_room_add_test503",
    "room_remove": "fp_local_edit_room_remove_test503",
    "opening_delete": "fp_local_edit_opening_delete_test503",
    "furniture_delete": "fp_local_edit_furniture_delete_test503",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--variant",
        choices=("local_delta", "control_gated"),
        default="local_delta",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"),
    )
    parser.add_argument(
        "--edit-cache",
        type=Path,
        default=Path("outputs/cache/fp_edit_detection_v1"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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


def raster(root: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(root / "rasters" / f"{name.replace('/', '__')}.npz") as payload:
        return {key: payload[key] for key in payload.files}


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    variant = "_v1" if args.seed == 19 else f"_seed{args.seed}_v1"
    original_root = Path(
        f"outputs/eval/fp_hard_metrics_full_test503_seed{args.seed}_v1"
    )
    operation_results = {}
    for operation, directory in EDITS.items():
        edit_root = Path("outputs/eval") / f"{directory}{variant}"
        control_root = Path("outputs/control_contracts") / f"{directory}_v1"
        manifest = json.loads(
            (control_root / "test" / "manifest.json").read_text(encoding="utf-8")
        )
        decompositions = []
        for record in manifest["rows"]:
            if not record["edited"]:
                continue
            name = record["name"]
            before = raster(original_root, name)
            after = raster(edit_root, name)
            footprint = before["footprint"].astype(bool)
            cache_path = (
                args.edit_cache / operation / f"{name.replace('/', '__')}.npz"
            )
            with np.load(cache_path) as payload:
                changed = payload["changed"].astype(bool)
                visible_after = payload["after_visible"].astype(np.uint8)
            target_before = before["target"].astype(np.uint8, copy=True)
            target_after = target_before.copy()
            target_after[changed] = visible_after[changed]
            edit_region = region_mask(
                target_before.shape,
                record["edit_region_normalized"],
            ) & footprint
            wall = (
                (target_before == taxonomy.class_id("wall"))
                | (target_after == taxonomy.class_id("wall"))
            )
            wall = binary_dilation(wall, iterations=1)
            valid = footprint & ~wall
            local_mask = edit_region
            if args.variant == "control_gated":
                local_mask = edit_region & changed
            local_after = compose_local_delta(
                torch.from_numpy(before["prediction"]),
                torch.from_numpy(after["prediction"]),
                torch.from_numpy(local_mask),
            ).numpy()
            decomposition = decompose_delta_error(
                target_before,
                target_after,
                before["prediction"],
                local_after,
                valid,
                edit_region,
                taxonomy,
            )
            decompositions.append(decomposition)
        operation_results[operation] = {
            "cases": len(decompositions),
            "summary": average_decompositions(decompositions),
            "rows": decompositions,
        }
        print(f"decomposed {operation}: {len(decompositions)} cases", flush=True)

    all_rows = [
        decomposition
        for operation in operation_results.values()
        for decomposition in operation["rows"]
    ]
    report = {
        "status": "complete",
        "seed": args.seed,
        "scope": "CubiCasa test503 local-delta edits",
        "variant": args.variant,
        "regional_mse": (
            "Mean squared difference between ground-truth and predicted edit "
            "change maps, computed independently within each region."
        ),
        "boundary_definition": (
            "Categorical-label boundaries involving walls or openings are "
            "dilated by two pixels. Remaining room-to-room boundaries form the "
            "room-boundary band. The edit interior excludes both bands."
        ),
        "regions_are_exclusive_within_edit_region": True,
        "operations": operation_results,
        "aggregate": average_decompositions(all_rows),
        "cases": len(all_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
