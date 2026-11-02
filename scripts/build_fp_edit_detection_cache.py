#!/usr/bin/env python3
"""Build seed-independent changed masks and post-edit controls for delta metrics."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from floorflow.data.control_cache import control_cache_path
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.edit_detection import decode_visible_controls

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
        "--output-root",
        type=Path,
        default=Path("outputs/cache/fp_edit_detection_v1"),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("outputs/control_contracts"),
    )
    parser.add_argument(
        "--raster-root",
        type=Path,
        default=Path("outputs/eval/fp_hard_metrics_full_test503_seed19_v1/rasters"),
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"),
    )
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def build_sample(
    args: tuple[str, str, str, str, str, str, str],
) -> str:
    torch.set_num_threads(1)
    (
        operation,
        name,
        source_root,
        edit_root,
        raster_root,
        output_root,
        taxonomy_path,
    ) = args
    raster_path = Path(raster_root) / f"{name.replace('/', '__')}.npz"
    with np.load(raster_path) as payload:
        footprint = payload["footprint"].astype(bool)
    taxonomy = FloorplanTaxonomy.read(Path(taxonomy_path))
    before = decode_visible_controls(
        control_cache_path(source_root, "test", name),
        footprint,
        taxonomy,
    )
    after = decode_visible_controls(
        control_cache_path(edit_root, "test", name),
        footprint,
        taxonomy,
    )
    output = (
        Path(output_root)
        / operation
        / f"{name.replace('/', '__')}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        changed=before != after,
        after_visible=after.astype(np.uint8),
    )
    return str(output)


def main() -> None:
    args = parse_args()
    jobs = []
    expected = 0
    for operation, edit_directory in EDITS.items():
        manifest_path = (
            args.control_root / edit_directory / "test" / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["rows"]:
            if not record["edited"]:
                continue
            expected += 1
            jobs.append(
                (
                    operation,
                    record["name"],
                    str(args.control_root / "oracle_v1"),
                    str(args.control_root / edit_directory),
                    str(args.raster_root),
                    str(args.output_root),
                    str(args.taxonomy),
                )
            )
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for _path in executor.map(build_sample, jobs):
            completed += 1
            if completed % 100 == 0:
                print(f"cached {completed}/{expected}", flush=True)
    print(json.dumps({"completed": completed, "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
