#!/usr/bin/env python3
"""Create name-preserving Raster2Seq inputs from FloorFlow samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

from floorflow.data.dataset import CubiCasaFloorplanDataset, load_or_rasterize
from floorflow.data.raster_control import (
    footprint_center,
    load_cubicasa_floor_rgb,
    raster_control_image,
)
from floorflow.data.svg import rasterize_cubicasa_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument(
        "--input-mode",
        choices=("control", "page-floor", "source-whitened"),
        default="control",
        help=(
            "Render aligned FloorFlow control, crop the target-free page-level "
            "floor drawing, or whiten the native image in SVG coordinates."
        ),
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        limit=None,
        write_cache=False,
    )
    selected = dataset.samples[
        int(args.offset) : int(args.offset) + int(args.limit)
    ]
    args.image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sample in enumerate(selected):
        sample_key = f"{sample.relative_case}__{sample.floor_id}"
        safe_name = sample_key.replace("/", "__")
        source = Path(sample.raster_path)
        native_image = Image.open(source).convert("RGB")
        native = np.asarray(native_image, dtype=np.uint8)
        if args.input_mode == "control":
            target, footprint = load_or_rasterize(
                sample,
                image_size=256,
                taxonomy=dataset.taxonomy,
                write_cache=False,
            )
            control = raster_control_image(
                target,
                footprint,
                dataset.taxonomy,
            )
            intensity = np.minimum(control[0], control[1])
            image = np.rint(intensity * 255.0).clip(0, 255).astype(np.uint8)
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif args.input_mode == "page-floor":
            target, footprint = load_or_rasterize(
                sample,
                image_size=256,
                taxonomy=dataset.taxonomy,
                write_cache=False,
            )
            floor_ids = sorted(
                {
                    str(item.floor_id)
                    for item in dataset.samples
                    if item.relative_case == sample.relative_case
                }
            )
            image = load_cubicasa_floor_rgb(
                source,
                floor_index=floor_ids.index(str(sample.floor_id)),
                floor_count=len(floor_ids),
                target_center=footprint_center(footprint),
            )
        else:
            target, _footprint = rasterize_cubicasa_svg(
                sample.svg_path,
                image_size=256,
                output_shape=native.shape[:2],
                floor_id=sample.floor_id,
                taxonomy=dataset.taxonomy,
            )
            occupancy = binary_fill_holes(target > 0)
            image = native.copy()
            image[~occupancy] = 255
        prepared = args.image_dir / f"{safe_name}.png"
        if prepared.exists():
            prepared.unlink()
        Image.fromarray(image, mode="RGB").save(prepared)
        rows.append(
            {
                "safe_name": safe_name,
                "sample_key": sample_key,
                "split": args.split,
                "split_index": int(args.offset) + index,
                "source_image": str(source),
                "source_size": [int(native.shape[1]), int(native.shape[0])],
                "input_mode": args.input_mode,
                "source_image_sha256": _sha256(source),
                "prepared_image_sha256": _sha256(prepared),
            }
        )
    payload = {
        "schema_version": "raster2seq-floorflow-input-map-v2",
        "input_mode": args.input_mode,
        "split": args.split,
        "offset": int(args.offset),
        "limit": int(args.limit),
        "rows": rows,
    }
    args.mapping.parent.mkdir(parents=True, exist_ok=True)
    args.mapping.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(rows), "mapping": str(args.mapping)}))


if __name__ == "__main__":
    main()

