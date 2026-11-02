#!/usr/bin/env python3
"""Convert a generated fpgen run into the floorflow raster cache layout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from floorflow.data.synthetic_data import convert_fpgen_gt
from floorflow.data.taxonomy import FloorplanTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2/taxonomy.json",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--test-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    paths = sorted(args.input.rglob("*.gt.json"))
    if not paths:
        raise FileNotFoundError(f"no fpgen GT files under {args.input}")
    splits = {"train": [], "val": [], "test": []}
    class_pixels = np.zeros(taxonomy.num_classes, dtype=np.int64)
    for index, path in enumerate(paths):
        if index % max(1, args.val_every) == 0:
            split = "val"
        elif index % max(1, args.test_every) == 1:
            split = "test"
        else:
            split = "train"
        target, footprint = convert_fpgen_gt(
            path,
            taxonomy,
            image_size=args.image_size,
        )
        relative = Path(split) / f"{index:06d}.npz"
        output = args.output_cache / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, target=target, footprint=footprint)
        splits[split].append(relative.as_posix())
        class_pixels += np.bincount(target.ravel(), minlength=taxonomy.num_classes)
    args.output_cache.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.taxonomy, args.output_cache / "taxonomy.json")
    manifest = {
        "source": str(args.input),
        "image_size": args.image_size,
        "splits": splits,
        "counts": {name: len(values) for name, values in splits.items()},
        "class_pixels": class_pixels.tolist(),
        "class_names": list(taxonomy.class_names),
    }
    (args.output_cache / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
