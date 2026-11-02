#!/usr/bin/env python3
"""Build the versioned polygon and occlusion cache for CubiCasa5K."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.data.dataset import FloorplanSample, collect_samples
from floorflow.data.geometry_cache import (
    build_floorplan_geometry,
    geometry_cache_path,
    load_geometry_cache,
    write_geometry_cache,
)
from floorflow.data.taxonomy import (
    FloorplanTaxonomy,
    build_taxonomy,
    scan_taxonomy_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_geometry_v1",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1/taxonomy.json",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/reports/geometry_cache_v1.json",
    )
    return parser.parse_args()


def _process(
    sample: FloorplanSample,
    split: str,
    cache_root: Path,
    image_size: int,
    taxonomy: FloorplanTaxonomy,
    overwrite: bool,
) -> dict:
    path = geometry_cache_path(
        cache_root,
        split,
        sample.relative_case,
        sample.floor_id,
    )
    if path.is_file() and not overwrite:
        loaded = load_geometry_cache(path)
        return {
            "case": str(sample.relative_case),
            "floor": sample.floor_id,
            "status": "reused",
            "rooms": int(len(loaded["room_offsets"]) - 1),
            "walls": int(len(loaded["wall_offsets"]) - 1),
            "openings": int(len(loaded["opening_offsets"]) - 1),
            "furniture": int(len(loaded["furniture_offsets"]) - 1),
        }
    try:
        geometry = build_floorplan_geometry(
            sample.svg_path,
            floor_id=sample.floor_id,
            image_size=image_size,
            taxonomy=taxonomy,
        )
        write_geometry_cache(geometry, path, overwrite=overwrite)
        return {
            "case": str(sample.relative_case),
            "floor": sample.floor_id,
            "status": "written",
            "rooms": int(len(geometry.rooms.offsets) - 1),
            "walls": int(len(geometry.walls.offsets) - 1),
            "openings": int(len(geometry.openings.offsets) - 1),
            "furniture": int(len(geometry.furniture.offsets) - 1),
            "room_support_pixels": int(geometry.room_support.sum()),
            "occlusion_pixels": int(geometry.occlusion_mask.sum()),
        }
    except Exception as error:
        return {
            "case": str(sample.relative_case),
            "floor": sample.floor_id,
            "status": "error",
            "error": repr(error),
        }
def main() -> None:
    args = parse_args()
    if args.taxonomy.is_file():
        taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    else:
        svg_paths = []
        for split in args.splits:
            for sample in collect_samples(args.data_root, split, floors=True):
                if sample.svg_path not in svg_paths:
                    svg_paths.append(sample.svg_path)
        taxonomy = build_taxonomy(scan_taxonomy_counts(svg_paths))
    args.cache_root.mkdir(parents=True, exist_ok=True)
    taxonomy.write(args.cache_root / "taxonomy.json")

    rows = []
    failures = []
    for split in args.splits:
        samples = collect_samples(
            args.data_root,
            split,
            cache_root=args.cache_root,
            limit=args.limit,
        )
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(
                    _process,
                    sample,
                    split,
                    args.cache_root,
                    args.image_size,
                    taxonomy,
                    args.overwrite,
                )
                for sample in samples
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                row["split"] = split
                rows.append(row)
                if row["status"] == "error":
                    failures.append(row)
                if completed % 100 == 0 or completed == len(samples):
                    print(f"[{split}] {completed}/{len(samples)}", flush=True)

    report = {
        "cache_version": "geometry-v1",
        "data_root": str(args.data_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "image_size": args.image_size,
        "taxonomy": str(args.taxonomy.resolve()),
        "splits": {
            split: sum(row["split"] == split for row in rows)
            for split in args.splits
        },
        "written": sum(row["status"] == "written" for row in rows),
        "reused": sum(row["status"] == "reused" for row in rows),
        "failures": len(failures),
        "rows": sorted(rows, key=lambda row: (row["split"], row["case"], row["floor"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("splits", "written", "reused", "failures")},
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
