#!/usr/bin/env python3
"""Cache CubiCasa5K SVGs as fixed-resolution semantic label maps."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.data.class_balance import write_class_frequency_report
from floorflow.data.dataset import FloorplanSample, collect_samples, load_or_rasterize
from floorflow.data.taxonomy import (
    FloorplanTaxonomy,
    build_taxonomy,
    scan_taxonomy_counts,
)


def _process(
    sample: FloorplanSample,
    image_size: int,
    taxonomy,
    reuse_existing: bool = True,
) -> tuple[str, str]:
    try:
        if reuse_existing and sample.cache_path is not None and sample.cache_path.is_file():
            return "ok", json.dumps({"case": str(sample.relative_case), "floor": sample.floor_id, "reused": True})
        target, footprint = load_or_rasterize(
            sample,
            image_size=image_size,
            taxonomy=taxonomy,
            write_cache=True,
        )
        return "ok", json.dumps(
            {
                "case": str(sample.relative_case),
                "floor": sample.floor_id,
                "target_foreground_fraction": float((target > 0).mean()),
                "footprint_fraction": float(footprint.mean()),
            }
        )
    except Exception as error:  # noqa: BLE001 - report every worker failure.
        return "error", json.dumps(
            {"case": str(sample.relative_case), "floor": sample.floor_id, "error": repr(error)}
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="limit each split for a smoke run")
    parser.add_argument(
        "--limit-by-split",
        nargs="*",
        default=[],
        metavar="SPLIT=COUNT",
        help="Override the per-split case limit without changing case order.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--room-min-count", type=int, default=100)
    parser.add_argument("--furniture-min-count", type=int, default=100)
    parser.add_argument(
        "--taxonomy-source",
        type=Path,
        default=None,
        help="Reuse an existing taxonomy so class ids match the warm-start checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_limits = {}
    for value in args.limit_by_split:
        split, separator, count = value.partition("=")
        if not separator or not split or not count.isdigit():
            raise ValueError(f"invalid --limit-by-split value: {value}")
        split_limits[split] = int(count)
    if args.overwrite:
        for split in args.splits:
            split_root = args.cache_root / split
            if not split_root.exists():
                continue
            for path in split_root.rglob("*.npz"):
                # These are generated cache files under an explicit outputs root.
                path.unlink()

    failures = 0
    summaries: dict[str, int] = {}
    svg_paths = []
    for split in args.splits:
        for sample in collect_samples(args.data_root, split, floors=True):
            if sample.svg_path not in svg_paths:
                svg_paths.append(sample.svg_path)
    taxonomy = (
        FloorplanTaxonomy.read(args.taxonomy_source)
        if args.taxonomy_source is not None
        else build_taxonomy(
            scan_taxonomy_counts(svg_paths),
            room_min_count=args.room_min_count,
            furniture_min_count=args.furniture_min_count,
        )
    )
    args.cache_root.mkdir(parents=True, exist_ok=True)
    taxonomy.write(args.cache_root / "taxonomy.json")
    print(f"taxonomy classes={taxonomy.num_classes}", flush=True)
    for split in args.splits:
        split_limit = split_limits.get(split, args.limit)
        samples = collect_samples(
            args.data_root,
            split,
            cache_root=args.cache_root,
            limit=split_limit,
        )
        print(f"[{split}] {len(samples)} cases", flush=True)
        completed = 0
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(
                    _process,
                    sample,
                    args.image_size,
                    taxonomy,
                    not args.overwrite,
                )
                for sample in samples
            ]
            for future in as_completed(futures):
                status, payload = future.result()
                completed += 1
                if status != "ok":
                    failures += 1
                    print(payload, flush=True)
                elif completed % 100 == 0 or completed == len(samples):
                    print(f"[{split}] {completed}/{len(samples)}", flush=True)
        summaries[split] = len(samples)

    report = {
        "data_root": str(args.data_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "image_size": args.image_size,
        "splits": summaries,
        "failures": failures,
        "class_names": taxonomy.class_names,
        "room_min_count": args.room_min_count,
        "furniture_min_count": args.furniture_min_count,
        "taxonomy_source": (
            str(args.taxonomy_source) if args.taxonomy_source is not None else None
        ),
        "limit_by_split": split_limits,
    }
    args.cache_root.mkdir(parents=True, exist_ok=True)
    frequency_path = write_class_frequency_report(
        args.cache_root,
        class_names=taxonomy.class_names,
        max_weight=20.0,
    )
    report["class_frequency_report"] = str(frequency_path)
    (args.cache_root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
