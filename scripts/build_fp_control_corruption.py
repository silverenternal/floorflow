#!/usr/bin/env python3
"""Create reproducible spatial-drop controls for robustness experiments."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from floorflow.data.control_cache import (
    control_cache_path,
    read_control_cache,
    write_control_cache,
)
from floorflow.data.control_contract import (
    CONTROL_SCHEMA_VERSION,
    sha256_file,
    summarize_control_file,
)
from floorflow.data.control_perturbation import (
    control_validity_masks,
    corrupt_controls,
)
from floorflow.data.dataset import CubiCasaFloorplanDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rates", nargs="+", type=float, default=(0.1, 0.25, 0.5))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--furniture-size", type=int, default=192)
    parser.add_argument("--support-size", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
        write_cache=False,
    )
    source_manifest_path = args.control_root / args.split / "manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    for rate in args.rates:
        if rate <= 0.0 or rate >= 1.0:
            raise ValueError(f"drop rate must be strictly between 0 and 1: {rate}")
        rate_name = f"p{round(rate * 100):03d}"
        variant_root = args.output_root / rate_name
        rows = []
        for sample in base_dataset.samples:
            name = f"{sample.relative_case}__{sample.floor_id}"
            source_path = control_cache_path(args.control_root, args.split, name)
            target_path = control_cache_path(variant_root, args.split, name)
            controls = read_control_cache(source_path)
            corrupted = corrupt_controls(
                controls,
                rate,
                seed=args.seed + len(rows),
            )
            validity_masks = control_validity_masks(
                controls,
                rate,
                seed=args.seed + len(rows),
            )
            write_control_cache(
                target_path,
                room_logits=corrupted["room_logits"][None],
                program=corrupted["program"][None],
                structure=corrupted["structure"][None],
                furniture=corrupted["furniture"][None],
                support=corrupted["support"][None],
                room_size=args.room_size,
                validity_masks={
                    key: value.cpu().numpy()
                    for key, value in validity_masks.items()
                },
            )
            summary = summarize_control_file(
                target_path,
                base_dataset.taxonomy,
                room_size=args.room_size,
                structure_size=args.structure_size,
                furniture_size=args.furniture_size,
                support_size=args.support_size,
            )
            rows.append(
                {
                    "name": name,
                    "file": summary.pop("path"),
                    **summary,
                }
            )
        manifest = {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "source": "oracle",
            "analysis_variant": (
                f"spatial_block_drop_{rate_name}_seed_{args.seed}"
            ),
            "target_used_for_control": True,
            "deployable": False,
            "split": args.split,
            "cases": len(rows),
            "source_manifest": str(source_manifest_path.resolve()),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "taxonomy_sha256": source_manifest["taxonomy_sha256"],
            "control_sizes": source_manifest["control_sizes"],
            "source_contract": {
                key: source_manifest.get(key)
                for key in (
                    "source",
                    "taxonomy_sha256",
                    "control_sizes",
                )
            },
            "drop_fraction": rate,
            "seed": args.seed,
            "output_root": str(variant_root.resolve()),
            "rows": rows,
        }
        path = variant_root / args.split / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "variant": rate_name,
                    "cases": len(rows),
                    "manifest": str(path),
                }
            )
        )


if __name__ == "__main__":
    main()
