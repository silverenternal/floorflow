#!/usr/bin/env python3
"""Build fallback-fused controls from corrupted serialized inputs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from floorflow.data.control_cache import (
    control_cache_path,
    read_control_cache,
    read_control_validity,
    write_control_cache,
)
from floorflow.data.control_contract import (
    CONTROL_SCHEMA_VERSION,
    sha256_file,
    summarize_control_file,
)
from floorflow.data.control_fusion import fuse_controls, inpaint_controls
from floorflow.data.dataset import CubiCasaFloorplanDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--corrupt-root", type=Path, required=True)
    parser.add_argument("--fallback-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--fill-mode",
        choices=("external", "inpaint"),
        default="external",
    )
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--furniture-size", type=int, default=192)
    parser.add_argument("--support-size", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
        write_cache=False,
    )
    corrupt_manifest_path = args.corrupt_root / args.split / "manifest.json"
    corrupt_manifest = json.loads(
        corrupt_manifest_path.read_text(encoding="utf-8")
    )
    fallback_manifest = None
    fallback_manifest_path = None
    if args.fill_mode == "external":
        if args.fallback_root is None:
            raise ValueError("--fallback-root is required for external fill")
        fallback_manifest_path = args.fallback_root / args.split / "manifest.json"
        fallback_manifest = json.loads(
            fallback_manifest_path.read_text(encoding="utf-8")
        )
        if fallback_manifest["split"] != args.split:
            raise ValueError("fallback control split mismatch")
        if fallback_manifest["source"] != "predicted":
            raise ValueError("fallback controls must come from a predicted source")
    rows = []
    for sample in dataset.samples:
        name = f"{sample.relative_case}__{sample.floor_id}"
        primary_path = control_cache_path(
            args.corrupt_root,
            args.split,
            name,
        )
        output_path = control_cache_path(args.output_root, args.split, name)
        primary = read_control_cache(primary_path)
        masks = read_control_validity(primary_path)
        if args.fill_mode == "inpaint":
            fused = inpaint_controls(primary, masks)
        else:
            fallback_path = control_cache_path(
                args.fallback_root,
                args.split,
                name,
            )
            fused = fuse_controls(
                primary,
                read_control_cache(fallback_path),
                masks,
            )
        write_control_cache(
            output_path,
            room_logits=fused["room_logits"][None],
            program=fused["program"][None],
            structure=fused["structure"][None],
            furniture=fused["furniture"][None],
            support=fused["support"][None],
            room_size=args.room_size,
            validity_masks=masks,
        )
        summary = summarize_control_file(
            output_path,
            dataset.taxonomy,
            room_size=args.room_size,
            structure_size=args.structure_size,
            furniture_size=args.furniture_size,
            support_size=args.support_size,
        )
        rows.append({"name": name, "file": summary.pop("path"), **summary})

    manifest = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": "oracle",
        "analysis_variant": f"{args.fill_mode}_fill",
        "target_used_for_control": True,
        "deployable": False,
        "split": args.split,
        "cases": len(rows),
        "corrupt_manifest": str(corrupt_manifest_path.resolve()),
        "corrupt_manifest_sha256": sha256_file(corrupt_manifest_path),
        "fallback_manifest": (
            str(fallback_manifest_path.resolve())
            if fallback_manifest_path is not None
            else None
        ),
        "fallback_manifest_sha256": (
            sha256_file(fallback_manifest_path)
            if fallback_manifest_path is not None
            else None
        ),
        "taxonomy_sha256": corrupt_manifest["taxonomy_sha256"],
        "control_sizes": corrupt_manifest["control_sizes"],
        "drop_fraction": corrupt_manifest["drop_fraction"],
        "output_root": str(args.output_root.resolve()),
        "rows": rows,
    }
    path = args.output_root / args.split / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(path),
                "drop_fraction": manifest["drop_fraction"],
                "cases": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
