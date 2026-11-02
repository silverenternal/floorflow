#!/usr/bin/env python3
"""Fuse room, structure, furniture, and support controls from separate roots."""

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
from floorflow.data.control_contract import CONTROL_SCHEMA_VERSION, sha256_file
from floorflow.data.dataset import CubiCasaFloorplanDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-root", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--furniture-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=64)
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
    roots = {
        "room_logits": args.room_root,
        "structure": args.structure_root,
        "furniture": args.furniture_root,
        "support": args.support_root,
    }
    manifests = {
        key: json.loads(
            (root / args.split / "manifest.json").read_text(encoding="utf-8")
        )
        for key, root in roots.items()
    }
    source_provenance = {
        (manifest["source"], manifest["target_used_for_control"], manifest["deployable"])
        for manifest in manifests.values()
    }
    if len(source_provenance) != 1:
        raise ValueError("component roots must share control provenance")
    rows = []
    for sample in dataset.samples:
        name = f"{sample.relative_case}__{sample.floor_id}"
        room = read_control_cache(
            control_cache_path(args.room_root, args.split, name)
        )
        structure = read_control_cache(
            control_cache_path(args.structure_root, args.split, name)
        )
        furniture = read_control_cache(
            control_cache_path(args.furniture_root, args.split, name)
        )
        support = read_control_cache(
            control_cache_path(args.support_root, args.split, name)
        )
        path = control_cache_path(args.output_root, args.split, name)
        write_control_cache(
            path,
            room_logits=room["room_logits"][None],
            program=room["program"][None],
            structure=structure["structure"][None],
            furniture=furniture["furniture"][None],
            support=support["support"][None],
            room_size=room["room_logits"].shape[-1],
        )
        rows.append(
            {
                "name": name,
                "file": str(path),
                "sha256": sha256_file(path),
            }
        )
        print(f"fused {len(rows)}/{len(dataset)}", flush=True)
    source, target_used, deployable = next(iter(source_provenance))
    manifest = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": source,
        "analysis_variant": "component_fusion",
        "target_used_for_control": target_used,
        "deployable": deployable,
        "split": args.split,
        "cases": len(rows),
        "component_roots": {
            key: str(root.resolve()) for key, root in roots.items()
        },
        "component_manifests": {
            key: {
                "path": str(
                    (root / args.split / "manifest.json").resolve()
                ),
                "sha256": sha256_file(root / args.split / "manifest.json"),
            }
            for key, root in roots.items()
        },
        "control_sizes": {
            "room": room["room_logits"].shape[-1],
            "structure": structure["structure"].shape[-1],
            "furniture": furniture["furniture"].shape[-1],
            "support": support["support"].shape[-1],
        },
        "taxonomy_sha256": sha256_file(args.cache_root / "taxonomy.json"),
        "output_root": str(args.output_root.resolve()),
        "rows": rows,
    }
    path = args.output_root / args.split / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
