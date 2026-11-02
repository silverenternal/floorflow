#!/usr/bin/env python3
"""Create a room-only control arm from serialized oracle controls."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

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
    parser.add_argument("--limit", type=int, default=58)
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
    source_manifest_path = args.control_root / args.split / "manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    rows = []
    for sample in dataset.samples:
        name = f"{sample.relative_case}__{sample.floor_id}"
        source_path = control_cache_path(args.control_root, args.split, name)
        controls = read_control_cache(source_path)
        zero_structure = torch.zeros_like(controls["structure"])
        zero_furniture = torch.zeros_like(controls["furniture"])
        zero_support = torch.zeros_like(controls["support"])
        target_path = control_cache_path(args.output_root, args.split, name)
        write_control_cache(
            target_path,
            room_logits=controls["room_logits"][None],
            program=controls["program"][None],
            structure=zero_structure[None],
            furniture=zero_furniture[None],
            support=zero_support[None],
            room_size=controls["room_logits"].shape[-1],
        )
        summary = summarize_control_file(
            target_path,
            dataset.taxonomy,
            room_size=controls["room_logits"].shape[-1],
            structure_size=controls["structure"].shape[-1],
            furniture_size=controls["furniture"].shape[-1],
            support_size=controls["support"].shape[-1],
        )
        rows.append({"name": name, "file": summary.pop("path"), **summary})
    manifest = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": "oracle",
        "analysis_variant": "room_only",
        "target_used_for_control": True,
        "deployable": False,
        "split": args.split,
        "cases": len(rows),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "taxonomy_sha256": source_manifest["taxonomy_sha256"],
        "control_sizes": source_manifest["control_sizes"],
        "output_root": str(args.output_root.resolve()),
        "rows": rows,
    }
    path = args.output_root / args.split / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
