#!/usr/bin/env python3
"""Create paired pre/post local edits on serialized controls."""

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
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.data.local_control_edits import LOCAL_EDIT_OPERATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation",
        choices=tuple(LOCAL_EDIT_OPERATIONS),
        required=True,
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=191)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--radius-pixels", type=int, default=8)
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
    source_manifest_path = args.source_root / args.split / "manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    edit_function = LOCAL_EDIT_OPERATIONS[args.operation]
    rows = []
    for index, sample in enumerate(dataset.samples):
        name = f"{sample.relative_case}__{sample.floor_id}"
        source_path = control_cache_path(args.source_root, args.split, name)
        controls = read_control_cache(source_path)
        edit = edit_function(controls, dataset.taxonomy, seed=args.seed + index)
        edited_controls = edit.controls if edit is not None else controls
        output_path = control_cache_path(args.output_root, args.split, name)
        write_control_cache(
            output_path,
            room_logits=edited_controls["room_logits"][None],
            program=edited_controls["program"][None],
            structure=edited_controls["structure"][None],
            furniture=edited_controls["furniture"][None],
            support=edited_controls["support"][None],
            room_size=edited_controls["room_logits"].shape[-1],
        )
        summary = summarize_control_file(
            output_path,
            dataset.taxonomy,
            room_size=edited_controls["room_logits"].shape[-1],
            structure_size=edited_controls["structure"].shape[-1],
            furniture_size=edited_controls["furniture"].shape[-1],
            support_size=edited_controls["support"].shape[-1],
        )
        record = {
            "name": name,
            "file": summary.pop("path"),
            "sha256": summary.pop("sha256"),
            "edited": edit is not None,
        }
        if edit is not None:
            height, width = edit.control_size
            x, y, box_width, box_height = edit.bbox
            radius = int(args.radius_pixels)
            record.update(
                {
                    "operation": edit.operation,
                    "control_key": edit.control_key,
                    "control_size": list(edit.control_size),
                    "edit_region_normalized": [
                        (x - radius) / width,
                        (y - radius) / height,
                        (box_width + 2 * radius) / width,
                        (box_height + 2 * radius) / height,
                    ],
                    "changed_pixels": edit.changed_pixels,
                }
            )
        rows.append(record)
        print(f"edited {len(rows)}/{len(dataset)}", flush=True)
    manifest = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": source_manifest["source"],
        "analysis_variant": f"local_edit_{args.operation}",
        "target_used_for_control": source_manifest["target_used_for_control"],
        "deployable": source_manifest["deployable"],
        "split": args.split,
        "cases": len(rows),
        "edited_cases": sum(row["edited"] for row in rows),
        "operation": args.operation,
        "seed": args.seed,
        "radius_pixels": args.radius_pixels,
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "taxonomy_sha256": source_manifest["taxonomy_sha256"],
        "control_sizes": source_manifest["control_sizes"],
        "output_root": str(args.output_root.resolve()),
        "rows": rows,
    }
    path = args.output_root / args.split / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(path),
                "operation": args.operation,
                "cases": len(rows),
                "edited_cases": manifest["edited_cases"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
