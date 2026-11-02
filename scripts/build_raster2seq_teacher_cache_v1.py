#!/usr/bin/env python3
"""Convert official Raster2Seq room JSONs into the FloorFlow teacher cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.data.raster2seq_proposal_cache import (
    MANIFEST_VERSION,
    write_teacher_case,
)
from floorflow.data.taxonomy import FloorplanTaxonomy

CATEGORY_TO_ROOM = {
    1: "room:Kitchen",
    2: "room:LivingRoom",
    3: "room:Bedroom",
    4: "room:Bath",
    5: "room:Entry",
    6: "room:Storage",
    7: "room:Garage",
    8: "room:Undefined",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    by_safe_name = {
        str(row["safe_name"]): row
        for row in mapping["rows"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    skipped_polygons = 0
    for path in sorted(args.predictions.glob("*.json")):
        try:
            source = by_safe_name[path.stem]
        except KeyError as error:
            raise KeyError(f"prediction is absent from mapping: {path.stem}") from error
        prediction = json.loads(path.read_text(encoding="utf-8"))
        polygons = []
        class_ids = []
        class_names = []
        for item in prediction:
            category = int(item["category_id"])
            if category not in CATEGORY_TO_ROOM:
                continue
            class_name = CATEGORY_TO_ROOM[category]
            polygon = np.asarray(item["segmentation"], dtype=np.float32)
            if (
                polygon.ndim != 2
                or polygon.shape[1:] != (2,)
                or len(np.unique(polygon, axis=0)) < 3
                or np.any(polygon < 0.0)
                or np.any(polygon > float(args.image_size - 1))
            ):
                skipped_polygons += 1
                continue
            polygons.append(polygon)
            class_ids.append(taxonomy.class_id(class_name))
            class_names.append(class_name)
        cache_file = f"cases/{path.stem}.npz"
        write_teacher_case(
            args.output_dir / cache_file,
            sample_key=str(source["sample_key"]),
            split=str(source["split"]),
            split_index=int(source["split_index"]),
            image_size=int(args.image_size),
            polygons=polygons,
            class_ids=class_ids,
            class_names=class_names,
            existence_scores=[1.0] * len(polygons),
            sequence_length=sum(len(polygon) for polygon in polygons),
            source_image_sha256=str(
                source.get(
                    "prepared_image_sha256",
                    source["source_image_sha256"],
                )
            ),
        )
        manifest_rows.append(
            {
                "sample_key": str(source["sample_key"]),
                "cache_file": cache_file,
                "source_image": str(source["source_image"]),
            }
        )
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "target_used_at_inference": False,
        "source": "official Raster2Seq cubicasa5k checkpoint",
        "skipped_polygons": skipped_polygons,
        "rows": manifest_rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(manifest_rows),
                "skipped_polygons": skipped_polygons,
                "manifest": str(args.manifest),
            }
        )
    )


if __name__ == "__main__":
    main()

