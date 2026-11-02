#!/usr/bin/env python3
"""Write a reproducibility report for the prepared floor-level dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.data.dataset import read_split
from floorflow.data.svg import list_cubicasa_floors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    args = parser.parse_args()
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    frequency = json.loads((args.cache_root / "class_frequency.json").read_text(encoding="utf-8"))
    split_counts = {
        split: len(list((args.cache_root / split).rglob("*.npz")))
        for split in ("train", "val", "test")
    }
    room_rare = [
        name for name, target in taxonomy.room_map.items()
        if target == "room_rare"
    ]
    furniture_rare = [
        name for name, target in taxonomy.furniture_map.items()
        if target == "furniture_rare"
    ]
    floor_mismatches = []
    for split in ("train", "val", "test"):
        for case in read_split(args.data_root / f"{split}.txt"):
            case_dir = args.data_root / case
            raster_floors = sorted(path.name for path in case_dir.glob("F*_scaled.png"))
            semantic_floors = list_cubicasa_floors(case_dir / "model.svg")
            if len(raster_floors) != len(semantic_floors):
                floor_mismatches.append(
                    {
                        "split": split,
                        "case": str(case),
                        "raster_floors": raster_floors,
                        "semantic_floors": semantic_floors,
                        "resolution": "SVG semantic floors are source of truth; raster-only floors have no target labels",
                    }
                )
    report = {
        "cache_root": str(args.cache_root),
        "split_counts": split_counts,
        "num_classes": taxonomy.num_classes,
        "num_target_channels": taxonomy.num_target_channels,
        "class_names": list(taxonomy.class_names),
        "pixel_counts": frequency["pixel_counts"],
        "pixel_fractions": frequency["pixel_fractions"],
        "source_room_types": taxonomy.source_counts.get("room", {}),
        "source_furniture_types": taxonomy.source_counts.get("furniture", {}),
        "rare_room_types_grouped": room_rare,
        "rare_furniture_types_grouped": furniture_rare,
        "door_types": taxonomy.source_counts.get("door", {}),
        "window_types": taxonomy.source_counts.get("window", {}),
        "floor_count_mismatch_cases": len(floor_mismatches),
        "floor_mismatches": floor_mismatches,
    }
    output = args.output or (args.cache_root / "dataset_report.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "split_counts": split_counts}, indent=2))


if __name__ == "__main__":
    main()
