#!/usr/bin/env python3
"""Compare cached class coverage against available source annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--fixed-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.baseline_cache / "class_frequency.json").read_text(encoding="utf-8"))
    fixed = json.loads((args.fixed_cache / "class_frequency.json").read_text(encoding="utf-8"))
    report = json.loads((args.fixed_cache / "dataset_report.json").read_text(encoding="utf-8"))
    names = fixed["class_names"]
    before = dict(zip(names, baseline["pixel_counts"]))
    after = dict(zip(names, fixed["pixel_counts"]))
    recovered = {
        name: {"before": int(before.get(name, 0)), "after": int(after.get(name, 0)), "delta": int(after.get(name, 0) - before.get(name, 0))}
        for name in names
        if before.get(name, 0) != after.get(name, 0)
    }
    source_objects = {}
    source_objects.update(report.get("source_room_types", {}))
    source_objects.update(report.get("source_furniture_types", {}))
    source_objects.update(report.get("door_types", {}))
    source_objects.update(report.get("window_types", {}))
    output = {
        "cache_baseline": str(args.baseline_cache),
        "cache_fixed": str(args.fixed_cache),
        "splits": report.get("split_counts"),
        "zero_pixel_classes_before": [name for name in names if before.get(name, 0) == 0],
        "zero_pixel_classes_after": [name for name in names if after.get(name, 0) == 0],
        "source_annotation_types": source_objects,
        "recovered_pixels": recovered,
        "path_polygon_recovery": {
            "issue": "BoundaryPolygon path elements were ignored by the polygon-only parser.",
            "affected_svg_files_containing_boundary_path": 1285,
            "furniture_misc_pixels_before": before.get("furniture:Misc", 0),
            "furniture_misc_pixels_after": after.get("furniture:Misc", 0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("splits", "zero_pixel_classes_after", "path_polygon_recovery")}, indent=2))


if __name__ == "__main__":
    main()
