#!/usr/bin/env python3
"""Audit official Raster2Seq room polygons on the FloorFlow room metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_metric_suite import room_metric_suite

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
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _annotation_polygon(item: dict) -> np.ndarray:
    return np.asarray(item["segmentation"][0], dtype=np.float64).reshape(-1, 2)


def _case_metrics(
    prediction_rows: list[dict],
    annotation: dict,
    taxonomy: FloorplanTaxonomy,
    size: int,
) -> dict[str, float | int]:
    category_to_id = {
        category: taxonomy.class_id(name)
        for category, name in CATEGORY_TO_ROOM.items()
    }
    prediction_image = Image.new("I", (size, size), 0)
    prediction_draw = ImageDraw.Draw(prediction_image)
    for item in prediction_rows:
        category = int(item["category_id"])
        if category not in category_to_id:
            continue
        polygon = np.asarray(item["segmentation"], dtype=np.float64).reshape(-1, 2)
        prediction_draw.polygon(
            [tuple(point) for point in polygon],
            fill=int(category_to_id[category]),
        )
    image = annotation["images"][0]
    height, width = int(image["height"]), int(image["width"])
    scale = min(size / height, size / width)
    resized_height = int(height * scale)
    resized_width = int(width * scale)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    target_image = Image.new("I", (size, size), 0)
    target_draw = ImageDraw.Draw(target_image)
    for item in annotation["annotations"]:
        category = int(item["category_id"])
        if category not in category_to_id:
            continue
        polygon = _annotation_polygon(item)
        polygon[:, 0] = polygon[:, 0] * scale + left
        polygon[:, 1] = polygon[:, 1] * scale + top
        target_draw.polygon(
            [tuple(point) for point in polygon],
            fill=int(category_to_id[category]),
        )
    prediction = np.asarray(prediction_image)
    target = np.asarray(target_image)
    footprint = (target > 0).astype(np.uint8)
    suite = room_metric_suite(
        prediction,
        target,
        footprint,
        taxonomy,
        predicted_room_support=prediction > 0,
        predicted_instance_support=prediction > 0,
        target_room_support=target > 0,
    )
    instance = suite["matched_instance"]
    return {
        "room_iou": float(suite["visible_semantic"]["room_macro_iou"]),
        "room_support_iou": float(
            suite["occlusion_aware_support"]["room_support_iou"]
        ),
        "instance_f1": float(instance["instance_f1_at_threshold"]),
        "instance_precision": float(
            instance["instance_precision_at_threshold"]
        ),
        "instance_recall": float(instance["instance_recall_at_threshold"]),
        "instance_type_accuracy": float(instance["matched_type_accuracy"]),
        "instance_count_abs_error": float(instance["count_abs_error"]),
        "predicted_instances": int(instance["predicted_instances"]),
        "target_instances": int(instance["target_instances"]),
    }


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    rows = []
    for path in sorted(args.predictions.glob("*.json")):
        annotation_path = args.annotations / path.name
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        prediction = json.loads(path.read_text(encoding="utf-8"))
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "case": path.stem,
                **_case_metrics(
                    prediction,
                    annotation,
                    taxonomy,
                    int(args.size),
                ),
            }
        )
    if not rows:
        raise ValueError("no prediction JSON files were found")
    names = [
        key
        for key in rows[0]
        if key != "case" and not key.startswith("predicted_")
        and not key.startswith("target_")
    ]
    report = {
        "schema_version": "raster2seq-official-room-audit-v1",
        "cases": len(rows),
        "size": int(args.size),
        "summary": {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in names
        },
        "rows": rows,
        "claim_boundary": (
            "Official reconstruction checkpoint audited on a named small "
            "subset; not a full-split external comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

