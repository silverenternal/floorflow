#!/usr/bin/env python3
"""Audit Raster2Seq proposals through FloorFlow structural completion."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.generation_plausibility import generation_plausibility
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.geometry.core import compose_known_exterior
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.inference.room_repair import (
    drop_unsupported_openings,
    merge_room_fragments,
)
from floorflow.representation.core import extract_room_instances

ROOT = Path(__file__).resolve().parents[1]
ROOM_CATEGORIES = {
    0: "room:Outdoor",
    1: "room:Kitchen",
    2: "room:LivingRoom",
    3: "room:Bedroom",
    4: "room:Bath",
    5: "room:Entry",
    6: "room:Storage",
    7: "room:Garage",
    8: "room:Undefined",
}
OPENING_CATEGORIES = {
    9: "window:Regular",
    10: "door:Swing Beside",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=(
            ROOT
            / "outputs/cache/raster2seq_cc5k_val_refined_v1/annotations/val.json"
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            ROOT / "outputs/external_baselines/raster2seq/official_eval_val503_v1/jsons"
        ),
    )
    parser.add_argument(
        "--official-results",
        type=Path,
        default=(
            ROOT
            / "outputs/external_baselines/raster2seq/official_eval_val503_v1/result_jsons"
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--image-ids", nargs="*", type=int)
    parser.add_argument("--audit-size", type=int, default=256)
    parser.add_argument("--boundary-width", type=int, default=7)
    parser.add_argument("--reconcile-gap-fraction", type=float, default=0.04)
    parser.add_argument("--reconcile-use-ip", action="store_true")
    parser.add_argument("--repair-open-endpoints", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster2seq_floorflow_audit_val503_v1.json",
    )
    return parser.parse_args()


def _audit_taxonomy() -> FloorplanTaxonomy:
    rooms = tuple(ROOM_CATEGORIES.values())
    openings = tuple(OPENING_CATEGORIES.values())
    names = ("background", "wall", *rooms, *openings)
    return FloorplanTaxonomy(
        class_names=names,
        room_map={name.removeprefix("room:"): name for name in rooms},
        furniture_map={},
        door_map={"Swing Beside": "door:Swing Beside"},
        window_map={"Regular": "window:Regular"},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def _polygon_array(segmentation) -> np.ndarray:
    values = np.asarray(segmentation, dtype=np.float32)
    if values.ndim == 2 and values.shape[0] == 1 and values.shape[1] % 2 == 0:
        values = values.reshape(-1, 2)
    if values.ndim == 1:
        values = values.reshape(-1, 2)
    elif values.ndim != 2 or values.shape[-1] != 2:
        raise ValueError(f"unsupported polygon shape: {values.shape}")
    return np.rint(values).astype(np.int32)


def _render(
    annotations: list[dict],
    shape: tuple[int, int],
    taxonomy: FloorplanTaxonomy,
    *,
    normalized: bool = False,
    source_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    output = np.zeros(shape, dtype=np.uint8)
    rooms = []
    openings = []
    for annotation in annotations:
        category = int(annotation["category_id"])
        polygon = _polygon_array(annotation["segmentation"])
        if normalized:
            if source_shape is None:
                raise ValueError("normalized polygons require source_shape")
            scale = min(256.0 / source_shape[0], 256.0 / source_shape[1])
            resized = (
                int(source_shape[0] * scale),
                int(source_shape[1] * scale),
            )
            pad_top = (256 - resized[0]) // 2
            pad_left = (256 - resized[1]) // 2
            polygon = np.rint(
                (
                    polygon.astype(np.float32)
                    - np.asarray([pad_left, pad_top], dtype=np.float32)
                )
                / scale
            ).astype(np.int32)
            if shape != source_shape:
                polygon = np.rint(
                    polygon.astype(np.float32)
                    * np.asarray(
                        [shape[1] - 1, shape[0] - 1],
                        dtype=np.float32,
                    )
                    / np.asarray(
                        [source_shape[1] - 1, source_shape[0] - 1],
                        dtype=np.float32,
                    )
                ).astype(np.int32)
        elif source_shape is not None:
            polygon = np.rint(
                polygon.astype(np.float32)
                * np.asarray(
                    [shape[1] - 1, shape[0] - 1],
                    dtype=np.float32,
                )
                / np.asarray(
                    [source_shape[1] - 1, source_shape[0] - 1],
                    dtype=np.float32,
                )
            ).astype(np.int32)
        if category in ROOM_CATEGORIES:
            rooms.append((polygon, taxonomy.class_id(ROOM_CATEGORIES[category])))
        elif category in OPENING_CATEGORIES:
            openings.append(
                (polygon, taxonomy.class_id(OPENING_CATEGORIES[category]))
            )
    for polygon, class_id in sorted(rooms, key=lambda item: -cv2.contourArea(item[0])):
        cv2.fillPoly(output, [polygon], color=int(class_id))
    for polygon, class_id in openings:
        if len(polygon) == 2:
            cv2.line(
                output,
                tuple(polygon[0]),
                tuple(polygon[1]),
                color=int(class_id),
                thickness=2,
            )
        else:
            cv2.polylines(output, [polygon], True, color=int(class_id), thickness=2)
    return output.astype(np.int64)


def _complete_structure(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    boundary_width: int,
) -> np.ndarray:
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    room_labels = labels.copy()
    room_labels[~np.isin(room_labels, room_ids)] = 0
    room_tensor = torch.from_numpy(room_labels)[None]
    seam = partition_wall_mask(room_tensor, radius=0)[0].numpy()
    completed = room_labels.copy()
    opening_ids = np.asarray(taxonomy.opening_ids, dtype=np.int64)
    preserved = np.isin(labels, opening_ids)
    completed[seam] = taxonomy.class_id("wall")
    completed[preserved] = labels[preserved]
    completed = drop_unsupported_openings(completed, taxonomy, radius=2)
    return compose_known_exterior(
        torch.from_numpy(completed)[None],
        torch.from_numpy(footprint.astype(np.float32))[None],
        boundary_width=boundary_width,
        opening_ids=taxonomy.opening_ids,
    )[0].numpy()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    taxonomy = _audit_taxonomy()
    dataset = json.loads(args.annotations.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for annotation in dataset["annotations"]:
        grouped[int(annotation["image_id"])].append(annotation)
    records = []
    requested_ids = set(args.image_ids or [])
    for case_index, image in enumerate(dataset["images"]):
        image_id = int(image["id"])
        if case_index < int(args.start_index):
            continue
        if requested_ids and image_id not in requested_ids:
            continue
        if args.limit is not None and len(records) >= int(args.limit):
            break
        prediction_path = args.predictions / f"{image_id:05d}.json"
        if not prediction_path.is_file():
            continue
        result_path = args.official_results / f"{image_id:05d}.json"
        if not result_path.is_file():
            continue
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        official = json.loads(result_path.read_text(encoding="utf-8"))
        source_shape = (int(image["height"]), int(image["width"]))
        shape = (args.audit_size, args.audit_size)
        prediction = _render(
            predictions,
            shape,
            taxonomy,
            normalized=True,
            source_shape=source_shape,
        )
        target = _render(
            grouped[image_id],
            shape,
            taxonomy,
            source_shape=source_shape,
        )
        predicted_rooms = np.isin(prediction, taxonomy.room_ids)
        target_rooms = np.isin(target, taxonomy.room_ids)
        completion_labels = merge_room_fragments(
            prediction,
            taxonomy,
            min_area=32,
        )
        footprint = np.isin(completion_labels, taxonomy.room_ids)
        completed = _complete_structure(
            completion_labels,
            footprint,
            taxonomy,
            boundary_width=args.boundary_width,
        )
        suite = room_metric_suite(
            prediction,
            target,
            target_rooms,
            taxonomy,
            predicted_room_support=predicted_rooms,
            predicted_instance_support=predicted_rooms,
            target_room_support=target_rooms,
        )
        instance = suite["matched_instance"]
        plausibility = generation_plausibility(
            completed,
            footprint,
            taxonomy,
            explicit_rooms=True,
            reconcile_gap_fraction=args.reconcile_gap_fraction,
            reconcile_use_ip=args.reconcile_use_ip,
            repair_open_endpoints=args.repair_open_endpoints,
        )
        records.append(
            {
                "image_id": image_id,
                "adapter_room_macro_iou": float(
                    suite["visible_semantic"]["room_macro_iou"]
                ),
                "adapter_instance_f1": float(
                    instance["instance_f1_at_threshold"]
                ),
                "adapter_type_accuracy": float(
                    instance["matched_type_accuracy"]
                ),
                "adapter_predicted_count": float(
                    instance["predicted_instances"]
                ),
                "adapter_target_count": float(
                    len(extract_room_instances(target, taxonomy))
                ),
                "adapter_count_abs_error": float(instance["count_abs_error"]),
                **{
                    f"official_{key}": float(value)
                    for key, value in official.items()
                    if isinstance(value, (int, float))
                },
                **plausibility,
            }
        )
    numeric_keys = [
        key
        for key, value in records[0].items()
        if key != "image_id" and isinstance(value, (int, float))
    ]
    report = {
        "schema_version": "raster2seq-floorflow-audit-v1",
        "scope": "Raster2Seq-proposal hybrid structural audit; not standalone FloorFlow",
        "split": "Raster2Seq val503",
        "cases": len(records),
        "test_opened": False,
        "target_used_at_inference": False,
        "scorer_boundary": (
            "official_* values come from Raster2Seq result_jsons; adapter_* "
            "values are project rasterization diagnostics; structural fields "
            "audit FloorFlow completion."
        ),
        "configuration": {
            "boundary_width": args.boundary_width,
            "reconcile_gap_fraction": args.reconcile_gap_fraction,
            "reconcile_use_ip": bool(args.reconcile_use_ip),
            "repair_open_endpoints": bool(args.repair_open_endpoints),
            "start_index": int(args.start_index),
            "image_ids": sorted(requested_ids),
            "room_source": "Raster2Seq proposals",
            "structure_source": "FloorFlow seam and exterior composition",
            "footprint_source": (
                "predicted room support after 32px fragment merge"
            ),
        },
        "summary": {
            key: _summary([float(row[key]) for row in records])
            for key in numeric_keys
        },
        "case_rows": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
