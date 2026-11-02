#!/usr/bin/env python3
"""Decompose a frozen target-free label set without generating with targets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=503)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def _binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def _boundary(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool) & ~binary_erosion(
        np.asarray(mask, dtype=bool),
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )


def _size_bin(fraction: float) -> str:
    if fraction < 0.02:
        return "lt_2pct"
    if fraction < 0.05:
        return "2_to_5pct"
    if fraction < 0.12:
        return "5_to_12pct"
    return "ge_12pct"


def _mean(rows: list[dict], name: str) -> float:
    return float(np.mean([float(row[name]) for row in rows]))


def main() -> None:
    args = parse_args()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.offset + args.limit,
        augment=False,
    )
    taxonomy = dataset.taxonomy
    room_ids = taxonomy.room_ids
    class_stats = {
        class_id: {
            "name": taxonomy.class_names[class_id],
            "intersection": 0,
            "union": 0,
            "target_area": 0,
            "predicted_area": 0,
            "present_cases": 0,
            "case_ious": [],
        }
        for class_id in room_ids
    }
    size_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"intersection": 0.0, "union": 0.0, "target_area": 0.0}
    )
    cases = []
    boundary_intersection = boundary_union = 0
    support_intersection = support_union = 0
    for index in range(args.offset, len(dataset)):
        label_path = args.labels_dir / f"case_{index:04d}.npy"
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        prediction = np.load(label_path)
        target, footprint, _floor_id, name = dataset[index]
        target = target.numpy()
        footprint = footprint.numpy().astype(bool)
        if prediction.shape != target.shape:
            raise ValueError(
                f"raster mismatch for {name}: {prediction.shape} vs {target.shape}"
            )
        predicted_rooms = np.isin(prediction, room_ids)
        target_rooms = np.isin(target, room_ids)
        support_intersection += int((predicted_rooms & target_rooms).sum())
        support_union += int((predicted_rooms | target_rooms).sum())
        predicted_boundary = _boundary(predicted_rooms & footprint)
        target_boundary = _boundary(target_rooms & footprint)
        boundary_intersection += int((predicted_boundary & target_boundary).sum())
        boundary_union += int((predicted_boundary | target_boundary).sum())
        suite = room_metric_suite(
            prediction,
            target,
            footprint,
            taxonomy,
            predicted_room_support=predicted_rooms,
            predicted_instance_support=predicted_rooms,
            target_room_support=target_rooms,
        )
        instance = suite["matched_instance"]
        footprint_area = max(1, int(footprint.sum()))
        for class_id, stats in class_stats.items():
            predicted = prediction == class_id
            expected = target == class_id
            intersection = int((predicted & expected).sum())
            union = int((predicted | expected).sum())
            stats["intersection"] += intersection
            stats["union"] += union
            stats["target_area"] += int(expected.sum())
            stats["predicted_area"] += int(predicted.sum())
            if expected.any():
                stats["present_cases"] += 1
                stats["case_ious"].append(_binary_iou(predicted, expected))
                bucket = size_stats[
                    _size_bin(float(expected.sum()) / footprint_area)
                ]
                bucket["intersection"] += intersection
                bucket["union"] += union
                bucket["target_area"] += int(expected.sum())
        cases.append(
            {
                "name": name,
                "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
                "room_support_iou": suite["occlusion_aware_support"][
                    "room_support_iou"
                ],
                "instance_geometric_iou": instance["matched_geometric_iou"],
                "instance_room_f1": instance["instance_f1_at_threshold"],
                "instance_type_accuracy": instance["matched_type_accuracy"],
                "matched_class_constrained_iou": instance[
                    "matched_class_constrained_iou"
                ],
                "predicted_instances": instance["predicted_instances"],
                "target_instances": instance["target_instances"],
                "coverage": float(
                    (predicted_rooms & footprint).sum()
                    / max(1, footprint.sum())
                ),
            }
        )
    classes = {}
    for class_id, stats in class_stats.items():
        classes[stats["name"]] = {
            "class_id": class_id,
            "global_iou": (
                float(stats["intersection"] / stats["union"])
                if stats["union"]
                else 1.0
            ),
            "mean_present_case_iou": (
                float(np.mean(stats["case_ious"]))
                if stats["case_ious"]
                else 1.0
            ),
            "target_area": stats["target_area"],
            "predicted_area": stats["predicted_area"],
            "present_cases": stats["present_cases"],
            "unexplained_target_area": stats["target_area"]
            - stats["intersection"],
        }
    ranked = sorted(
        classes.values(),
        key=lambda row: (-row["unexplained_target_area"], row["global_iou"]),
    )[:3]
    report = {
        "protocol": {
            "split": args.split,
            "cases": len(cases),
            "labels_dir": str(args.labels_dir),
            "targets_used_for_generation": False,
            "targets_used_for_metrics_only": True,
        },
        "aggregate": {
            key: _mean(cases, key)
            for key in (
                "room_macro_iou",
                "room_support_iou",
                "instance_geometric_iou",
                "instance_room_f1",
                "instance_type_accuracy",
                "matched_class_constrained_iou",
                "coverage",
            )
        },
        "support_iou": float(
            support_intersection / max(1, support_union)
        ),
        "boundary_f1": float(
            2.0
            * boundary_intersection
            / max(1.0, boundary_union + boundary_intersection)
        ),
        "per_class": classes,
        "per_size": {
            key: float(value["intersection"] / max(1.0, value["union"]))
            for key, value in size_stats.items()
        },
        "largest_recoverable_target_area": ranked,
        "worst_cases": sorted(cases, key=lambda row: row["room_macro_iou"])[:20],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(
        json.dumps(
            {
                "boundary_f1": report["boundary_f1"],
                "per_size": report["per_size"],
                "largest_recoverable": ranked,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
