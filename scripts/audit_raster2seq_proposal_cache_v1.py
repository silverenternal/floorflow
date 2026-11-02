#!/usr/bin/env python3
"""Audit Raster2Seq proposal-cache integrity and teacher quality."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from floorflow.data.dataset import CubiCasaRasterControlDataset, load_or_rasterize
from floorflow.data.raster2seq_proposal_cache import (
    iter_teacher_cases,
    read_manifest,
    sha256_file,
)
from floorflow.data.raster_control import crop_floor_masks
from floorflow.evaluation.raster2seq_external import render_raster2seq_predictions
from floorflow.evaluation.room_corner_metrics import room_corner_angle_metrics
from floorflow.evaluation.room_metric_suite import room_metric_suite

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--floorflow-cache",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/raster2seq_proposal_cache_contract_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster2seq_teacher_cache_audit_v1.json",
    )
    parser.add_argument("--expected-cases", type=int, default=None)
    parser.add_argument("--page-floor-crop", action="store_true")
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    numeric = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        key: _summary([float(row[key]) for row in rows])
        for key in numeric
    }


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    split = str(manifest["split"])
    taxonomy_path = args.floorflow_cache / "taxonomy.json"
    if sha256_file(taxonomy_path) != manifest["taxonomy_sha256"]:
        raise ValueError("taxonomy hash does not match the teacher manifest")
    dataset = CubiCasaRasterControlDataset(
        args.data_root,
        split,
        image_size=256,
        cache_root=args.floorflow_cache,
        augment=False,
        limit=None,
        image_mode="native",
    )
    sample_map = {
        f"{sample.relative_case}__{sample.floor_id}": sample
        for sample in dataset.samples
    }
    expected_keys = [
        f"{sample.relative_case}__{sample.floor_id}"
        for sample in dataset.samples[
            int(manifest["start_index"]) :
            int(manifest["start_index"]) + int(manifest["limit"])
        ]
    ]
    row_keys = [str(row["sample_key"]) for row in manifest["rows"]]
    if row_keys != expected_keys:
        raise ValueError("manifest rows do not match the frozen sample order")
    if args.expected_cases is not None and len(row_keys) != int(args.expected_cases):
        raise ValueError("manifest case count differs from the expected count")
    manifest_root = args.manifest.parent
    case_rows = []
    for row, case in zip(
        manifest["rows"],
        iter_teacher_cases(args.manifest, taxonomy=dataset.taxonomy),
        strict=True,
    ):
        if str(row["sample_key"]) != case.sample_key:
            raise ValueError("manifest and case sample keys differ")
        if int(row["split_index"]) != case.split_index:
            raise ValueError("manifest and case split indices differ")
        cache_path = manifest_root / str(row["cache_file"])
        if sha256_file(cache_path) != row["cache_sha256"]:
            raise ValueError(f"cache hash mismatch for {case.sample_key}")
        sample = sample_map[case.sample_key]
        if sha256_file(sample.raster_path) != case.source_image_sha256:
            raise ValueError(f"source raster hash mismatch for {case.sample_key}")
        target, footprint = load_or_rasterize(
            sample,
            image_size=256,
            taxonomy=dataset.taxonomy,
            write_cache=False,
        )
        if args.page_floor_crop:
            target, footprint = crop_floor_masks(
                target,
                footprint,
                image_size=256,
            )
        polygons = [
            np.rint(
                polygon * float(case.image_size - 1)
            ).astype(np.int32)
            for polygon in case.polygons
        ]
        prediction = render_raster2seq_predictions(
            polygons,
            case.class_ids,
            (case.image_size, case.image_size),
            dataset.taxonomy,
        )
        predicted_rooms = np.isin(prediction, dataset.taxonomy.room_ids)
        target_rooms = np.isin(target, dataset.taxonomy.room_ids)
        suite = room_metric_suite(
            prediction,
            target,
            footprint.astype(bool),
            dataset.taxonomy,
            predicted_room_support=predicted_rooms,
            predicted_instance_support=predicted_rooms,
            target_room_support=target_rooms,
        )
        instance = suite["matched_instance"]
        corner = room_corner_angle_metrics(
            prediction,
            target,
            dataset.taxonomy,
            footprint=footprint,
        )
        case_rows.append(
            {
                "sample_key": case.sample_key,
                "split_index": case.split_index,
                "subsets": list(row.get("subsets", [])),
                "room_count": len(case.polygons),
                "room_macro_iou": float(
                    suite["visible_semantic"]["room_macro_iou"]
                ),
                "room_support_iou": float(
                    suite["occlusion_aware_support"]["room_support_iou"]
                ),
                "instance_geometric_iou": float(
                    instance["matched_geometric_iou"]
                ),
                "instance_f1": float(
                    instance["instance_f1_at_threshold"]
                ),
                "type_accuracy": float(
                    instance["matched_type_accuracy"]
                ),
                "count_abs_error": float(instance["count_abs_error"]),
                "corner_f1": float(corner["corner_f1"]),
                "angle_f1": float(corner["angle_f1"]),
                "mean_existence": (
                    float(case.existence_scores.mean())
                    if len(case.existence_scores)
                    else 0.0
                ),
            }
        )
    requirements = {
        "complete_coverage": len(case_rows) == len(expected_keys),
        "no_target_room_pixels_or_room_masks": (
            manifest["input_protocol"]["target_room_class_pixels_used"] is False
            and manifest["input_protocol"]["target_room_masks_used_for_teacher_output"]
            is False
        ),
        "deployable_student_target_free": (
            manifest["student_inference_target_free"] is True
        ),
        "teacher_classes_untrusted": (
            manifest["teacher_class_outputs_are_trusted"] is False
        ),
        "source_hashes_match": True,
        "cache_hashes_match": True,
        "taxonomy_matches": True,
        "sample_order_matches": True,
    }
    summary = _aggregate(case_rows)
    quality = {
        "room_macro_iou": summary["room_macro_iou"],
        "room_support_iou": summary["room_support_iou"],
        "instance_geometric_iou": summary["instance_geometric_iou"],
        "instance_f1": summary["instance_f1"],
        "type_accuracy": summary["type_accuracy"],
        "count_abs_error": summary["count_abs_error"],
        "corner_f1": summary["corner_f1"],
        "angle_f1": summary["angle_f1"],
    }
    gate = contract["teacher_quality_gate"]
    quality_gate = {
        "room_support_iou": (
            quality["room_support_iou"]["mean"]
            >= float(gate["calibration64_room_support_iou_min"])
        ),
        "instance_f1": (
            quality["instance_f1"]["mean"]
            >= float(gate["calibration64_instance_f1_min"])
        ),
        "instance_geometric_iou": (
            quality["instance_geometric_iou"]["mean"]
            >= float(gate["calibration64_instance_geometric_iou_min"])
        ),
        "count_abs_error": (
            quality["count_abs_error"]["mean"]
            <= float(gate["calibration64_count_abs_error_max"])
        ),
    }
    quality_gate["passed"] = all(quality_gate.values())
    subset_rows: dict[str, list[dict]] = defaultdict(list)
    for row in case_rows:
        for subset in row["subsets"]:
            subset_rows[subset].append(row)
    report = {
        "schema_version": "raster2seq-teacher-cache-audit-v1",
        "scope": "Raster2Seq proposal teacher audit; target labels are evaluation-only",
        "manifest": str(args.manifest),
        "split": split,
        "cases": len(case_rows),
        "test_opened": False,
        "target_used_at_inference": False,
        "teacher_role": "predicted_control_not_ground_truth",
        "requirements": requirements,
        "quality": quality,
        "quality_gate": quality_gate,
        "teacher_class_outputs_usable": False,
        "student_use": (
            "class-agnostic proposal identity and mask geometry only"
            if quality_gate["passed"]
            else "do not distill"
        ),
        "subsets": {
            name: {
                "cases": len(rows),
                "quality": _aggregate(rows),
            }
            for name, rows in sorted(subset_rows.items())
        },
        "case_rows": case_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
