#!/usr/bin/env python3
"""Evaluate deterministic identity refiners on Raster2Seq val predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Raster2Seq"))

from datasets import build_dataset
from evaluations.rplan_eval.Evaluator import Evaluator_RPlan
from util.eval_utils import compute_f1

from floorflow.inference.raster2seq_identity_refiner import (
    Raster2SeqRefineConfig,
    refine_raster2seq_items,
)

OPENING_IDS = frozenset({9, 10})
METRICS = (
    "room_prec",
    "room_rec",
    "room_iou",
    "corner_prec",
    "corner_rec",
    "angles_prec",
    "angles_rec",
    "room_missing_ratio",
    "room_sem_prec",
    "room_sem_rec",
    "window_door_prec",
    "window_door_rec",
)
CONFIGS = {
    "identity": Raster2SeqRefineConfig(),
    "clean_005": Raster2SeqRefineConfig(simplify_tolerance=0.05),
    "clean_010": Raster2SeqRefineConfig(simplify_tolerance=0.1),
    "clean_020": Raster2SeqRefineConfig(simplify_tolerance=0.2),
    "clean_025": Raster2SeqRefineConfig(simplify_tolerance=0.25),
    "clean_030": Raster2SeqRefineConfig(simplify_tolerance=0.3),
    "clean_050": Raster2SeqRefineConfig(simplify_tolerance=0.5),
    "snap_025": Raster2SeqRefineConfig(snap_tolerance=0.25),
    "snap_035": Raster2SeqRefineConfig(snap_tolerance=0.35),
    "snap_050": Raster2SeqRefineConfig(snap_tolerance=0.5),
    "snap_100": Raster2SeqRefineConfig(snap_tolerance=1.0),
    "snap_150": Raster2SeqRefineConfig(snap_tolerance=1.5),
    "clean_snap_050": Raster2SeqRefineConfig(
        simplify_tolerance=0.25,
        snap_tolerance=0.5,
    ),
    "clean_snap_100": Raster2SeqRefineConfig(
        simplify_tolerance=0.25,
        snap_tolerance=1.0,
    ),
    "dedupe_098": Raster2SeqRefineConfig(duplicate_iou=0.98),
    "clean_snap050_dedupe098": Raster2SeqRefineConfig(
        simplify_tolerance=0.25,
        snap_tolerance=0.5,
        duplicate_iou=0.98,
    ),
    "clean_snap100_dedupe095": Raster2SeqRefineConfig(
        simplify_tolerance=0.25,
        snap_tolerance=1.0,
        duplicate_iou=0.95,
    ),
    "min_area_0_5": Raster2SeqRefineConfig(minimum_room_area=0.5),
    "min_area_1": Raster2SeqRefineConfig(minimum_room_area=1.0),
    "min_area_2": Raster2SeqRefineConfig(minimum_room_area=2.0),
    "min_area_4": Raster2SeqRefineConfig(minimum_room_area=4.0),
    "min_area_8": Raster2SeqRefineConfig(minimum_room_area=8.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/official_eval_val503_v1/jsons",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "outputs/cache/raster2seq_cc5k_val_refined_v1/annotations/val.json",
    )
    parser.add_argument(
        "--official-results",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/official_eval_val503_v1/results.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster2seq_identity_refiner_val_v1.json",
    )
    parser.add_argument(
        "--selected-output",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/identity_refined_val503_v1",
    )
    return parser.parse_args()


def _case_ground_truth(record: dict) -> tuple[list, list, list, list]:
    room_polys = []
    room_types = []
    opening_polys = []
    opening_types = []
    instances = record["instances"]
    for polygon, category in zip(
        instances.gt_masks.polygons,
        instances.gt_classes.detach().cpu().tolist(),
        strict=True,
    ):
        polygon = np.asarray(polygon).reshape(-1, 2).astype(np.int32)
        category = int(category)
        if category in OPENING_IDS:
            opening_polys.append(polygon)
            opening_types.append(category)
        else:
            room_polys.append(polygon)
            room_types.append(category)
    return room_polys, room_types, opening_polys, opening_types


def _metric_row(
    items: list[dict],
    ground_truth: tuple[list, list, list, list],
    evaluator: Evaluator_RPlan,
) -> dict:
    room_polys = []
    room_types = []
    opening_polys = []
    opening_types = []
    for item in items:
        polygon = np.rint(
            np.asarray(item["segmentation"], dtype=np.float64).reshape(-1, 2)
        ).astype(np.int32)
        category = int(item["category_id"])
        if category in OPENING_IDS:
            opening_polys.append(polygon)
            opening_types.append(category)
        else:
            room_polys.append(polygon)
            room_types.append(category)
    values = evaluator.evaluate_scene(
        room_polys=room_polys,
        gt_polys=ground_truth[0],
        room_types=room_types,
        gt_polys_types=ground_truth[1],
        window_door_lines=opening_polys,
        gt_window_doors_list=ground_truth[2],
        window_door_lines_types=opening_types,
        gt_window_doors_type_list=ground_truth[3],
        img_size=(256, 256),
    )
    return {
        name: float(values[name])
        for name in METRICS
    }


def _summary(rows: list[dict]) -> dict:
    means = {
        name: float(np.mean([row[name] for row in rows]))
        for name in METRICS
    }
    f1_inputs = {
        "room": means,
        "corner": means,
        "angles": means,
        "room_sem": means,
        "window_door": means,
    }
    for name, values in f1_inputs.items():
        compute_f1(values, [name])
    return {
        "cases": len(rows),
        **means,
        "room_f1": f1_inputs["room"]["room_f1"],
        "corner_f1": f1_inputs["corner"]["corner_f1"],
        "angles_f1": f1_inputs["angles"]["angles_f1"],
        "room_sem_f1": f1_inputs["room_sem"]["room_sem_f1"],
        "window_door_f1": f1_inputs["window_door"]["window_door_f1"],
    }


def main() -> None:
    args = parse_args()
    dataset = build_dataset(
        "val",
        SimpleNamespace(
            dataset_name="cubicasa",
            dataset_root=str(args.annotations.parents[1]),
            image_size=256,
            image_norm=False,
            poly2seq=True,
            num_bins=32,
            seq_len=512,
            add_cls_token=False,
            per_token_sem_loss=True,
            mask_format="polygon",
            disable_image_transform=False,
            semantic_classes=12,
            wd_only=False,
        ),
    )
    evaluator = Evaluator_RPlan(
        disable_overlapping_filter=True,
        iou_thres=0.5,
        wd_as_line=True,
    )
    rows = {name: [] for name in CONFIGS}
    refined_cases = {name: {} for name in CONFIGS}
    removal_stats = {name: {} for name in CONFIGS}
    for index in range(len(dataset)):
        record = dataset[index]
        image_id = int(record["image_id"])
        prediction_path = args.predictions / f"{image_id:05d}.json"
        items = json.loads(prediction_path.read_text(encoding="utf-8"))
        ground_truth = _case_ground_truth(record)
        for name, config in CONFIGS.items():
            refined, stats = refine_raster2seq_items(items, config)
            rows[name].append(
                {
                    "image_id": image_id,
                    **_metric_row(refined, ground_truth, evaluator),
                }
            )
            refined_cases[name][f"{image_id:05d}"] = refined
            removal_stats[name][f"{image_id:05d}"] = stats
    summaries = {name: _summary(values) for name, values in rows.items()}
    official = json.loads(args.official_results.read_text(encoding="utf-8"))
    identity = summaries["identity"]
    baseline_metrics = (
        "room_f1",
        "room_iou",
        "corner_f1",
        "window_door_f1",
    )
    baseline_error = max(
        abs(float(identity[key]) - float(official[key]))
        for key in baseline_metrics
    )
    semantic_error = abs(
        float(identity["room_sem_f1"]) - float(official["room_sem_f1"])
    )
    ranked = sorted(
        CONFIGS,
        key=lambda name: (
            summaries[name]["room_f1"],
            summaries[name]["room_iou"],
            summaries[name]["corner_f1"],
        ),
        reverse=True,
    )
    selected = ranked[0]
    output = {
        "status": "complete" if baseline_error < 1.0e-10 else "baseline_mismatch",
        "scope": "Raster2Seq official CubiCasa val503 identity refinement",
        "configs": {
            name: {
                "parameters": vars(CONFIGS[name]),
                "metrics": summaries[name],
                "relative_room_f1": summaries[name]["room_f1"] / identity["room_f1"] - 1.0,
            }
            for name in CONFIGS
        },
        "ranking": ranked,
        "selected": selected,
        "official_reproduction": {
            "official": official,
            "identity": identity,
            "maximum_absolute_error": baseline_error,
            "checked_metrics": baseline_metrics,
            "room_semantic_diagnostic_delta": semantic_error,
            "room_semantic_boundary": (
                "Room semantic F1 remains diagnostic because direct dataset "
                "replay does not reproduce the official class mapping exactly; "
                "RoomF1, RoomIoU, corner F1 and opening F1 are exact."
            ),
        },
        "case_rows": {
            name: {
                "metrics": rows[name],
                "removal_stats": removal_stats[name],
            }
            for name in CONFIGS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    args.selected_output.mkdir(parents=True, exist_ok=True)
    selected_dir = args.selected_output / "jsons"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for case_id, items in refined_cases[selected].items():
        (selected_dir / f"{case_id}.json").write_text(
            json.dumps(items, indent=2) + "\n",
            encoding="utf-8",
        )
    (args.selected_output / "selection.json").write_text(
        json.dumps(
            {
                "config": selected,
                "parameters": vars(CONFIGS[selected]),
                "metrics": summaries[selected],
                "baseline_metrics": identity,
                "selected_on": "val503_only",
                "official_test509_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "baseline_error": baseline_error,
                "selected": selected,
                "baseline": identity,
                "selected_metrics": summaries[selected],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if output["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
