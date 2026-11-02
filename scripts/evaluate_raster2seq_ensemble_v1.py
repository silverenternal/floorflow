#!/usr/bin/env python3
"""Evaluate deterministic polygon ensembles of Raster2Seq checkpoints on val."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Raster2Seq"))

from datasets import build_dataset
from evaluate_raster2seq_identity_refiner_v1 import (
    _case_ground_truth,
    _metric_row,
    _summary,
)
from evaluations.rplan_eval.Evaluator import Evaluator_RPlan

from floorflow.inference.raster2seq_identity_refiner import fuse_raster2seq_items

BASELINE = "alpha_1.00"
CONFIGS = {
    BASELINE: ("1.00", None, None, 0.5),
    "alpha_0.25": ("0.25", None, None, 0.5),
    "alpha_0.50": ("0.50", None, None, 0.5),
    "alpha_0.75": ("0.75", None, None, 0.5),
    "union_050_075_iou050": ("0.50", "0.75", "union", 0.5),
    "union_025_075_iou050": ("0.25", "0.75", "union", 0.5),
    "intersection_050_075_iou050": ("0.50", "0.75", "intersection", 0.5),
    "intersection_025_075_iou050": ("0.25", "0.75", "intersection", 0.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "outputs/cache/raster2seq_cc5k_val_refined_v1",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/"
        "official_eval_val503_blend_{alpha}_v1/jsons",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster2seq_ensemble_val_v1.json",
    )
    return parser.parse_args()


def _load_items(directory: Path, image_id: int) -> list[dict]:
    return json.loads(
        (directory / f"{image_id:05d}.json").read_text(encoding="utf-8")
    )


def main() -> None:
    args = parse_args()
    dataset = build_dataset(
        "val",
        SimpleNamespace(
            dataset_name="cubicasa",
            dataset_root=str(args.dataset_root),
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
    for index in range(len(dataset)):
        record = dataset[index]
        image_id = int(record["image_id"])
        ground_truth = _case_ground_truth(record)
        predictions = {
            alpha: _load_items(
                Path(
                    str(args.prediction_root).format(alpha=alpha)
                ),
                image_id,
            )
            for alpha in ("0.25", "0.50", "0.75", "1.00")
        }
        for name, (primary_alpha, secondary_alpha, mode, threshold) in CONFIGS.items():
            primary = predictions[primary_alpha]
            if secondary_alpha is None:
                items = primary
            else:
                items, _stats = fuse_raster2seq_items(
                    primary,
                    predictions[secondary_alpha],
                    mode=str(mode),
                    match_iou=threshold,
                )
            rows[name].append(
                {
                    "image_id": image_id,
                    **_metric_row(items, ground_truth, evaluator),
                }
            )
    summaries = {name: _summary(values) for name, values in rows.items()}
    eligible = {
        name: values
        for name, values in summaries.items()
        if name != BASELINE
        and values["room_f1"] >= summaries[BASELINE]["room_f1"]
        and values["room_iou"] >= summaries[BASELINE]["room_iou"]
        and values["corner_f1"] >= summaries[BASELINE]["corner_f1"]
    }
    selected = max(
        eligible or {BASELINE: summaries[BASELINE]},
        key=lambda name: summaries[name]["room_f1"],
    )
    report = {
        "status": "complete",
        "scope": "Raster2Seq val503 deterministic polygon ensemble",
        "selection_split": "val",
        "official_test509_used": False,
        "configs": {
            name: {"metrics": values}
            for name, values in summaries.items()
        },
        "eligible": sorted(eligible.values()),
        "selected": selected,
        "case_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "selected": selected,
                "metrics": summaries[selected],
                "eligible": {
                    name: summaries[name] for name in eligible
                },
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
