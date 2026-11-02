#!/usr/bin/env python3
"""Calibrate global structured-overlay thresholds on validation data."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.objectives.condition_objective import partition_room_labels, room_macro_iou_from_labels
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.heads.structured_objects import (
    compose_structured_objects,
    constrain_structured_objects,
    object_group_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.85, 0.95])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, taxonomy, _ = load_checkpoint_model(args.checkpoint, device)
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    candidates = list(
        itertools.product(args.thresholds, args.thresholds, (False, True))
    )
    rows = {candidate: [] for candidate in candidates}
    partition_rows = []
    with torch.no_grad():
        for target, footprint, floor_ids, _names in loader:
            target = target.to(device)
            footprint = footprint.to(device)
            zero = torch.zeros(
                target.shape[0],
                model.target_channels,
                *target.shape[-2:],
                device=device,
            )
            _, aux = model.forward_with_aux(
                zero,
                torch.zeros(target.shape[0], device=device),
                footprint,
                floor_ids=floor_ids.to(device),
            )
            room_labels = partition_room_labels(aux["room_partition_logits"], taxonomy) * footprint.long()
            for index in range(target.shape[0]):
                partition_rows.append(
                    {
                        "room_macro_iou": room_macro_iou_from_labels(
                            room_labels[index : index + 1],
                            target[index : index + 1],
                            taxonomy,
                        )
                    }
                )
            for candidate, row_list in rows.items():
                opening_threshold, furniture_threshold, predicted_walls = candidate
                objects = object_group_predictions(
                    aux["structured_objects"],
                    taxonomy,
                    opening_threshold=opening_threshold,
                    furniture_threshold=furniture_threshold,
                    score_scale="probability",
                )
                objects = constrain_structured_objects(
                    objects,
                    room_labels,
                    footprint,
                    taxonomy,
                    use_predicted_walls=predicted_walls,
                )
                prediction = compose_structured_objects(
                    room_labels,
                    objects,
                    footprint,
                    taxonomy,
                )
                for index in range(target.shape[0]):
                    row_list.append(
                        floorplan_metrics(
                            prediction[index : index + 1],
                            target[index : index + 1],
                            footprint[index : index + 1],
                            taxonomy,
                        )
                    )
    partition_iou = _mean(partition_rows, "room_macro_iou")
    records = []
    for (opening_threshold, furniture_threshold, predicted_walls), row_list in rows.items():
        record = {
            "opening_threshold": opening_threshold,
            "furniture_threshold": furniture_threshold,
            "use_predicted_walls": predicted_walls,
            "room_macro_iou": _mean(row_list, "room_macro_iou"),
            "room_group_macro_iou": _mean(row_list, "room_group_macro_iou"),
            "room_area_coverage": _mean(row_list, "room_area_coverage"),
            "opening_f1": _mean(row_list, "opening_f1"),
            "furniture_iou": _mean(row_list, "furniture_iou"),
            "wall_iou": _mean(row_list, "wall_iou"),
            "interior_wall_iou": _mean(row_list, "interior_wall_iou"),
        }
        record["eligible"] = bool(
            record["room_macro_iou"] >= partition_iou - 0.01
            and record["room_area_coverage"] >= 0.80
        )
        record["selection_score"] = (
            0.55 * record["room_macro_iou"]
            + 0.20 * record["opening_f1"]
            + 0.20 * record["furniture_iou"]
            + 0.05 * record["wall_iou"]
        )
        records.append(record)
    eligible = [record for record in records if record["eligible"] and record["opening_f1"] > 0.0]
    if not eligible:
        raise RuntimeError("no structured-overlay candidate satisfies the room-preservation guard")
    selected = max(eligible, key=lambda record: record["selection_score"])
    report = {
        "claim_role": "validation_only_global_overlay_calibration",
        "split": args.split,
        "cases": len(dataset),
        "partition_only_room_macro_iou": partition_iou,
        "selection_policy": "room IoU within 0.01 of partition-only, coverage >= 0.80, then fixed weighted score",
        "selected": selected,
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
