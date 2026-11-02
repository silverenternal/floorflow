#!/usr/bin/env python3
"""Diagnose room-identity and count ceilings on the native raster control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_raster_predicted_control_bridge import (
    RasterPredictedControlDataset,
)

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.raster_control_bank import load_raster_room_model
from floorflow.inference.instance_semantics import (
    consolidate_room_instances,
    enforce_room_instance_count,
    merge_room_instances_to_count,
)
from floorflow.representation.core import extract_room_instances

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/room_identity_diagnostic_val64_v3.json",
    )
    return parser.parse_args()


def _case_metrics(
    labels: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    target_instances = extract_room_instances(target, taxonomy, min_area=8)
    suite = room_metric_suite(
        labels,
        target,
        footprint,
        taxonomy,
        predicted_room_support=np.isin(labels, room_ids),
        predicted_instance_support=np.isin(labels, room_ids),
        target_room_support=np.isin(target, room_ids),
    )
    instance = suite["matched_instance"]
    return {
        "room_macro_iou": float(suite["visible_semantic"]["room_macro_iou"]),
        "instance_f1": float(instance["instance_f1_at_threshold"]),
        "matched_geometric_iou": float(instance["matched_geometric_iou"]),
        "type_accuracy": float(instance["matched_type_accuracy"]),
        "class_constrained_iou": float(
            instance["matched_class_constrained_iou"]
        ),
        "predicted_count": float(instance["predicted_instances"]),
        "target_count": float(len(target_instances)),
        "count_abs_error": float(
            abs(int(instance["predicted_instances"]) - len(target_instances))
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    dataset = RasterPredictedControlDataset(
        ROOT / "outputs/control_contracts/model_predicted_full_v1",
        "val",
        args.cache_root,
        args.data_root,
        limit=args.limit,
        offset=args.offset,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    model = load_raster_room_model(args.room_checkpoint, taxonomy, device)
    arms = (
        "baseline",
        "consolidated",
        "oracle_count_adjacent_merge",
        "oracle_count_global_merge",
    )
    rows = {arm: [] for arm in arms}
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    for target, footprint, _floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        logits = F.interpolate(
            model(footprint, control_image=image),
            size=(args.room_size, args.room_size),
            mode="bilinear",
            align_corners=False,
        )
        footprint_room = F.interpolate(
            footprint[:, None].float(),
            size=(args.room_size, args.room_size),
            mode="nearest",
        )[:, 0].long()
        labels = room_ids[logits.argmax(dim=1)] * footprint_room
        consolidated = consolidate_room_instances(
            labels,
            logits,
            footprint_room,
            taxonomy.room_ids,
        )
        for index, name in enumerate(names):
            target_np = target[index].cpu().numpy()
            target_count = len(
                extract_room_instances(target_np, taxonomy, min_area=8)
            )
            merged = merge_room_instances_to_count(
                labels[index : index + 1],
                logits[index : index + 1],
                footprint_room[index : index + 1],
                taxonomy.room_ids,
                target_count,
            )[0]
            adjacent_merged = enforce_room_instance_count(
                labels[index : index + 1],
                logits[index : index + 1],
                footprint_room[index : index + 1],
                taxonomy.room_ids,
                target_count,
            )[0]
            for arm, values in (
                ("baseline", labels[index]),
                ("consolidated", consolidated[index]),
                ("oracle_count_adjacent_merge", adjacent_merged),
                ("oracle_count_global_merge", merged),
            ):
                rows[arm].append(
                    {
                        "case": str(name),
                        **_case_metrics(
                            F.interpolate(
                                values[None, None].float(),
                                size=target.shape[-2:],
                                mode="nearest",
                            )[0, 0].long().cpu().numpy(),
                            target_np,
                            footprint[index].cpu().numpy(),
                            taxonomy,
                        ),
                    }
                )
    report = {
        "schema_version": "raster-room-identity-diagnostic-v3",
        "scope": "validation mechanism diagnostic",
        "test_opened": False,
        "target_used_at_inference": False,
        "oracle_information": ["oracle_count_merge uses target room count only"],
        "validation_cases": len(dataset),
        "validation_offset": args.offset,
        "summary": {
            arm: {
                key: float(np.mean([row[key] for row in arm_rows]))
                for key in arm_rows[0]
                if key != "case"
            }
            for arm, arm_rows in rows.items()
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
