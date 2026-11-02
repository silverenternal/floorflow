#!/usr/bin/env python3
"""Evaluate a raster instance-control checkpoint on a fixed room split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.data.predicted_control_raster import (
    DualControlRasterDataset,
    PredictedControlRasterDataset,
)
from floorflow.heads.raster_control_bank_loading import (
    load_raster_instance_control_model,
)
from floorflow.training.raster_instance_evaluation import evaluate
from floorflow.training.room_sampling import prepare_raster_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--page-floor-crop", action="store_true")
    parser.add_argument("--query-count-gate", action="store_true")
    parser.add_argument("--control-raster-root", type=Path)
    parser.add_argument("--dual-control-raster", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset_class = CubiCasaRasterControlDataset
    kwargs = {"image_mode": "floor" if args.page_floor_crop else "native"}
    if args.control_raster_root is not None:
        dataset_class = (
            DualControlRasterDataset
            if args.dual_control_raster
            else PredictedControlRasterDataset
        )
        kwargs = {
            "control_root": args.control_raster_root,
            "page_floor_crop": args.page_floor_crop,
        }
    dataset = dataset_class(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
        limit=None,
        **kwargs,
    )
    removed = prepare_raster_dataset(
        dataset,
        limit=args.limit,
        offset=args.offset,
    )
    if removed:
        print(f"removed empty validation targets: {removed}", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    model = load_raster_instance_control_model(
        args.checkpoint,
        dataset.taxonomy,
        device,
        query_count_gate=(True if args.query_count_gate else None),
    )
    case_rows = evaluate(
        model,
        loader,
        dataset.taxonomy,
        device,
        (0.0, 0.25, 0.5, 0.75, 1.0),
        return_case_rows=True,
    )
    metrics = {
        weight: {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in rows[0]
            if key != "case"
        }
        for weight, rows in case_rows.items()
    }
    report = {
        "schema_version": "raster-instance-control-evaluation-v1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "page_floor_crop": args.page_floor_crop,
        "query_count_gate": args.query_count_gate,
        "metrics": {str(key): value for key, value in metrics.items()},
        "case_rows": {
            str(key): rows for key, rows in case_rows.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "checkpoint": report["checkpoint"],
                "split": report["split"],
                "offset": report["offset"],
                "limit": report["limit"],
                "metrics": report["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
