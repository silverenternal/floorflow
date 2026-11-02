"""Validation-only confidence sweep for the mask-partition readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    collate_rplan_items,
    load_rplan_rows,
)
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.rplan_mask_partition import (
    mask_partition_query_metrics,
)
from floorflow.model.unified_floorplan_generator import (
    UnifiedFloorplanGenerator,
    UnifiedGeneratorConfig,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT / ".data/external/CE2EPlan/dataset/dataset_json/data_val.json"
)
DEFAULT_CHECKPOINTS = (
    "outputs/train/floorflow_resnet_mask_partition1024_b8_v1/best.pt",
    "outputs/train/floorflow_mask_partition1024_seed19_v1/best.pt",
    "outputs/train/floorflow_mask_partition1024_seed29_v1/best.pt",
    "outputs/train/floorflow_mask_partition1024_seed37_v1/best.pt",
    "outputs/train/floorflow_mask_partition1024_seed43_v1/best.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        dest="checkpoints",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--val-cases", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/rplan_mask_threshold_sweep_v1.json",
    )
    return parser.parse_args()


def _evaluate_checkpoint(
    checkpoint: Path,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    device = torch.device(args.device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    model = UnifiedFloorplanGenerator(
        taxonomy,
        UnifiedGeneratorConfig(**payload["config"]["model"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    limits = RPlanTensorLimits(
        **{
            key: int(value)
            for key, value in payload["config"]["tensors"].items()
        }
    )
    dataset = RPlanBoundaryDataset(
        args.data,
        taxonomy,
        rows=load_rplan_rows(
            args.data,
            limit=int(args.val_cases),
        ),
        limits=limits,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=collate_rplan_items,
        pin_memory=device.type == "cuda",
    )
    totals = {
        threshold: {
            key: 0.0
            for key in (
                "matched_geometric_iou",
                "instance_f1_at_threshold",
                "matched_type_accuracy",
                "count_abs_error",
            )
        }
        for threshold in args.thresholds
    }
    seen = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            output = model(
                batch["boundary_mask"],
                batch["boundary_valid"],
                batch["boundary_points"],
                batch["entrance_points"],
                batch["entrance_valid"],
            )
            size = batch["target_labels"].shape[0]
            for threshold in args.thresholds:
                metrics = mask_partition_query_metrics(
                    output,
                    batch["target_labels"],
                    batch["boundary_mask"],
                    taxonomy,
                    confidence_threshold=float(threshold),
                )
                for key, value in metrics.items():
                    totals[threshold][key] += float(value) * size
            seen += size
    return {
        str(threshold): {
            key: value / max(1, seen)
            for key, value in metrics.items()
        }
        for threshold, metrics in totals.items()
    }


def main() -> None:
    args = parse_args()
    if torch.device(args.device).type == "cuda" and args.device != "cuda:0":
        raise ValueError("RPLAN evaluation is restricted to cuda:0")
    checkpoints = [
        Path(value)
        for value in (args.checkpoints or DEFAULT_CHECKPOINTS)
    ]
    result = {}
    for checkpoint in checkpoints:
        result[str(checkpoint)] = _evaluate_checkpoint(checkpoint, args)
    aggregate = {}
    for threshold in args.thresholds:
        key = str(threshold)
        metrics = {
            metric: float(
                np.mean(
                    [
                        row[key][metric]
                        for row in result.values()
                    ]
                )
            )
            for metric in (
                "matched_geometric_iou",
                "instance_f1_at_threshold",
                "matched_type_accuracy",
                "count_abs_error",
            )
        }
        metrics["selection_score"] = (
            metrics["matched_geometric_iou"]
            + 0.25 * metrics["instance_f1_at_threshold"]
        )
        aggregate[key] = metrics
    selected = max(
        aggregate,
        key=lambda value: aggregate[value]["selection_score"],
    )
    report = {
        "schema": "rplan-mask-threshold-sweep-v1",
        "validation_file": str(args.data),
        "validation_cases": int(args.val_cases),
        "test_opened": False,
        "checkpoints": result,
        "aggregate": aggregate,
        "selected_threshold": float(selected),
        "threshold_selection_metric": (
            "mean Room IoU + 0.25 * mean instance F1"
        ),
        "claim_boundary": (
            "Threshold selection uses the declared validation prefix only."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_threshold": float(selected), **aggregate[selected]}))


if __name__ == "__main__":
    main()
