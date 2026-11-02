#!/usr/bin/env python3
"""Compare dense room stage, query instances, and refined room readouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.room_mask_fusion import render_room_masks
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/train/room_capability_fast64_v1/best.pt",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/room_capability_fast64_readout_diagnosis_v1.json",
    )
    return parser.parse_args()


def _metrics(prediction, target, footprint, taxonomy):
    room_ids = torch.tensor(taxonomy.room_ids, device=prediction.device)
    prediction_support = torch.isin(prediction, room_ids)
    target_support = torch.isin(target, room_ids)
    result = room_metric_suite(
        prediction.detach().cpu().numpy(),
        target.detach().cpu().numpy(),
        footprint.detach().cpu().numpy(),
        taxonomy,
        predicted_room_support=prediction_support.detach().cpu().numpy(),
        predicted_instance_support=prediction_support.detach().cpu().numpy(),
        target_room_support=target_support.detach().cpu().numpy(),
    )
    instance = result["matched_instance"]
    return {
        "room_iou": float(result["visible_semantic"]["room_macro_iou"]),
        "instance_f1": float(instance["instance_f1_at_threshold"]),
        "type_accuracy": float(instance["matched_type_accuracy"]),
        "count_error": float(instance["count_abs_error"]),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    model = build_predicted_channel_model(taxonomy, config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    runtime = CadStageRuntime(
        taxonomy,
        device,
        room_checkpoint=config["room_checkpoint"],
        structure_checkpoint=config["structure_checkpoint"],
        centerline_checkpoint=config["centerline_checkpoint"],
        opening_checkpoint=config["opening_checkpoint"],
        room_blend=float(config["room_blend"]),
        structure_size=128,
    )
    dataset = RasterPredictedControlDataset(
        Path(config["predicted_root"]),
        "val",
        Path(config["cache_root"]),
        Path(config["data_root"]),
        image_size=int(config["image_size"]),
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    rows = {"dense": [], "query": [], "refined": []}
    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        controls = _predicted_controls(
            Path(config["predicted_root"]),
            "val",
            names,
            args,
            device,
        )
        controls, _ = runtime(
            controls,
            footprint,
            image.to(device),
            floor_ids.to(device),
        )
        output = model(controls, footprint)
        size = target.shape[-2:]
        dense_logits = torch.nn.functional.interpolate(
            output["stage_rooms_logits"],
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        refined_logits = torch.nn.functional.interpolate(
            output["room_instance_logits"],
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        query_prediction = {
            "class_logits": output["room_instance_class_logits"],
            "mask_logits": torch.nn.functional.interpolate(
                output["room_instance_mask_logits"],
                size=size,
                mode="bilinear",
                align_corners=False,
            ),
        }
        query_labels = render_room_masks(
            query_prediction,
            taxonomy,
            footprint,
        )
        dense_labels = room_ids[dense_logits.argmax(dim=1)] * footprint.long()
        refined_labels = room_ids[refined_logits.argmax(dim=1)] * footprint.long()
        for index in range(target.shape[0]):
            rows["dense"].append(
                _metrics(
                    dense_labels[index],
                    target[index],
                    footprint[index],
                    taxonomy,
                )
            )
            rows["query"].append(
                _metrics(
                    query_labels[index],
                    target[index],
                    footprint[index],
                    taxonomy,
                )
            )
            rows["refined"].append(
                _metrics(
                    refined_labels[index],
                    target[index],
                    footprint[index],
                    taxonomy,
                )
            )
    summary = {
        name: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
        }
        for name, values in rows.items()
    }
    report = {
        "schema_version": "room-capability-readout-diagnosis-v1",
        "checkpoint": str(args.checkpoint),
        "cases": len(rows["dense"]),
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
