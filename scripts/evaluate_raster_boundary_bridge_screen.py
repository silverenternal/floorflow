#!/usr/bin/env python3
"""Evaluate boundary-refined room controls through the CAD bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from evaluate_raster_control_router_screen import (
    _decode_layers,
    _metric_row,
    _postprocess,
    _structure_outputs,
)
from torch.utils.data import DataLoader
from train_raster_predicted_control_bridge import (
    RasterPredictedControlDataset,
)
from train_raster_room_boundary_refiner import room_inputs
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.coarse_structure_refiner import coarse_structure_control
from floorflow.heads.joint_control_repair import JointControlRepair
from floorflow.heads.predicted_room_identity import PredictedRoomIdentityHead
from floorflow.heads.raster_control_bank import load_raster_room_model

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v1",
    )
    parser.add_argument(
        "--structure-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_structure_bridge_screen256_v1/best.pt",
    )
    parser.add_argument(
        "--joint-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_predicted_control_bridge_screen256_v1/best.pt",
    )
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument(
        "--boundary-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_boundary_refiner_screen256_v2/best.pt",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--room-base-channels", type=int, default=24)
    parser.add_argument("--structure-base-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster_boundary_bridge_screen256_v1.json",
    )
    return parser.parse_args()


def _load_bridge(path: Path, taxonomy, args) -> JointControlRepair:
    model = JointControlRepair(
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        room_base_channels=args.room_base_channels,
        structure_base_channels=args.structure_base_channels,
    ).to(args.device)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return model.eval()


def _load_boundary(path: Path, taxonomy, device) -> PredictedRoomIdentityHead:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = PredictedRoomIdentityHead(
        int(config["feature_channels"]),
        len(taxonomy.room_ids),
        hidden_channels=int(config["hidden_channels"]),
        blocks=int(config["blocks"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    dataset = RasterPredictedControlDataset(
        args.predicted_root,
        "val",
        args.cache_root,
        args.data_root,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    room_model = load_raster_room_model(
        args.room_checkpoint,
        taxonomy,
        device,
    )
    boundary = _load_boundary(args.boundary_checkpoint, taxonomy, device)
    structure_model = _load_bridge(args.structure_checkpoint, taxonomy, args)
    joint_model = _load_bridge(args.joint_checkpoint, taxonomy, args)
    arms = {
        "raw_structure/exterior": [],
        "boundary_structure/exterior": [],
        "boundary_structure/repair": [],
        "boundary_joint/exterior": [],
        "boundary_joint/repair": [],
    }
    room_count = len(taxonomy.room_ids)
    with torch.inference_mode():
        for target, footprint, _floor_ids, names, image in loader:
            target = target.to(device)
            footprint = footprint.to(device)
            image = image.to(device)
            predicted = _predicted_controls(
                args.predicted_root,
                "val",
                names,
                args,
                device,
            )
            features, raw_logits = room_inputs(
                room_model,
                footprint,
                image,
            )
            boundary_logits = boundary(features, raw_logits)["room_logits"]
            clean = coarse_structure_control(
                target,
                footprint,
                taxonomy,
                size=args.structure_size,
            )
            for prefix, logits, model, repair_room in (
                ("raw_structure", raw_logits, structure_model, False),
                ("boundary_structure", boundary_logits, structure_model, False),
                ("boundary_joint", boundary_logits, joint_model, True),
            ):
                room_logits = F.interpolate(
                    logits,
                    size=(args.room_size, args.room_size),
                    mode="bilinear",
                    align_corners=False,
                )
                output = _structure_outputs(
                    model,
                    predicted,
                    room_logits,
                    footprint,
                    args,
                    repair_room=repair_room,
                )
                layers, _rooms, _wall, _opening = _decode_layers(
                    output["room_logits_full"],
                    output["structure"],
                    footprint,
                    taxonomy,
                )
                variants = ("exterior",)
                if prefix != "raw_structure":
                    variants = ("exterior", "repair")
                for variant in variants:
                    labels = _postprocess(layers, variant, taxonomy)
                    for index, name in enumerate(names):
                        arms[f"{prefix}/{variant}"].append(
                            {
                                "case": str(name),
                                **_metric_row(
                                    labels[index],
                                    target[index],
                                    footprint[index],
                                    taxonomy,
                                    clean[index],
                                    room_count,
                                ),
                            }
                        )
    summary = {
        name: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if key != "case"
        }
        for name, rows in arms.items()
    }
    report = {
        "schema_version": "raster-boundary-bridge-screen-v1",
        "scope": "mechanism screening only",
        "validation_cases": len(dataset),
        "target_used_at_inference": False,
        "test_opened": False,
        "summary": summary,
        "promotion_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
