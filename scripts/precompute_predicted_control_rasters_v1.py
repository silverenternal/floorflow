#!/usr/bin/env python3
"""Cache room-decoder controls rendered from predicted non-room layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from evaluate_raster_control_reprojection import (
    apply_topology,
    reprojected_control,
)
from evaluate_raster_control_router_screen import (
    _decode_layers,
    _structure_outputs,
)
from torch.utils.data import DataLoader
from train_raster_predicted_control_bridge import (
    RasterPredictedControlDataset,
)
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.control_contract import sha256_file
from floorflow.data.predicted_control_raster import (
    predicted_control_raster_path,
    predicted_furniture_probability,
)
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.joint_control_loading import load_joint_control_repair
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
        "--data-root",
        type=Path,
        default=ROOT / ".data/cubicasa5k",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument(
        "--structure-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_structure_bridge_full_v1/best.pt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/cache/predicted_control_raster_v1",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--room-base-channels", type=int, default=24)
    parser.add_argument("--structure-base-channels", type=int, default=32)
    parser.add_argument("--boundary-width", type=int, default=7)
    parser.add_argument(
        "--include-predicted-furniture",
        action="store_true",
        help=(
            "Include the currently uncalibrated predicted furniture layer in "
            "the room-decoder control."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    dataset = RasterPredictedControlDataset(
        args.predicted_root,
        args.split,
        args.cache_root,
        args.data_root,
        image_size=args.image_size,
        limit=args.limit,
        offset=args.offset,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    room_model = load_raster_room_model(
        args.room_checkpoint,
        taxonomy,
        device,
    )
    structure_model, _structure_config = load_joint_control_repair(
        args.structure_checkpoint,
        taxonomy,
        device,
    )
    structure_model.eval()
    paths = []
    for target, footprint, _floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        predicted = _predicted_controls(
            args.predicted_root,
            args.split,
            names,
            args,
            device,
        )
        native_logits = F.interpolate(
            room_model(footprint, control_image=image),
            size=(args.room_size, args.room_size),
            mode="bilinear",
            align_corners=False,
        )
        output = _structure_outputs(
            structure_model,
            predicted,
            native_logits,
            footprint,
            args,
            repair_room=False,
        )
        _layers, native_rooms, wall, opening = _decode_layers(
            native_logits,
            output["structure"],
            footprint,
            taxonomy,
        )
        visible = apply_topology(
            native_rooms,
            wall,
            opening,
            footprint,
            taxonomy,
            args.boundary_width,
        )
        furniture = (
            F.interpolate(
                predicted_furniture_probability(
                    predicted["furniture"],
                    len(taxonomy.room_ids),
                )[:, None],
                size=footprint.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            if args.include_predicted_furniture
            else None
        )
        for index, name in enumerate(names):
            control = reprojected_control(
                visible[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                (
                    furniture[index].cpu().numpy()
                    if furniture is not None
                    else None
                ),
            )
            path = predicted_control_raster_path(
                args.output_root,
                args.split,
                str(name),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                image=control.numpy().astype(np.float16),
            )
            paths.append(path)
    manifest = {
        "schema_version": "predicted-control-raster-v1",
        "control_source": "predicted_control",
        "target_used_at_inference": False,
        "room_labels_in_control": False,
        "split": args.split,
        "offset": int(args.offset),
        "cases": len(dataset),
        "files": len(paths),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "checkpoints": {
            "room": {
                "path": str(args.room_checkpoint),
                "sha256": sha256_file(args.room_checkpoint),
            },
            "structure": {
                "path": str(args.structure_checkpoint),
                "sha256": sha256_file(args.structure_checkpoint),
            },
        },
    }
    manifest_path = args.output_root / args.split / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
