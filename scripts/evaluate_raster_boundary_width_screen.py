#!/usr/bin/env python3
"""Screen known-exterior wall widths for raster reconstruction topology."""

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
    _structure_outputs,
)
from torch.utils.data import DataLoader
from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.geometry.core import compose_known_exterior
from floorflow.heads.coarse_structure_refiner import coarse_structure_control
from floorflow.heads.joint_control_repair import JointControlRepair
from floorflow.heads.raster_control_bank import load_raster_room_model
from floorflow.inference.room_polygon_readout import vectorize_room_masks
from floorflow.inference.room_repair import apply_floorplan_repairs

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
    parser.add_argument("--widths", type=int, nargs="+", default=(3, 5, 7, 9))
    parser.add_argument("--repair-rooms", action="store_true")
    parser.add_argument("--parsable-fallback", action="store_true")
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
        default=ROOT / "outputs/reports/raster_boundary_width_calibration64_v1.json",
    )
    return parser.parse_args()


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
        offset=args.offset,
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
    structure_model = JointControlRepair(
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        room_base_channels=args.room_base_channels,
        structure_base_channels=args.structure_base_channels,
    ).to(device)
    payload = torch.load(
        args.structure_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    structure_model.load_state_dict(payload["model"])
    structure_model.eval()
    widths = tuple(int(value) for value in args.widths)
    arms = {str(width): [] for width in widths}
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
            room_logits = F.interpolate(
                room_model(footprint, control_image=image),
                size=(args.room_size, args.room_size),
                mode="bilinear",
                align_corners=False,
            )
            output = _structure_outputs(
                structure_model,
                predicted,
                room_logits,
                footprint,
                args,
                repair_room=False,
            )
            layers, _rooms, _wall, _opening = _decode_layers(
                room_logits,
                output["structure"],
                footprint,
                taxonomy,
            )
            visible = layers.visible_labels()
            if args.repair_rooms:
                visible = torch.stack(
                    [
                        torch.from_numpy(
                            apply_floorplan_repairs(
                                row.cpu().numpy(),
                                mask.cpu().numpy().astype(np.uint8),
                                taxonomy,
                            )
                        ).to(device)
                        for row, mask in zip(
                            visible,
                            footprint,
                            strict=True,
                        )
                    ]
                )
            clean = coarse_structure_control(
                target,
                footprint,
                taxonomy,
                size=args.structure_size,
            )
            for width in widths:
                labels = compose_known_exterior(
                    visible,
                    footprint,
                    boundary_width=width,
                    opening_ids=taxonomy.opening_ids,
                )
                for index, name in enumerate(names):
                    case_labels = labels[index]
                    if args.parsable_fallback:
                        structural = structural_legality_metrics(
                            case_labels.cpu().numpy(),
                            case_labels.cpu().numpy(),
                            taxonomy,
                        )
                        if structural["parsable"] < 1.0:
                            case_labels = torch.from_numpy(
                                vectorize_room_masks(
                                    case_labels.cpu().numpy(),
                                    footprint[index].cpu().numpy().astype(bool),
                                    taxonomy,
                                    epsilon=1.5,
                                )
                            ).to(device)
                    arms[str(width)].append(
                        {
                            "case": str(name),
                            **_metric_row(
                                case_labels,
                                target[index],
                                footprint[index],
                                taxonomy,
                                clean[index],
                                room_count,
                            ),
                        }
                    )
    summary = {
        width: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if key != "case"
        }
        for width, rows in arms.items()
    }
    report = {
        "schema_version": "raster-boundary-width-screen-v1",
        "scope": "mechanism screening only",
        "validation_cases": len(dataset),
        "validation_offset": args.offset,
        "test_opened": False,
        "target_used_at_inference": False,
        "summary": summary,
        "cases": arms[str(widths[0])] if len(widths) == 1 else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
