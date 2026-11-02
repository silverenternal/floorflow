#!/usr/bin/env python3
"""Re-decode room identity from predicted non-room structure controls."""

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

from floorflow.data.predicted_control_raster import (
    predicted_furniture_probability,
)
from floorflow.data.raster_control import image_normalization, raster_control_image
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.core import compose_known_exterior
from floorflow.heads.coarse_structure_refiner import coarse_structure_control
from floorflow.heads.joint_control_repair import JointControlRepair
from floorflow.heads.raster_control_bank import load_raster_room_model
from floorflow.representation.layered_floorplan import LayeredFloorplan

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v1",
    )
    parser.add_argument(
        "--native-room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument(
        "--clean-room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_full_v1/best.pt",
    )
    parser.add_argument(
        "--structure-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_structure_bridge_full_v1/best.pt",
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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--room-base-channels", type=int, default=24)
    parser.add_argument("--structure-base-channels", type=int, default=32)
    parser.add_argument("--boundary-width", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster_control_reprojection_calibration64_v1.json",
    )
    return parser.parse_args()


def reprojected_control(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy,
    furniture: np.ndarray | None,
) -> torch.Tensor:
    image = raster_control_image(labels, footprint, taxonomy)
    if furniture is not None and furniture.shape == footprint.shape:
        image[1] = np.where(furniture > 0.5, 0.55, 1.0).astype(np.float32)
    return torch.from_numpy(image_normalization(image)).float()


def apply_topology(
    room_labels: torch.Tensor,
    wall: torch.Tensor,
    opening: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy,
    boundary_width: int,
) -> torch.Tensor:
    layers = LayeredFloorplan(
        room_labels=room_labels,
        wall=wall.long(),
        opening=opening,
        furniture=torch.zeros_like(room_labels),
        footprint=footprint,
        taxonomy=taxonomy,
    )
    return compose_known_exterior(
        layers.visible_labels(),
        footprint,
        boundary_width=boundary_width,
        opening_ids=taxonomy.opening_ids,
    )


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
    native_room = load_raster_room_model(
        args.native_room_checkpoint,
        taxonomy,
        device,
    )
    clean_room = load_raster_room_model(
        args.clean_room_checkpoint,
        taxonomy,
        device,
    )
    structure = JointControlRepair(
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        room_base_channels=args.room_base_channels,
        structure_base_channels=args.structure_base_channels,
    ).to(device)
    structure_payload = torch.load(
        args.structure_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    structure.load_state_dict(structure_payload["model"])
    structure.eval()
    blends = (0.0, 0.25, 0.5, 0.75, 1.0)
    arms = {str(weight): [] for weight in blends}
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
            native_logits = F.interpolate(
                native_room(footprint, control_image=image),
                size=(args.room_size, args.room_size),
                mode="bilinear",
                align_corners=False,
            )
            output = _structure_outputs(
                structure,
                predicted,
                native_logits,
                footprint,
                args,
                repair_room=False,
            )
            _native_layers, native_rooms, wall, opening = _decode_layers(
                native_logits,
                output["structure"],
                footprint,
                taxonomy,
            )
            native_visible = apply_topology(
                native_rooms,
                wall,
                opening,
                footprint,
                taxonomy,
                args.boundary_width,
            )
            furniture = F.interpolate(
                predicted_furniture_probability(
                    predicted["furniture"],
                    len(taxonomy.room_ids),
                )[:, None],
                size=footprint.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            reprojection = torch.stack(
                [
                    reprojected_control(
                        native_visible[index].cpu().numpy(),
                        footprint[index].cpu().numpy(),
                        taxonomy,
                        furniture[index].cpu().numpy(),
                    )
                    for index in range(footprint.shape[0])
                ]
            ).to(device)
            clean_logits = clean_room(
                footprint,
                control_image=reprojection,
            )
            native_probability = native_logits.softmax(dim=1)
            clean_probability = F.interpolate(
                clean_logits,
                size=native_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).softmax(dim=1)
            clean = coarse_structure_control(
                target,
                footprint,
                taxonomy,
                size=args.structure_size,
            )
            room_ids = torch.tensor(taxonomy.room_ids, device=device)
            for weight in blends:
                probability = (
                    (1.0 - weight) * native_probability
                    + weight * clean_probability
                )
                rooms = F.interpolate(
                    room_ids[probability.argmax(dim=1)][:, None].float(),
                    size=footprint.shape[-2:],
                    mode="nearest",
                )[:, 0].long()
                rooms *= footprint.long()
                labels = apply_topology(
                    rooms,
                    wall,
                    opening,
                    footprint,
                    taxonomy,
                    args.boundary_width,
                )
                for index, name in enumerate(names):
                    arms[str(weight)].append(
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
        weight: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if key != "case"
        }
        for weight, rows in arms.items()
    }
    report = {
        "schema_version": "raster-control-reprojection-screen-v1",
        "scope": "mechanism screening only",
        "validation_cases": len(dataset),
        "validation_offset": args.offset,
        "test_opened": False,
        "target_used_at_inference": False,
        "room_labels_in_control": False,
        "summary": summary,
        "cases": arms[str(blends[-1])] if len(blends) == 1 else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
