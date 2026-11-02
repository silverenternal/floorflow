#!/usr/bin/env python3
"""Evaluate coarse-layout wall/opening refinement with fpeval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.heads.coarse_structure_refiner import (
    CoarseStructureRefiner,
    coarse_structure_control,
)
from floorflow.heads.geometry_heads import structural_candidate_mask
from floorflow.representation.layered_floorplan import LayeredFloorplan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--control-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wall-threshold", type=float, default=0.5)
    parser.add_argument("--opening-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(set().union(*(row.keys() for row in rows)))
    return {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
        if any(key in row for row in rows)
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    taxonomy = dataset.taxonomy
    model = CoarseStructureRefiner(
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
        len(taxonomy.opening_ids),
        base_channels=int(payload["config"].get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    rows: list[dict[str, float]] = []
    raster_dir = args.output_dir / "rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    for target, footprint, _floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        control = coarse_structure_control(
            target,
            footprint,
            taxonomy,
            size=args.control_size,
        )
        output = model(control)
        wall_probability = (
            F.interpolate(
                output["wall_logits"].sigmoid(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        )
        opening_probability = (
            F.interpolate(
                output["opening_logits"].sigmoid(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        )
        type_logits = F.interpolate(
            output["opening_type_logits"],
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        room_control = control[:, : len(taxonomy.room_ids)]
        room_index = F.interpolate(
            room_control,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        room_labels = torch.zeros_like(target)
        for index, class_id in enumerate(taxonomy.room_ids):
            room_labels = torch.where(
                room_index == index,
                torch.full_like(room_labels, class_id),
                room_labels,
            )
        room_labels = room_labels * footprint.long()
        candidates = structural_candidate_mask(
            room_labels,
            footprint,
            taxonomy,
            radius=4,
        )
        opening_labels = room_ids.new_tensor(taxonomy.opening_ids)[
            type_logits.argmax(dim=1)
        ]
        opening_labels = torch.where(
            (opening_probability >= args.opening_threshold) & candidates,
            opening_labels,
            torch.zeros_like(opening_labels),
        )
        wall = (
            wall_probability >= args.wall_threshold
        ) & footprint.bool() & (opening_labels == 0)
        layers = LayeredFloorplan(
            room_labels=room_labels,
            wall=wall.long(),
            opening=opening_labels.long(),
            furniture=torch.zeros_like(target),
            footprint=footprint,
            taxonomy=taxonomy,
        )
        prediction = layers.visible_labels()
        for index, name in enumerate(names):
            metrics = floorplan_metrics(
                prediction[index : index + 1],
                target[index : index + 1],
                footprint[index : index + 1],
                taxonomy,
                room_support=room_labels[index : index + 1],
            )
            fpeval = fpeval_raster_metrics(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
                tolerance_fractions=(0.05,),
            )
            summary = fpeval["summary"]["0.05"]
            metrics.update(
                {
                    "fpeval_wall_f1": summary["wall_f1"],
                    "fpeval_opening_f1": summary["opening_f1"],
                    "fpeval_room_f1": summary["room_f1"],
                    "fpeval_validity": summary["validity"],
                    "fpeval_watertight": float(
                        fpeval["prediction_validity"]["watertight"]
                    ),
                }
            )
            rows.append(metrics)
            path = raster_dir / f"{name.replace('/', '__')}.npz"
            np.savez_compressed(
                path,
                prediction=prediction[index].cpu().numpy().astype(np.uint8),
                target=target[index].cpu().numpy().astype(np.uint8),
                footprint=footprint[index].cpu().numpy().astype(np.uint8),
                room_labels=room_labels[index].cpu().numpy().astype(np.uint8),
            )
        print(f"evaluated {len(rows)}/{len(dataset)}", flush=True)
    metrics = aggregate(rows)
    report = {
        "cases": len(rows),
        "task_variant": "contour + room program + coarse room/structure layout",
        "best_of_n_used": False,
        "metrics": metrics,
        "checkpoint": str(args.checkpoint),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "per_case.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
