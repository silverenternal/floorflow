#!/usr/bin/env python3
"""Evaluate second-stage furniture refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.heads.coarse_furniture_refiner import (
    CoarseFurnitureRefiner,
    coarse_furniture_control,
)
from floorflow.heads.structured_objects import object_group_metrics


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
    parser.add_argument("--control-size", type=int, default=192)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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
    model = CoarseFurnitureRefiner(
        len(taxonomy.room_ids) + len(taxonomy.furniture_ids) + 1,
        len(taxonomy.furniture_ids),
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
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=device)
    rows: list[dict[str, float]] = []
    raster_dir = args.output_dir / "rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)
    for target, footprint, _floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        control = coarse_furniture_control(
            target,
            footprint,
            taxonomy,
            size=args.control_size,
        )
        output = model(control, output_size=target.shape[-2:])
        probability = F.interpolate(
            output["foreground_logits"].sigmoid(),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        class_logits = F.interpolate(
            output["class_logits"],
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        furniture = torch.where(
            probability >= args.threshold,
            furniture_ids[class_logits.argmax(dim=1)],
            torch.zeros_like(target),
        ) * footprint.long()
        for index, name in enumerate(names):
            predictions = {
                "furniture": furniture[index : index + 1],
                "opening": torch.zeros_like(furniture[index : index + 1]),
                "wall": torch.zeros_like(furniture[index : index + 1]),
            }
            metrics = object_group_metrics(
                predictions,
                target[index : index + 1],
                taxonomy,
            )
            overlap = furniture[index].bool() & torch.isin(
                target[index],
                furniture_ids,
            )
            if overlap.any():
                metrics["furniture_type_accuracy"] = float(
                    (
                        furniture[index][overlap]
                        == target[index][overlap]
                    ).float().mean()
                )
            else:
                metrics["furniture_type_accuracy"] = 0.0
            rows.append(metrics)
            path = raster_dir / f"{name.replace('/', '__')}.npz"
            np.savez_compressed(
                path,
                prediction=furniture[index].cpu().numpy().astype(np.uint8),
                target=target[index].cpu().numpy().astype(np.uint8),
                footprint=footprint[index].cpu().numpy().astype(np.uint8),
            )
        print(f"evaluated {len(rows)}/{len(dataset)}", flush=True)
    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
    report = {
        "cases": len(rows),
        "task_variant": "contour + room program + coarse CAD layout",
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
