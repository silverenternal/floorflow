#!/usr/bin/env python3
"""Calibrate detection-style opening readout on a validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.geometry.opening_readout import (
    detections_from_labels,
    opening_detection_metrics,
    read_group_opening_detections,
    read_opening_detections,
    read_wall_interval_detections,
)
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=191)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.3, 0.5, 0.7])
    parser.add_argument("--wall-checkpoint", type=Path, default=None)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        args.device,
    )
    model.eval()
    wall_model = None
    if args.wall_checkpoint is not None:
        wall_model, _wall_config, _taxonomy, _state = load_checkpoint_model(
            args.wall_checkpoint,
            args.device,
        )
        if not hasattr(wall_model, "geometry_heads"):
            raise ValueError("wall checkpoint has no geometry heads")
        wall_model.requires_grad_(False)
        wall_model.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    rows = []
    with torch.no_grad():
        for target, footprint, floor_ids, names in loader:
            target = target.to(args.device)
            footprint = footprint.to(args.device)
            zero = torch.zeros(
                target.shape[0],
                model.target_channels,
                *target.shape[-2:],
                device=args.device,
            )
            _, aux = model.forward_with_aux(
                zero,
                torch.zeros(target.shape[0], device=args.device),
                footprint,
                floor_ids=floor_ids.to(args.device),
            )
            objects = aux["structured_objects"]
            opening_scores = torch.cat(
                [objects["opening_background_logits"][:, None], objects["opening_logits"]],
                dim=1,
            ).softmax(dim=1)[:, 1:]
            opening_foreground = 1.0 - torch.cat(
                [objects["opening_background_logits"][:, None], objects["opening_logits"]],
                dim=1,
            ).softmax(dim=1)[:, 0]
            wall_mask = objects["wall_logits"].sigmoid() >= 0.5
            if wall_model is not None:
                wall_zero = torch.zeros(
                    target.shape[0],
                    wall_model.target_channels,
                    *target.shape[-2:],
                    device=args.device,
                )
                with torch.no_grad():
                    wall_features = wall_model._encode_features(
                        wall_zero,
                        torch.zeros(target.shape[0], device=args.device),
                        footprint,
                        floor_ids=floor_ids.to(args.device),
                    )
                    wall_centerline = wall_model.geometry_heads(wall_features)[
                        "wall_centerline_logits"
                    ].sigmoid()
                wall_mask = F.interpolate(
                    wall_centerline,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0] >= 0.5
            for index, name in enumerate(names):
                labels = np.zeros(target[index].shape, dtype=np.int64)
                labels[wall_mask[index].cpu().numpy()] = taxonomy.class_id("wall")
                wall_plan = raster_labels_to_wall_plan(labels, taxonomy)
                expected = detections_from_labels(
                    target[index].cpu().numpy(),
                    taxonomy,
                    wall_plan,
                )
                row = {"case": name}
                for threshold in args.thresholds:
                    predicted = read_opening_detections(
                        opening_scores[index].cpu().numpy(),
                        taxonomy.opening_ids,
                        taxonomy.class_names,
                        taxonomy,
                        threshold=threshold,
                        wall_plan=wall_plan,
                        min_area=args.min_area,
                    )
                    row[str(threshold)] = opening_detection_metrics(
                        predicted,
                        expected,
                        tolerance=args.tolerance,
                    )
                    row[str(threshold)]["wall_attached_fraction"] = (
                        sum(item.wall_index is not None for item in predicted)
                        / max(1, len(predicted))
                    )
                    group_predicted = read_group_opening_detections(
                        opening_foreground[index].cpu().numpy(),
                        objects["opening_logits"][index].cpu().numpy(),
                        taxonomy.opening_ids,
                        taxonomy.class_names,
                        taxonomy,
                        threshold=threshold,
                        wall_plan=wall_plan,
                        min_area=args.min_area,
                    )
                    row[f"group_{threshold}"] = opening_detection_metrics(
                        group_predicted,
                        expected,
                        tolerance=args.tolerance,
                    )
                    row[f"group_{threshold}"]["wall_attached_fraction"] = (
                        sum(item.wall_index is not None for item in group_predicted)
                        / max(1, len(group_predicted))
                    )
                    interval_predicted = read_wall_interval_detections(
                        opening_foreground[index].cpu().numpy(),
                        objects["opening_logits"][index].cpu().numpy(),
                        taxonomy.opening_ids,
                        taxonomy.class_names,
                        wall_plan,
                        threshold=threshold,
                    )
                    row[f"interval_{threshold}"] = opening_detection_metrics(
                        interval_predicted,
                        expected,
                        tolerance=args.tolerance,
                    )
                rows.append(row)
    metrics = {
        str(threshold): {
            key: float(
                np.mean([row[str(threshold)][key] for row in rows])
            )
            for key in ("precision", "recall", "f1", "wall_attached_fraction")
        }
        for threshold in args.thresholds
    }
    metrics.update(
        {
            f"group_{threshold}": {
                key: float(
                    np.mean([row[f"group_{threshold}"][key] for row in rows])
                )
                for key in ("precision", "recall", "f1", "wall_attached_fraction")
            }
            for threshold in args.thresholds
        }
    )
    metrics.update(
        {
            f"interval_{threshold}": {
                key: float(
                    np.mean([row[f"interval_{threshold}"][key] for row in rows])
                )
                for key in ("precision", "recall", "f1")
            }
            for threshold in args.thresholds
        }
    )
    report = {
        "case_count": len(rows),
        "tolerance": args.tolerance,
        "metrics": metrics,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
