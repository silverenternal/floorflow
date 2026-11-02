#!/usr/bin/env python3
"""Evaluate junction, centerline, opening, and room-boundary heatmaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from scipy.ndimage import maximum_filter
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.geometry_heads import (
    HEATMAP_KEYS,
    build_geometry_targets,
    opening_center_mask,
)
from floorflow.heads.structured_objects import object_group_predictions
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=191)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _prf(prediction: np.ndarray, target: np.ndarray, tolerance: int) -> dict[str, float]:
    if not prediction.any() and not target.any():
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    precision = (
        np.logical_and(prediction, binary_dilation(target, structure)).sum()
        / max(1, prediction.sum())
    )
    recall = (
        np.logical_and(target, binary_dilation(prediction, structure)).sum()
        / max(1, target.sum())
    )
    f1 = 2.0 * precision * recall / max(1.0e-8, precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _postprocess(key: str, probability: np.ndarray, threshold: float) -> np.ndarray:
    mask = probability >= float(threshold)
    if key in {"junction", "opening_center"}:
        return mask & (probability >= maximum_filter(probability, size=3))
    if key in {"wall_centerline", "room_boundary"}:
        return skeletonize(mask)
    return mask


def main() -> None:
    args = parse_args()
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        args.device,
    )
    model.eval()
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
            raw_predictions = {
                key: aux["geometry"][f"{key}_logits"].sigmoid()
                for key in HEATMAP_KEYS
            }
            object_openings = object_group_predictions(
                aux["structured_objects"],
                taxonomy,
                opening_threshold=0.5,
                score_scale="probability",
            )["opening"]
            targets = build_geometry_targets(target, footprint, taxonomy)
            for index, name in enumerate(names):
                row = {
                    "case": name,
                    "target_wall_pixels": int(
                        (target[index] == taxonomy.class_id("wall")).sum().item()
                    ),
                }
                case_predictions = {
                    key: _postprocess(
                        key,
                        raw_predictions[key][index, 0].detach().cpu().numpy(),
                        args.threshold,
                    )
                    for key in HEATMAP_KEYS
                }
                for key in HEATMAP_KEYS:
                    row[key] = _prf(
                        case_predictions[key],
                        targets[key][index, 0].cpu().numpy() > 0.5,
                        args.tolerance,
                    )
                row["object_opening_center"] = _prf(
                    opening_center_mask(
                        object_openings[index].cpu().numpy(),
                        taxonomy,
                    ),
                    targets["opening_center"][index, 0].cpu().numpy() > 0.5,
                    args.tolerance,
                )
                labels = np.zeros(target[index].shape, dtype=np.int64)
                labels[
                    case_predictions["wall_centerline"]
                ] = taxonomy.class_id("wall")
                plan = raster_labels_to_wall_plan(labels, taxonomy)
                row["readout_walls"] = len(plan.walls)
                row["readout_rooms"] = len(plan.rooms)
                rows.append(row)
    summary = {
        key: {
            metric: float(np.mean([row[key][metric] for row in rows]))
            for metric in ("precision", "recall", "f1")
        }
        for key in HEATMAP_KEYS
    }
    summary["readout_room_fraction"] = float(
        np.mean([row["readout_rooms"] > 0 for row in rows])
    )
    summary["object_opening_center"] = {
        metric: float(
            np.mean(
                [row["object_opening_center"][metric] for row in rows]
            )
        )
        for metric in ("precision", "recall", "f1")
    }
    median_walls = float(np.median([row["target_wall_pixels"] for row in rows]))
    for label, predicate in (
        ("small_plan", lambda value: value <= median_walls),
        ("large_plan", lambda value: value > median_walls),
    ):
        group = [
            row for row in rows if predicate(row["target_wall_pixels"])
        ]
        summary[label] = {
            "cases": len(group),
            "wall_centerline_f1": float(
                np.mean([row["wall_centerline"]["f1"] for row in group])
            ),
            "readout_room_fraction": float(
                np.mean([row["readout_rooms"] > 0 for row in group])
            ),
        }
    report = {
        "case_count": len(rows),
        "threshold": args.threshold,
        "tolerance_pixels": args.tolerance,
        "metrics": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
