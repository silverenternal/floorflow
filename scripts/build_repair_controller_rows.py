#!/usr/bin/env python3
"""Build guarded repair acceptance rows from generated floorplans."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.geometry.core import compose_known_exterior
from floorflow.inference.core import semantic_endpoint_labels
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.inference.repair_features import repair_decision_features
from floorflow.inference.room_repair import apply_floorplan_repairs
from floorflow.inference.repair_acceptance import guarded_summary_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--split", default="val")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--use-fpeval-edit-cost", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/repair_controller/rows.csv")
    args = parser.parse_args()
    model, config, taxonomy, _ = load_checkpoint_model(args.checkpoint, args.device)
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    rows = []
    with torch.no_grad():
        for target, footprint, floor_ids, names in loader:
            target, footprint = target.to(args.device), footprint.to(args.device)
            labels = semantic_endpoint_labels(model, target, footprint, floor_ids.to(args.device))
            labels = compose_known_exterior(labels, footprint, opening_ids=taxonomy.opening_ids)
            for index, name in enumerate(names):
                raw = labels[index].cpu().numpy()
                footprint_np = footprint[index].cpu().numpy()
                repaired = apply_floorplan_repairs(raw, footprint_np.astype("uint8"), taxonomy)
                raw_metrics = floorplan_metrics(
                    labels[index : index + 1],
                    target[index : index + 1],
                    footprint[index : index + 1],
                    taxonomy,
                )
                repaired_metrics = floorplan_metrics(
                    torch.from_numpy(repaired)[None].to(args.device),
                    target[index : index + 1],
                    footprint[index : index + 1],
                    taxonomy,
                )
                raw_reward = (
                    raw_metrics["room_area_coverage"]
                    + raw_metrics["opening_wall_alignment"]
                    + raw_metrics["furniture_room_containment"]
                )
                repaired_reward = (
                    repaired_metrics["room_area_coverage"]
                    + repaired_metrics["opening_wall_alignment"]
                    + repaired_metrics["furniture_room_containment"]
                )
                accepted = (
                    repaired_metrics["room_macro_iou"] >= raw_metrics["room_macro_iou"]
                    and repaired_reward >= raw_reward
                    and repaired_metrics["furniture_iou"]
                    >= raw_metrics["furniture_iou"] - 1.0e-8
                )
                fpeval_delta = None
                if args.use_fpeval_edit_cost:
                    raw_fpeval = fpeval_raster_metrics(
                        raw,
                        target[index].cpu().numpy(),
                        taxonomy,
                        tolerance_fractions=(0.05,),
                    )["summary"]["0.05"]
                    repaired_fpeval = fpeval_raster_metrics(
                        repaired,
                        target[index].cpu().numpy(),
                        taxonomy,
                        tolerance_fractions=(0.05,),
                    )["summary"]["0.05"]
                    decision = guarded_summary_decision(
                        raw_fpeval,
                        repaired_fpeval,
                    )
                    accepted = accepted and decision.accepted
                    fpeval_delta = decision.deltas["edit_cost"]
                rows.append(
                    {
                        "case": name,
                        **repair_decision_features(raw, repaired, footprint_np, taxonomy),
                        "delta_reward": repaired_reward - raw_reward,
                        "delta_room_iou": repaired_metrics["room_macro_iou"] - raw_metrics["room_macro_iou"],
                        "accepted": int(accepted),
                        "fpeval_edit_cost_delta": fpeval_delta,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "acceptance_rate": sum(row["accepted"] for row in rows) / len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
