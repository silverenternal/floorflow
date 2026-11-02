#!/usr/bin/env python3
"""Evaluate conservative blends of raster vector proposal residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from train_raster_vector_proposal_refiner import (
    proposal_targets,
    raster_features,
    structural_labels,
)
from train_residential_predicted_refiner import _safe_fpeval

from floorflow.data.raster_vector_geometry_dataset import (
    RasterVectorGeometryDataset,
)
from floorflow.evaluation.fevp_metric_contracts import (
    FEVPTolerances,
    watertight_diagnostics,
)
from floorflow.evaluation.room_corner_metrics import room_corner_angle_metrics
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.heads.raster_control_bank import load_raster_room_model
from floorflow.heads.vector_proposal_refiner import VectorProposalRefiner

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_vector_proposal_refiner_screen256_v1/best.pt",
    )
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_geometry_v1",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster_vector_proposal_blend256_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = RasterVectorGeometryDataset(
        args.geometry_root,
        args.data_root,
        "val",
        limit=args.limit,
        offset=args.offset,
    )
    taxonomy = dataset.taxonomy
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    room_model = load_raster_room_model(
        args.room_checkpoint,
        taxonomy,
        device,
    )
    refiner = VectorProposalRefiner(
        in_channels=128 + len(taxonomy.room_ids),
        hidden_dim=int(config["hidden_dim"]),
        residual_scale=0.05,
    ).to(device)
    refiner.load_state_dict(payload["refiner"])
    refiner.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    scales = (0.0, 0.25, 0.5, 0.75, 1.0)
    rows = {scale: [] for scale in scales}
    with torch.inference_mode():
        for batch in loader:
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            features = raster_features(
                room_model,
                moved["footprint"],
                moved["image"],
            )
            logits = room_model(
                moved["footprint"],
                control_image=moved["image"],
            )
            room_ids = torch.tensor(taxonomy.room_ids, device=device)
            labels = room_ids[logits.argmax(dim=1)] * moved["footprint"].long()
            proposals = proposal_targets(
                moved,
                labels,
                taxonomy,
                int(config["max_rooms"]),
                int(config["polygon_points"]),
            )
            proposals = {
                key: value.to(device)
                for key, value in proposals.items()
            }
            output = refiner(
                features,
                proposals["room_polygons"],
                proposals["room_valid"],
                torch.zeros(labels.shape[0], 1, 2, 2, device=device),
                torch.zeros(labels.shape[0], 1, dtype=torch.bool, device=device),
                torch.zeros(labels.shape[0], 1, 2, device=device),
                torch.zeros(labels.shape[0], 1, dtype=torch.bool, device=device),
            )
            for scale in scales:
                polygons = (
                    proposals["room_polygons"]
                    + scale
                    * 0.05
                    * output["room_residual"]
                ).clamp(0.0, 1.0)
                arm_labels = structural_labels(
                    polygons,
                    proposals["room_classes"],
                    proposals["room_valid"],
                    moved["footprint"],
                    taxonomy,
                )
                for index in range(labels.shape[0]):
                    support = torch.isin(arm_labels[index], room_ids)
                    target_support = torch.isin(
                        moved["target"][index],
                        room_ids,
                    )
                    suite = room_metric_suite(
                        arm_labels[index].cpu().numpy(),
                        moved["target"][index].cpu().numpy(),
                        moved["footprint"][index].cpu().numpy(),
                        taxonomy,
                        predicted_room_support=support.cpu().numpy(),
                        predicted_instance_support=support.cpu().numpy(),
                        target_room_support=target_support.cpu().numpy(),
                    )
                    instance = suite["matched_instance"]
                    corner = room_corner_angle_metrics(
                        arm_labels[index].cpu().numpy(),
                        moved["target"][index].cpu().numpy(),
                        taxonomy,
                    )
                    structural = structural_legality_metrics(
                        arm_labels[index].cpu().numpy(),
                        moved["target"][index].cpu().numpy(),
                        taxonomy,
                    )
                    fpeval = _safe_fpeval(
                        arm_labels[index].cpu().numpy(),
                        moved["target"][index].cpu().numpy(),
                        taxonomy,
                    )
                    watertight = watertight_diagnostics(
                        raster_labels_to_wall_plan(
                            arm_labels[index].cpu().numpy(),
                            taxonomy,
                        ),
                        taxonomy=taxonomy,
                        labels=arm_labels[index].cpu().numpy(),
                        footprint=moved["footprint"][index].cpu().numpy(),
                        tolerances=FEVPTolerances(),
                    )
                    rows[scale].append(
                        {
                            "room_macro_iou": suite["visible_semantic"][
                                "room_macro_iou"
                            ],
                            "instance_room_f1": instance[
                                "instance_f1_at_threshold"
                            ],
                            "room_closure_rate": structural[
                                "room_closure_rate"
                            ],
                            "parsable": structural["parsable"],
                            "validity": float(
                                fpeval["prediction_validity"]["valid"]
                            ),
                            "watertight": float(
                                fpeval["prediction_validity"]["watertight"]
                            ),
                            "hole_count": watertight["hole_count"],
                            "dangling_endpoint_rate": watertight[
                                "dangling_endpoint_rate"
                            ],
                            "wall_continuity_rate": structural[
                                "wall_continuity_rate"
                            ],
                            "self_intersection_rate": structural[
                                "self_intersection_rate"
                            ],
                            "node_f1": structural["node_f1"],
                            "edge_overlap": structural["edge_overlap"],
                            "ged": structural["ged"],
                            **corner,
                        }
                    )
    summary = {
        scale: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
        }
        for scale, values in rows.items()
    }
    report = {
        "schema_version": "raster-vector-proposal-blend-v1",
        "scope": "mechanism screening only",
        "validation_cases": len(dataset),
        "validation_offset": args.offset,
        "test_opened": False,
        "target_used_at_inference": False,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
