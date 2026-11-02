#!/usr/bin/env python3
"""Refine deployable raster room proposals without changing room identity."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from train_vector_proposal_refiner_smoke import _masked_mae, _resample

from floorflow.data.raster_vector_geometry_dataset import (
    RasterVectorGeometryDataset,
)
from floorflow.evaluation.room_corner_metrics import room_corner_angle_metrics
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.geometry.core import compose_known_exterior
from floorflow.geometry.room_polygons import room_polygon_rows
from floorflow.heads.raster_control_bank import load_raster_room_model
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.heads.vector_proposal_refiner import VectorProposalRefiner
from floorflow.representation.polygon_targets import rasterize_room_polygons

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/raster_vector_proposal_refiner_screen256_v1",
    )
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--max-rooms", type=int, default=24)
    parser.add_argument("--polygon-points", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2908)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def raster_features(room_model, footprint, image):
    with torch.no_grad():
        hidden = room_model.encode_control(footprint, control_image=image)
        probability = room_model.final(hidden).softmax(dim=1)
    return torch.cat([hidden, probability], dim=1)


def proposal_targets(batch, labels, taxonomy, max_rooms, points):
    batch_size = labels.shape[0]
    proposals = torch.zeros(batch_size, max_rooms, points, 2)
    classes = torch.zeros(batch_size, max_rooms, dtype=torch.long)
    valid = torch.zeros(batch_size, max_rooms, dtype=torch.bool)
    room_lookup = {
        class_id: index
        for index, class_id in enumerate(taxonomy.room_ids)
    }
    scale = float(labels.shape[-1])
    for index in range(batch_size):
        rows = room_polygon_rows(
            labels[index].cpu().numpy(),
            taxonomy,
            min_area=8,
            epsilon=1.5,
            footprint=batch["footprint"][index].cpu().numpy(),
            tile_footprint=True,
        )
        if not rows:
            continue
        proposal_polygons = [
            _resample(np.asarray(row["polygon"], dtype=np.float32) / scale, points)
            for row in rows
        ]
        proposal_classes = np.asarray(
            [room_lookup.get(taxonomy.class_id(row["label"]), 0) for row in rows]
        )
        proposal_centers = np.stack(
            [polygon.mean(axis=0) for polygon in proposal_polygons]
        )
        target_count = int(batch["room_exists"][index].sum())
        target_centers = (
            batch["room_polygons"][index].detach().cpu().mean(dim=1).numpy()
        )
        target_classes = batch["room_classes"][index].detach().cpu().numpy()
        cost = np.linalg.norm(
            proposal_centers[:, None] - target_centers[None, :target_count],
            axis=-1,
        )
        cost += 0.5 * (
            proposal_classes[:, None]
            != target_classes[None, :target_count]
        )
        rows_idx, columns_idx = linear_sum_assignment(cost)
        for proposal_index, target_index in zip(
            rows_idx,
            columns_idx,
            strict=True,
        ):
            proposals[index, target_index] = torch.from_numpy(
                proposal_polygons[proposal_index]
            )
            classes[index, target_index] = int(
                proposal_classes[proposal_index]
            )
            valid[index, target_index] = True
    return {
        "room_polygons": proposals,
        "room_classes": classes,
        "room_valid": valid,
    }


def structural_labels(polygons, classes, valid, footprint, taxonomy):
    labels = rasterize_room_polygons(
        polygons,
        classes,
        valid,
        taxonomy,
        height=footprint.shape[-2],
        width=footprint.shape[-1],
    )
    seam = partition_wall_mask(labels, radius=1)
    labels = torch.where(
        seam,
        torch.full_like(labels, taxonomy.class_id("wall")),
        labels,
    )
    return compose_known_exterior(
        labels,
        footprint,
        opening_ids=taxonomy.opening_ids,
    )


@torch.inference_mode()
def evaluate(
    refiner,
    room_model,
    loader,
    taxonomy,
    device,
    max_rooms,
    points,
):
    refiner.eval()
    rows = []
    offsets = []
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
        room_logits = room_model(
            moved["footprint"],
            control_image=moved["image"],
        )
        room_ids = torch.tensor(taxonomy.room_ids, device=device)
        labels = room_ids[room_logits.argmax(dim=1)] * moved["footprint"].long()
        proposals = proposal_targets(
            moved,
            labels,
            taxonomy,
            max_rooms,
            points,
        )
        empty_walls = torch.zeros(
            labels.shape[0], 1, 2, 2, device=device
        )
        empty_wall_valid = torch.zeros(
            labels.shape[0], 1, dtype=torch.bool, device=device
        )
        empty_openings = torch.zeros(
            labels.shape[0], 1, 2, device=device
        )
        empty_opening_valid = torch.zeros(
            labels.shape[0], 1, dtype=torch.bool, device=device
        )
        refined = refiner(
            features,
            proposals["room_polygons"].to(device),
            proposals["room_valid"].to(device),
            empty_walls,
            empty_wall_valid,
            empty_openings,
            empty_opening_valid,
        )["room_polygons"]
        offsets.append(
            float(
                (refined - proposals["room_polygons"].to(device))
                .abs()
                .mean()
            )
        )
        for arm, polygons in (
            ("baseline", proposals["room_polygons"].to(device)),
            ("refined", refined),
        ):
            arm_labels = structural_labels(
                polygons,
                proposals["room_classes"].to(device),
                proposals["room_valid"].to(device),
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
                rows.append(
                    {
                        "arm": arm,
                        "room_macro_iou": suite["visible_semantic"][
                            "room_macro_iou"
                        ],
                        "instance_room_f1": instance[
                            "instance_f1_at_threshold"
                        ],
                        "room_closure_rate": structural[
                            "room_closure_rate"
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
    return {
        arm: {
            key: float(np.mean([row[key] for row in rows if row["arm"] == arm]))
            for key in rows[0]
            if key != "arm"
        }
        for arm in ("baseline", "refined")
    } | {"mean_coordinate_offset": float(np.mean(offsets))}


def selection_score(metrics):
    return (
        metrics["refined"]["corner_f1"]
        + metrics["refined"]["angle_f1"]
        + metrics["refined"]["instance_room_f1"]
        + 0.2 * metrics["refined"]["room_closure_rate"]
        + 0.1 * metrics["refined"]["wall_continuity_rate"]
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_dataset = RasterVectorGeometryDataset(
        args.geometry_root,
        args.data_root,
        "train",
        limit=args.limit_train,
    )
    taxonomy = train_dataset.taxonomy
    val_dataset = RasterVectorGeometryDataset(
        args.geometry_root,
        args.data_root,
        "val",
        limit=args.limit_val,
    )
    room_model = load_raster_room_model(
        args.room_checkpoint,
        taxonomy,
        device,
    )
    room_model.requires_grad_(False)
    refiner = VectorProposalRefiner(
        in_channels=128 + len(taxonomy.room_ids),
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    history = []
    best_score = -float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        refiner.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            features = raster_features(
                room_model,
                moved["footprint"],
                moved["image"],
            )
            room_logits = room_model(
                moved["footprint"],
                control_image=moved["image"],
            )
            room_ids = torch.tensor(taxonomy.room_ids, device=device)
            labels = room_ids[room_logits.argmax(dim=1)] * moved[
                "footprint"
            ].long()
            proposals = proposal_targets(
                moved,
                labels,
                taxonomy,
                args.max_rooms,
                args.polygon_points,
            )
            valid = proposals["room_valid"].to(device)
            output = refiner(
                features,
                proposals["room_polygons"].to(device),
                valid,
                torch.zeros(labels.shape[0], 1, 2, 2, device=device),
                torch.zeros(labels.shape[0], 1, dtype=torch.bool, device=device),
                torch.zeros(labels.shape[0], 1, 2, device=device),
                torch.zeros(labels.shape[0], 1, dtype=torch.bool, device=device),
            )
            loss = _masked_mae(
                output["room_polygons"],
                moved["room_polygons"][:, : args.max_rooms],
                valid,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                refiner.parameters(),
                args.gradient_clip,
            )
            optimizer.step()
            running += float(loss.detach()) * labels.shape[0]
            seen += labels.shape[0]
        metrics = evaluate(
            refiner,
            room_model,
            val_loader,
            taxonomy,
            device,
            args.max_rooms,
            args.polygon_points,
        )
        score = selection_score(metrics)
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "selection_score": score,
            **metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {
            "refiner": refiner.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": vars(args),
            "metrics": metrics,
        }
        torch.save(state, args.output_dir / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(state, args.output_dir / "best.pt")
    report = {
        "schema_version": "raster-vector-proposal-refiner-screen-v1",
        "scope": "mechanism screening only",
        "train_cases": len(train_dataset),
        "validation_cases": len(val_dataset),
        "test_opened": False,
        "target_used_at_inference": False,
        "identity_preserved": True,
        "best_score": best_score,
        "history": history,
        "config": vars(args),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(args.output_dir / "report.json"), "best_score": best_score}, indent=2))


if __name__ == "__main__":
    main()
