#!/usr/bin/env python3
"""Screen a learned boundary residual on native-raster room controls."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from train_residential_predicted_refiner import _safe_fpeval

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.evaluation.room_corner_metrics import room_corner_angle_metrics
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.geometry.core import compose_known_exterior, footprint_band
from floorflow.heads.predicted_room_identity import (
    PredictedRoomIdentityHead,
    room_identity_loss,
)
from floorflow.heads.raster_control_bank import load_raster_room_model
from floorflow.heads.structured_objects import partition_wall_mask

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
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
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/raster_room_boundary_refiner_v1",
    )
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2906)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def raster_edge(image: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=1, keepdim=True)
    horizontal = torch.zeros_like(gray)
    vertical = torch.zeros_like(gray)
    horizontal[:, :, :, 1:] = (
        gray[:, :, :, 1:] - gray[:, :, :, :-1]
    ).abs()
    vertical[:, :, 1:, :] = (
        gray[:, :, 1:, :] - gray[:, :, :-1, :]
    ).abs()
    return (horizontal + vertical).clamp(0.0, 2.0) / 2.0


def room_inputs(
    room_model,
    footprint: torch.Tensor,
    image: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        logits = room_model(footprint, control_image=image)
    probability = logits.softmax(dim=1)
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    band = footprint_band(footprint.float(), kernel_size=5)
    features = torch.cat(
        [probability, image, footprint.float(), raster_edge(image), band],
        dim=1,
    )
    return features, logits


def structural_room_labels(
    logits: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy,
) -> torch.Tensor:
    room_ids = torch.tensor(taxonomy.room_ids, device=logits.device)
    labels = room_ids[logits.argmax(dim=1)] * footprint.long()
    wall = partition_wall_mask(labels, radius=1)
    labels = torch.where(
        wall,
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
    head,
    room_model,
    loader,
    taxonomy,
    device: torch.device,
) -> dict[str, float]:
    head.eval()
    rows = []
    for target, footprint, image, _floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        features, base_logits = room_inputs(room_model, footprint, image)
        logits = head(features, base_logits)["room_logits"]
        labels = structural_room_labels(logits, footprint, taxonomy)
        room_ids = torch.tensor(taxonomy.room_ids, device=device)
        for index in range(target.shape[0]):
            support = torch.isin(labels[index], room_ids)
            target_support = torch.isin(target[index], room_ids)
            suite = room_metric_suite(
                labels[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=support.cpu().numpy(),
                predicted_instance_support=support.cpu().numpy(),
                target_room_support=target_support.cpu().numpy(),
            )
            instance = suite["matched_instance"]
            corner = room_corner_angle_metrics(
                labels[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
            )
            structural = structural_legality_metrics(
                labels[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
            )
            fpeval = _safe_fpeval(
                labels[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
            )
            fpeval_summary = fpeval.get("summary", {}).get("0.05", {})
            rows.append(
                {
                    "room_macro_iou": suite["visible_semantic"][
                        "room_macro_iou"
                    ],
                    "instance_geometric_iou": instance[
                        "matched_geometric_iou"
                    ],
                    "instance_room_f1": instance[
                        "instance_f1_at_threshold"
                    ],
                    "instance_type_accuracy": instance[
                        "matched_type_accuracy"
                    ],
                    "validity": float(
                        fpeval["prediction_validity"]["valid"]
                    ),
                    "watertight": float(
                        fpeval["prediction_validity"]["watertight"]
                    ),
                    "parsable": structural["parsable"],
                    "room_closure_rate": structural["room_closure_rate"],
                    "wall_continuity_rate": structural[
                        "wall_continuity_rate"
                    ],
                    "self_intersection_rate": structural[
                        "self_intersection_rate"
                    ],
                    "room_overlap_ratio": structural["room_overlap_ratio"],
                    "out_of_boundary_ratio": structural[
                        "out_of_boundary_ratio"
                    ],
                    "node_f1": structural["node_f1"],
                    "edge_overlap": structural["edge_overlap"],
                    "ged": structural["ged"],
                    "wall_f1": float(fpeval_summary.get("wall_f1", 0.0)),
                    "opening_f1": float(
                        fpeval_summary.get("opening_f1", 0.0)
                    ),
                    "edit_cost": float(
                        fpeval_summary.get("edit_cost", 0.0)
                    ),
                    **corner,
                }
            )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["corner_f1"]
        + metrics["angle_f1"]
        + 0.2 * metrics["instance_room_f1"]
        + 0.1 * metrics["room_macro_iou"]
        + 0.1 * metrics["watertight"]
        + 0.1 * metrics["wall_continuity_rate"]
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "train",
        image_size=256,
        cache_root=args.cache_root,
        augment=True,
        limit=args.limit_train,
        image_mode="native",
    )
    taxonomy = train_dataset.taxonomy
    val_dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_val,
        image_mode="native",
    )
    room_model = load_raster_room_model(
        args.room_checkpoint,
        taxonomy,
        device,
    )
    room_model.requires_grad_(False)
    feature_channels = len(taxonomy.room_ids) + 6
    head = PredictedRoomIdentityHead(
        feature_channels,
        len(taxonomy.room_ids),
        hidden_channels=args.hidden_channels,
        blocks=args.blocks,
    ).to(device)
    positive = torch.from_numpy(
        load_positive_weights(
            args.cache_root,
            taxonomy.num_target_channels,
            20.0,
        )
    ).to(device)
    room_indices = torch.tensor(
        [class_id - 1 for class_id in taxonomy.room_ids],
        device=device,
    )
    room_weights = positive[room_indices]
    room_weights = room_weights / room_weights.mean().clamp_min(1.0e-6)
    optimizer = torch.optim.AdamW(
        head.parameters(),
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
    baseline = evaluate(head, room_model, val_loader, taxonomy, device)
    history = []
    best_score = selection_score(baseline)
    best_metrics = baseline
    torch.save(
        {
            "model": head.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": vars(args) | {"feature_channels": feature_channels},
            "metrics": baseline,
        },
        args.output_dir / "baseline.pt",
    )
    for epoch in range(args.epochs):
        head.train()
        running = 0.0
        seen = 0
        for target, footprint, image, _floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            image = image.to(device)
            features, base_logits = room_inputs(room_model, footprint, image)
            logits = head(features, base_logits)["room_logits"]
            loss, components = room_identity_loss(
                logits,
                target,
                footprint,
                taxonomy.room_ids,
                class_weights=room_weights,
                dice_weight=args.dice_weight,
                boundary_weight=args.boundary_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(),
                args.gradient_clip,
            )
            optimizer.step()
            running += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
        metrics = evaluate(head, room_model, val_loader, taxonomy, device)
        score = selection_score(metrics)
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            **{key: float(value) for key, value in components.items()},
            **metrics,
            "selection_score": score,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {
            "model": head.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": vars(args) | {"feature_channels": feature_channels},
            "metrics": metrics,
        }
        torch.save(state, args.output_dir / "last.pt")
        if score > best_score:
            best_score = score
            best_metrics = metrics
            torch.save(state, args.output_dir / "best.pt")
    report = {
        "schema_version": "raster-room-boundary-refiner-screen-v1",
        "scope": "mechanism screening only",
        "train_cases": len(train_dataset),
        "validation_cases": len(val_dataset),
        "test_opened": False,
        "target_used_at_inference": False,
        "baseline": baseline,
        "best": best_metrics,
        "best_score": best_score,
        "history": history,
        "config": vars(args),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(args.output_dir / "report.json"), "metrics": best_metrics}, indent=2))


if __name__ == "__main__":
    main()
