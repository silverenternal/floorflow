#!/usr/bin/env python3
"""Train the detection-first room-aware two-level polygon decoder."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.polygon_loop import polygon_loop_loss
from floorflow.heads.room_polygon_decoder import RasterPolygonControl
from floorflow.representation.polygon_targets import (
    decode_polygon_loops,
    polygon_instance_targets,
)

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
        default=ROOT / "outputs/train/room_polygon_decoder_screen256_v1",
    )
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--points", type=int, default=24)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--class-weight", type=float, default=0.25)
    parser.add_argument("--polygon-weight", type=float, default=2.0)
    parser.add_argument("--chamfer-weight", type=float, default=1.0)
    parser.add_argument("--mask-dice-weight", type=float, default=1.0)
    parser.add_argument("--valid-weight", type=float, default=1.0)
    parser.add_argument("--valid-positive-weight", type=float, default=4.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--train-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_room_encoder(model: RasterPolygonControl, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.room.load_state_dict(payload["model"])


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["room_macro_iou"]
        + 0.25 * metrics["instance_f1"]
        + 0.1 * metrics["type_accuracy"]
    )


@torch.inference_mode()
def evaluate(
    model: RasterPolygonControl,
    loader: DataLoader,
    taxonomy,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model.eval()
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    rows = {"polygon": [], "composed": []}
    for target, footprint, image, _floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        outputs = model(footprint, control_image=image)
        polygon_labels = decode_polygon_loops(
            outputs,
            taxonomy,
            height=target.shape[-1],
            width=target.shape[-1],
            validity_threshold=0.0,
        )
        dense_labels = room_ids[
            outputs["dense_logits"].argmax(dim=1)
        ] * footprint.long()
        polygon_room = torch.isin(polygon_labels, room_ids)
        composed_labels = torch.where(polygon_room, polygon_labels, dense_labels)
        for index in range(target.shape[0]):
            expected = target[index].cpu().numpy()
            mask = footprint[index].cpu().numpy()
            target_support = np.isin(expected, taxonomy.room_ids)
            for name, labels in (
                ("polygon", polygon_labels[index].cpu().numpy()),
                ("composed", composed_labels[index].cpu().numpy()),
            ):
                predicted_support = np.isin(labels, taxonomy.room_ids)
                suite = room_metric_suite(
                    labels,
                    expected,
                    mask,
                    taxonomy,
                    predicted_room_support=predicted_support,
                    predicted_instance_support=predicted_support,
                    target_room_support=target_support,
                )
                instance = suite["matched_instance"]
                rows[name].append(
                    {
                        "room_macro_iou": float(
                            suite["visible_semantic"]["room_macro_iou"]
                        ),
                        "instance_geometric_iou": float(
                            instance["matched_geometric_iou"]
                        ),
                        "instance_f1": float(
                            instance["instance_f1_at_threshold"]
                        ),
                        "type_accuracy": float(
                            instance["matched_type_accuracy"]
                        ),
                        "count_abs_error": float(
                            instance["count_abs_error"]
                        ),
                    }
                )
    return {
        name: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
        }
        for name, values in rows.items()
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index != 0:
        raise ValueError("room polygon training is restricted to cuda:0")
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
    model = RasterPolygonControl(
        taxonomy,
        decoder_channels=args.decoder_channels,
        hidden_dim=args.hidden_dim,
        num_queries=args.queries,
        points=args.points,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    load_room_encoder(model, args.room_checkpoint)
    if not args.train_encoder:
        model.room.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
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
    best_score = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        if not args.train_encoder:
            model.room.eval()
        running = 0.0
        component_sums: dict[str, float] = {}
        steps = 0
        seen = 0
        for target, footprint, image, _floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            image = image.to(device)
            outputs = model(footprint, control_image=image)
            classes, polygons, valid = polygon_instance_targets(
                target,
                taxonomy,
                max_instances=args.queries,
                points=args.points,
            )
            loss, components = polygon_loop_loss(
                outputs,
                classes,
                polygons,
                valid,
                class_weight=args.class_weight,
                polygon_weight=args.polygon_weight,
                chamfer_weight=args.chamfer_weight,
                mask_dice_weight=args.mask_dice_weight,
                valid_weight=args.valid_weight,
                valid_positive_weight=args.valid_positive_weight,
                mask_rasterizer="diff_ras",
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.gradient_clip,
            )
            optimizer.step()
            running += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
            steps += 1
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + float(
                    value.detach().cpu()
                )
        validation = evaluate(
            model,
            val_loader,
            taxonomy,
            device,
        )
        current_score = max(
            selection_score(metrics)
            for metrics in validation.values()
        )
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "train_components": {
                key: value / max(1, steps)
                for key, value in sorted(component_sums.items())
            },
            "validation": validation,
            "selection_score": current_score,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": vars(args),
            "metrics": validation,
        }
        torch.save(state, args.output_dir / "last.pt")
        if current_score > best_score:
            best_score = current_score
            torch.save(state, args.output_dir / "best.pt")
    report = {
        "status": "complete",
        "scope": "detection-first room-aware two-level polygon decoder",
        "input_protocol": "native_cubicasa_raster",
        "target_used_at_inference": False,
        "train_cases": len(train_dataset),
        "validation_cases": len(val_dataset),
        "best_selection_score": best_score,
        "history": history,
        "config": vars(args),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
