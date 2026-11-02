#!/usr/bin/env python3
"""Train geometry heads only on room-transition and envelope candidates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, maximum_filter
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.geometry_heads import (
    build_geometry_targets,
    geometry_heatmap_loss,
    structural_candidate_mask,
)
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/train/p6_geometry_full_v1/best.pt"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/geometry_candidate_head_v1"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def frozen_features(model, target, footprint, floor_ids):
    zeros = torch.zeros(
        target.shape[0],
        model.target_channels,
        *target.shape[-2:],
        device=target.device,
    )
    with torch.no_grad():
        return model._encode_features(
            zeros,
            torch.zeros(target.shape[0], device=target.device),
            footprint,
            floor_ids=floor_ids,
        )


def _binary_f1(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: int = 2,
    wall: bool = False,
) -> float:
    predicted = probability >= 0.7
    if wall:
        predicted = skeletonize(predicted)
        expected = skeletonize(target > 0.5)
    else:
        predicted &= predicted >= maximum_filter(probability, size=3)
        expected = target > 0.5
    if not predicted.any() and not expected.any():
        return 1.0
    if not predicted.any() or not expected.any():
        return 0.0
    predicted_near = binary_dilation(predicted, iterations=tolerance)
    expected_near = binary_dilation(expected, iterations=tolerance)
    precision = (predicted & expected_near).sum() / max(1, predicted.sum())
    recall = (expected & predicted_near).sum() / max(1, expected.sum())
    return float(2.0 * precision * recall / max(1.0e-8, precision + recall))


def validate(model, loader, taxonomy, device):
    model.geometry_heads.eval()
    values = []
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        features = frozen_features(
            model,
            target,
            footprint,
            floor_ids,
        )
        with torch.no_grad():
            predictions = model.geometry_heads(features)
        targets = build_geometry_targets(target, footprint, taxonomy)
        for index in range(target.shape[0]):
            values.append(
                {
                    "wall_f1": _binary_f1(
                        predictions["wall_centerline_logits"][index, 0]
                        .sigmoid()
                        .cpu()
                        .numpy(),
                        targets["wall_centerline"][index, 0].cpu().numpy(),
                        wall=True,
                    ),
                    "opening_f1": _binary_f1(
                        predictions["opening_center_logits"][index, 0]
                        .sigmoid()
                        .cpu()
                        .numpy(),
                        targets["opening_center"][index, 0].cpu().numpy(),
                    ),
                }
            )
    return {
        key: float(np.mean([value[key] for value in values]))
        for key in ("wall_f1", "opening_f1")
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        device,
    )
    if not hasattr(model, "geometry_heads"):
        raise ValueError("checkpoint does not expose geometry heads")
    model.requires_grad_(False)
    model.eval()
    head = model.geometry_heads
    head.requires_grad_(True)
    image_size = int(config.get("image_size", 256))
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=image_size,
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=True,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=image_size,
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.05,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        head.train()
        total = 0.0
        seen = 0
        for target, footprint, floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            floor_ids = floor_ids.to(device)
            features = frozen_features(model, target, footprint, floor_ids)
            targets = build_geometry_targets(target, footprint, taxonomy)
            candidates = structural_candidate_mask(
                target,
                footprint,
                taxonomy,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.precision == "bf16" and device.type == "cuda",
            ):
                predictions = head(features)
                loss, _components = geometry_heatmap_loss(
                    predictions,
                    targets,
                    structural_mask=candidates,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": total / max(1, seen),
            "seconds": time.perf_counter() - started,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if (epoch + 1) % max(1, args.validate_every) == 0 or epoch == args.epochs - 1:
            metrics = validate(model, val_loader, taxonomy, device)
            record.update(metrics)
            score = metrics["wall_f1"] + metrics["opening_f1"]
            checkpoint = {
                "geometry_heads": head.state_dict(),
                "base_checkpoint": str(args.checkpoint),
                "taxonomy": taxonomy.to_json(),
                "config": {**vars(args), "image_size": image_size},
                "metrics": metrics,
                "epoch": epoch,
            }
            torch.save(checkpoint, args.output_dir / "last.pt")
            if score > best:
                best = score
                torch.save(checkpoint, args.output_dir / "best.pt")
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
