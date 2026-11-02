#!/usr/bin/env python3
"""Train room-support refinement independently from visible room labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.control_cache import CachedControlDataset
from floorflow.data.dataset import CubiCasaGeometryDataset
from floorflow.heads.coarse_support_refiner import (
    CoarseSupportRefiner,
    coarse_support_control,
)
from floorflow.training.distribution_runtime import set_training_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_geometry_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/coarse_support_refiner_v1"),
    )
    parser.add_argument("--control-size", type=int, default=192)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--control-cache", type=Path, default=None)
    parser.add_argument("--initialize-from", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def focal_dice(logits, target, valid):
    probability = logits.sigmoid().clamp(1.0e-4, 1.0 - 1.0e-4)
    positive = target * (1.0 - probability).pow(2.0) * probability.log()
    negative = (
        (1.0 - target).pow(4.0)
        * probability.pow(2.0)
        * (1.0 - probability).log()
    )
    focal = -((positive + negative) * valid).sum() / valid.sum().clamp_min(1.0)
    dice_probability = logits.sigmoid() * valid
    target = target * valid
    intersection = (dice_probability * target).sum(dim=(2, 3))
    denominator = dice_probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1.0 - (
        (2.0 * intersection + 1.0) / (denominator + 1.0)
    ).mean()
    return focal + dice


@torch.inference_mode()
def validate(model, loader, taxonomy, device, control_size, output_size):
    model.eval()
    values = []
    for batch in loader:
        if len(batch) == 7:
            target, footprint, _floor_ids, _names, support, _occlusion, controls = batch
            control = controls["support"].to(device)
        else:
            target, footprint, _floor_ids, _names, support, _occlusion = batch
            control = None
        target = target.to(device)
        footprint = footprint.to(device)
        support = support.to(device)
        if control is None:
            control = coarse_support_control(
                target,
                footprint,
                taxonomy,
                size=control_size,
            )
        probability = model(
            control,
            output_size=(output_size, output_size),
        ).sigmoid()[:, 0]
        predicted = probability >= 0.5
        expected = (support > 0.5) & footprint.bool()
        intersection = (predicted & expected).sum(dim=(1, 2))
        union = (predicted | expected).sum(dim=(1, 2))
        values.extend(
            (intersection.float() / union.clamp_min(1).float()).cpu().tolist()
        )
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    set_training_seed(args.seed)
    device = torch.device(args.device)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train = CubiCasaGeometryDataset(
        args.data_root,
        "train",
        args.geometry_root,
        augment=args.control_cache is None,
        limit=args.limit_train,
    )
    val = CubiCasaGeometryDataset(
        args.data_root,
        "val",
        args.geometry_root,
        augment=False,
        limit=args.limit_val,
    )
    if args.control_cache is not None:
        train = CachedControlDataset(train, args.control_cache, "train")
        val = CachedControlDataset(val, args.control_cache, "val")
    taxonomy = train.taxonomy
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    input_channels = len(taxonomy.room_ids) + 4
    model = CoarseSupportRefiner(
        input_channels,
        base_channels=args.base_channels,
    ).to(device)
    if args.initialize_from is not None:
        payload = torch.load(
            args.initialize_from,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.05,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        total = 0.0
        seen = 0
        for batch in train_loader:
            if len(batch) == 7:
                target, footprint, _floor_ids, _names, support, _occlusion, controls = batch
                control = controls["support"].to(device)
            else:
                target, footprint, _floor_ids, _names, support, _occlusion = batch
                control = None
            target = target.to(device)
            footprint = footprint.to(device)
            support = support.to(device)
            if control is None:
                control = coarse_support_control(
                    target,
                    footprint,
                    taxonomy,
                    size=args.control_size,
                )
            logits = model(
                control,
                output_size=(args.output_size, args.output_size),
            )
            loss = focal_dice(
                logits,
                support[:, None].float(),
                footprint[:, None].float(),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": total / max(1, seen),
            "seconds": time.perf_counter() - started,
        }
        if (epoch + 1) % max(1, args.validate_every) == 0 or epoch == args.epochs - 1:
            iou = validate(
                model,
                val_loader,
                taxonomy,
                device,
                args.control_size,
                args.output_size,
            )
            record["support_iou"] = iou
            checkpoint = {
                "model": model.state_dict(),
                "taxonomy": taxonomy.to_json(),
                "config": {
                    **vars(args),
                    "input_channels": input_channels,
                },
                "metrics": {"support_iou": iou},
            }
            torch.save(checkpoint, args.output_dir / "last.pt")
            if iou > best:
                best = iou
                torch.save(checkpoint, args.output_dir / "best.pt")
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
