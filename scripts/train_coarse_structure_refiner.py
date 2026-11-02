#!/usr/bin/env python3
"""Train the coarse-layout second-stage wall/opening refiner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader

from floorflow.data.control_cache import CachedControlDataset
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.coarse_structure_refiner import (
    CoarseStructureRefiner,
    coarse_structure_control,
)
from floorflow.heads.geometry_heads import structural_candidate_mask
from floorflow.training.distribution_runtime import set_training_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/coarse_structure_refiner_v1"),
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--control-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=16)
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


def focal_binary_loss(logits, target, valid):
    probability = logits.sigmoid().clamp(1.0e-4, 1.0 - 1.0e-4)
    positive = target * (1.0 - probability).pow(2.0) * probability.log()
    negative = (
        (1.0 - target).pow(4.0)
        * probability.pow(2.0)
        * (1.0 - probability).log()
    )
    return -((positive + negative) * valid).sum() / valid.sum().clamp_min(1.0)


def dice_loss(logits, target, valid):
    probability = logits.sigmoid() * valid
    target = target * valid
    intersection = (probability * target).sum(dim=(2, 3))
    denominator = probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def binary_f1(probability, target, tolerance=2):
    predicted = probability >= 0.5
    expected = target >= 0.5
    if not predicted.any() and not expected.any():
        return 1.0
    if not predicted.any() or not expected.any():
        return 0.0
    predicted_near = binary_dilation(predicted, iterations=tolerance)
    expected_near = binary_dilation(expected, iterations=tolerance)
    precision = (predicted & expected_near).sum() / max(1, predicted.sum())
    recall = (expected & predicted_near).sum() / max(1, expected.sum())
    return float(2.0 * precision * recall / max(1.0e-8, precision + recall))


@torch.inference_mode()
def validate(model, loader, taxonomy, device, control_size):
    model.eval()
    rows = []
    for batch in loader:
        if len(batch) == 5:
            target, footprint, _floor_ids, _names, controls = batch
            control = controls["structure"].to(device)
        else:
            target, footprint, _floor_ids, _names = batch
            control = None
        target = target.to(device)
        footprint = footprint.to(device)
        if control is None:
            control = coarse_structure_control(
                target,
                footprint,
                taxonomy,
                size=control_size,
            )
        output = model(control)
        candidates = structural_candidate_mask(
            target,
            footprint,
            taxonomy,
            radius=4,
        )
        wall = (target == taxonomy.class_id("wall")).float()[:, None]
        opening = torch.isin(
            target,
            target.new_tensor(taxonomy.opening_ids),
        ).float()[:, None]
        for index in range(target.shape[0]):
            rows.append(
                {
                    "wall_f1": binary_f1(
                        output["wall_logits"][index, 0].sigmoid().cpu().numpy(),
                        wall[index, 0].cpu().numpy(),
                    ),
                    "opening_f1": binary_f1(
                        (
                            output["opening_logits"][index, 0].sigmoid()
                            * candidates[index]
                        ).cpu().numpy(),
                        opening[index, 0].cpu().numpy(),
                    ),
                }
            )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("wall_f1", "opening_f1")
    }


def main() -> None:
    args = parse_args()
    set_training_seed(args.seed)
    device = torch.device(args.device)
    taxonomy = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=1,
        augment=False,
    ).taxonomy
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=args.control_cache is None,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    if args.control_cache is not None:
        train = CachedControlDataset(train, args.control_cache, "train")
        val = CachedControlDataset(val, args.control_cache, "val")
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    input_channels = (
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3
    )
    model = CoarseStructureRefiner(
        input_channels,
        len(taxonomy.opening_ids),
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
    opening_ids = torch.tensor(taxonomy.opening_ids, device=device)
    lookup = torch.full(
        (taxonomy.num_classes,),
        -1,
        dtype=torch.long,
        device=device,
    )
    lookup[opening_ids] = torch.arange(len(opening_ids), device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        totals = {"loss": 0.0, "wall": 0.0, "opening": 0.0, "type": 0.0}
        seen = 0
        for batch in train_loader:
            if len(batch) == 5:
                target, footprint, _floor_ids, _names, controls = batch
                control = controls["structure"].to(device)
            else:
                target, footprint, _floor_ids, _names = batch
                control = None
            target = target.to(device)
            footprint = footprint.to(device)
            if control is None:
                control = coarse_structure_control(
                    target,
                    footprint,
                    taxonomy,
                    size=args.control_size,
                )
            output = model(control)
            candidates = structural_candidate_mask(
                target,
                footprint,
                taxonomy,
                radius=4,
            )[:, None].float()
            wall_target = (target == taxonomy.class_id("wall")).float()[:, None]
            opening_target = torch.isin(
                target,
                opening_ids,
            ).float()[:, None]
            wall_loss = focal_binary_loss(
                output["wall_logits"],
                wall_target,
                footprint[:, None].float(),
            ) + dice_loss(
                output["wall_logits"],
                wall_target,
                footprint[:, None].float(),
            )
            opening_loss = focal_binary_loss(
                output["opening_logits"],
                opening_target,
                candidates,
            ) + dice_loss(
                output["opening_logits"],
                opening_target,
                candidates,
            )
            local_class = lookup[target]
            valid_type = local_class >= 0
            if valid_type.any():
                type_loss = F.cross_entropy(
                    output["opening_type_logits"].permute(0, 2, 3, 1)[valid_type],
                    local_class[valid_type],
                )
            else:
                type_loss = output["opening_logits"].sum() * 0.0
            loss = wall_loss + 2.0 * opening_loss + 0.5 * type_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, value in (
                ("loss", loss),
                ("wall", wall_loss),
                ("opening", opening_loss),
                ("type", type_loss),
            ):
                totals[name] += float(value.detach()) * target.shape[0]
            seen += target.shape[0]
        scheduler.step()
        record = {
            "epoch": epoch,
            **{f"train_{key}": value / max(1, seen) for key, value in totals.items()},
            "seconds": time.perf_counter() - started,
        }
        if (epoch + 1) % max(1, args.validate_every) == 0 or epoch == args.epochs - 1:
            metrics = validate(
                model,
                val_loader,
                taxonomy,
                device,
                args.control_size,
            )
            record.update(metrics)
            checkpoint = {
                "model": model.state_dict(),
                "taxonomy": taxonomy.to_json(),
                "config": {
                    **vars(args),
                    "input_channels": input_channels,
                    "num_opening_classes": len(taxonomy.opening_ids),
                },
                "metrics": metrics,
            }
            torch.save(checkpoint, args.output_dir / "last.pt")
            score = metrics["wall_f1"] + metrics["opening_f1"]
            if score > best:
                best = score
                torch.save(checkpoint, args.output_dir / "best.pt")
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
