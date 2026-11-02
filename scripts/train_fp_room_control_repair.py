#!/usr/bin/env python3
"""Train a mask-aware room-control repair prior."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.control_perturbation import random_spatial_block_mask
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.coarse_layout_refiner import CoarseLayoutRefiner
from floorflow.representation.coarse_layout import coarse_room_layout_logits
from floorflow.training.distribution_runtime import set_training_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--min-dropout", type=float, default=0.0)
    parser.add_argument("--max-dropout", type=float, default=0.5)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def room_loss(logits: torch.Tensor, labels: torch.Tensor, room_ids: torch.Tensor) -> torch.Tensor:
    lookup = torch.full(
        (max(int(labels.max()), int(room_ids.max())) + 1,),
        -100,
        dtype=torch.long,
        device=labels.device,
    )
    lookup[room_ids] = torch.arange(len(room_ids), device=labels.device)
    target = lookup[labels.clamp_max(len(lookup) - 1)]
    ce = F.cross_entropy(logits, target.clamp_min(0), ignore_index=-100)
    probability = logits.softmax(dim=1)
    one_hot = F.one_hot(
        target.clamp_min(0),
        len(room_ids),
    ).permute(0, 3, 1, 2).float()
    valid = (target >= 0)[:, None].float()
    one_hot = one_hot * valid
    intersection = (probability * one_hot).sum(dim=(0, 2, 3))
    denominator = (probability * valid).sum(dim=(0, 2, 3)) + one_hot.sum(
        dim=(0, 2, 3)
    )
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return ce + dice


def support_iou(
    logits: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    room_ids: torch.Tensor,
) -> float:
    predicted = logits.max(dim=1).values > 0.0
    expected = torch.isin(target, room_ids)
    predicted = predicted & footprint.bool()
    union = torch.logical_or(predicted, expected).sum()
    if int(union) == 0:
        return 1.0
    return float(torch.logical_and(predicted, expected).sum() / union)


@torch.no_grad()
def validate(
    model,
    loader,
    taxonomy,
    room_ids,
    args,
) -> dict[str, float]:
    model.eval()
    results = {}
    for fraction in (0.0, 0.1, 0.25):
        values = []
        for target, footprint, _floor_ids, _names in loader:
            target = target.to(args.device)
            footprint = footprint.to(args.device)
            clean = coarse_room_layout_logits(
                target,
                taxonomy,
                size=args.layout_size,
            )
            masks = random_spatial_block_mask(
                target.shape[0],
                clean.shape[-2:],
                fraction,
                generator=torch.Generator().manual_seed(
                    args.seed + int(fraction * 1000)
                ),
            ).to(args.device)
            repaired = model(clean * (~masks)[:, None], footprint, ~masks)
            values.append(
                support_iou(repaired, target, footprint, room_ids)
            )
        results[f"support_iou_p{round(fraction * 100):03d}"] = sum(
            values
        ) / len(values)
    results["selection_score"] = sum(results.values()) / len(results)
    return results


def main() -> None:
    args = parse_args()
    set_training_seed(args.seed)
    args.device = torch.device(args.device)
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=False,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    taxonomy = train.taxonomy
    room_ids = torch.tensor(taxonomy.room_ids, device=args.device)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.device.type == "cuda",
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.type == "cuda",
    )
    model = CoarseLayoutRefiner(
        len(taxonomy.room_ids),
        masked_input=True,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1.0e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        total = 0.0
        seen = 0
        for target, footprint, _floor_ids, _names in train_loader:
            target = target.to(args.device)
            footprint = footprint.to(args.device)
            clean = coarse_room_layout_logits(
                target,
                taxonomy,
                size=args.layout_size,
            )
            fractions = (
                args.min_dropout
                + (args.max_dropout - args.min_dropout)
                * torch.rand(target.shape[0])
            )
            masks = random_spatial_block_mask(
                target.shape[0],
                clean.shape[-2:],
                fractions,
            ).to(args.device)
            repaired = model(
                clean * (~masks)[:, None],
                footprint,
                ~masks,
            )
            loss = room_loss(repaired, target, room_ids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
        metrics = validate(
            model,
            val_loader,
            taxonomy,
            room_ids,
            args,
        )
        record = {
            "epoch": epoch,
            "train_loss": total / max(1, seen),
            "seconds": time.perf_counter() - started,
            **metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        checkpoint = {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "layout_size": args.layout_size,
            "masked_input": True,
            "seed": args.seed,
            "metrics": metrics,
            "config": vars(args) | {"device": str(args.device)},
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["selection_score"] > best:
            best = metrics["selection_score"]
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
