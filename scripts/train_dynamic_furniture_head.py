#!/usr/bin/env python3
"""Train room-conditioned furniture instance queries on frozen contour features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.data_balance import (
    cached_class_presence_weights,
    make_weighted_sampler,
)
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.heads.dynamic_object_instance import (
    DynamicFurnitureInstanceHead,
    object_instance_targets,
)
from floorflow.heads.room_mask_decoder import room_mask_loss, room_mask_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution-checkpoint", type=Path, required=True)
    parser.add_argument("--initialize-head", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/dynamic_furniture_queries_v1"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".data/cubicasa5k"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=5126)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=24)
    parser.add_argument("--min-queries", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--class-balance", action="store_true")
    parser.add_argument("--rare-sampler", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def forward_head(model, head, footprint):
    with torch.no_grad():
        initial = model.contour_initial_state(footprint)
        features = initial["features"]
        room_ids = torch.tensor(
            model.hub.hierarchy.room_ids,
            device=footprint.device,
        )
        room_probability = initial["pixel"]["probability"][
            :, room_ids
        ].sum(dim=1, keepdim=True)
    return head(features, room_probability, footprint)


def evaluate(head, model, loader, taxonomy, device, max_instances):
    head.eval()
    rows = []
    for target, footprint, _floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint[:, None].to(device)
        prediction = forward_head(model, head, footprint)
        classes, masks, valid = object_instance_targets(
            target,
            taxonomy.furniture_ids,
            max_instances=max_instances,
        )
        rows.append(
            room_mask_metrics(
                prediction,
                classes,
                masks,
                valid,
            )
        )
    return {
        key: sum(float(row[key]) for row in rows) / max(1, len(rows))
        for key in rows[0]
    } if rows else {"mask_iou": 0.0}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, taxonomy, _ = restore_distribution_model(
        args.distribution_checkpoint,
        device,
    )
    model.requires_grad_(False).eval()
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=True,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    train_sampler = (
        make_weighted_sampler(
            cached_class_presence_weights(
                train,
                taxonomy.furniture_ids,
                power=0.5,
            )
        )
        if args.rare_sampler
        else None
    )
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    head = DynamicFurnitureInstanceHead(
        model.base.output.in_channels,
        len(taxonomy.furniture_ids),
        hidden_dim=args.hidden_dim,
        max_queries=args.max_queries,
        min_queries=args.min_queries,
    ).to(device)
    if args.initialize_head is not None:
        payload = torch.load(
            args.initialize_head,
            map_location="cpu",
            weights_only=False,
        )
        head.load_state_dict(payload["head"])
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=1.0e-4,
    )
    class_weights = None
    if args.class_balance:
        positive = torch.from_numpy(
            load_positive_weights(
                args.cache_root,
                taxonomy.num_target_channels,
                20.0,
            )
        )
        class_weights = positive[
            torch.tensor(taxonomy.furniture_ids) - 1
        ].to(device)
        class_weights /= class_weights.mean().clamp_min(1.0e-6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    for epoch in range(args.epochs):
        head.train()
        total = 0.0
        seen = 0
        for target, footprint, _floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint[:, None].to(device)
            prediction = forward_head(model, head, footprint)
            classes, masks, valid = object_instance_targets(
                target,
                taxonomy.furniture_ids,
                max_instances=args.max_queries,
            )
            loss, _components = room_mask_loss(
                prediction,
                classes,
                masks,
                valid,
                class_weights=class_weights,
                boundary_weight=2.0,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
        metrics = evaluate(
            head,
            model,
            val_loader,
            taxonomy,
            device,
            args.max_queries,
        )
        row = {
            "epoch": epoch,
            "train_loss": total / max(1, seen),
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["mask_iou"] > best:
            best = metrics["mask_iou"]
            torch.save(
                {
                    "head": head.state_dict(),
                    "taxonomy": taxonomy.to_json(),
                    "config": {
                        "max_queries": args.max_queries,
                        "min_queries": args.min_queries,
                        "hidden_dim": args.hidden_dim,
                        "in_channels": model.base.output.in_channels,
                        "architecture": "mask2former_dynamic_furniture",
                    },
                    "history": history,
                },
                args.output_dir / "best.pt",
            )
    (args.output_dir / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
