#!/usr/bin/env python3
"""Train only the semantic-element adapter on a frozen pixel distribution model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import (
    restore_distribution_model,
    save_distribution_model,
)
from floorflow.distribution.element_loss import element_set_loss
from floorflow.distribution.element_targets import ElementTargets, element_targets
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.training.runtime_acceleration import RuntimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize-distribution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=96)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def query_counts(queries: int = 8) -> dict[str, int]:
    return {
        "wall": queries,
        "opening": queries,
        "room": queries,
        "furniture": queries,
        "support": max(1, queries // 2),
    }


@torch.inference_mode()
def validate(model, loader, taxonomy, device) -> dict[str, float]:
    model.eval()
    values = []
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    for target, footprint, floor_ids, _ in loader:
        target = target.to(device)
        footprint = footprint[:, None].to(device)
        output = model.contour_initial_state(
            footprint,
            floor_ids=floor_ids.to(device),
        )
        prediction = (
            output["pixel"]["logits"].argmax(dim=1) * footprint[:, 0].long()
        )
        for index in range(target.shape[0]):
            suite = room_metric_suite(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index, 0].cpu().numpy(),
                taxonomy,
                predicted_room_support=torch.isin(
                    prediction[index], room_ids
                ).cpu().numpy(),
                predicted_instance_support=torch.isin(
                    prediction[index], room_ids
                ).cpu().numpy(),
                target_room_support=torch.isin(
                    target[index], room_ids
                ).cpu().numpy(),
            )
            instance = suite["matched_instance"]
            values.append(
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
                }
            )
    return {
        key: float(np.mean([row[key] for row in values]))
        for key in values[0]
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    runtime = RuntimeConfig(
        precision=args.precision,
        persistent_workers=args.workers > 0,
    )
    model, taxonomy, payload = restore_distribution_model(
        args.initialize_distribution,
        device,
        element_mask_version="v2",
    )
    model.base.requires_grad_(False)
    model.hub.pixel.requires_grad_(False)
    model.cfm_bridge.requires_grad_(False)
    model.hub.element_feedback_scale.data.zero_()
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=int(
            payload.get("training_args", {}).get("image_size", 256)
        ),
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
    loader_options = runtime.dataloader_options(device, args.workers)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    optimizer = torch.optim.AdamW(
        model.hub.element.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_updates = args.max_updates or args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_updates),
        eta_min=args.lr * 0.1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    update = 0
    stop = False
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        seen = 0
        for target, footprint, floor_ids, _ in train_loader:
            target_cpu = target
            target = target.to(device)
            footprint = footprint[:, None].to(device)
            floor_ids = floor_ids.to(device)
            targets = element_targets(
                target_cpu,
                taxonomy,
                model.hub.hierarchy,
                max_queries=query_counts(),
            )
            targets = {
                name: ElementTargets(
                    masks=value.masks.to(device),
                    types=value.types.to(device),
                    valid=value.valid.to(device),
                )
                for name, value in targets.items()
            }
            with runtime.autocast(device):
                output = model.contour_initial_state(
                    footprint,
                    floor_ids=floor_ids,
                )
                loss, _ = element_set_loss(
                    output["elements"],
                    targets,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.hub.element.parameters(),
                args.grad_clip,
            )
            optimizer.step()
            scheduler.step()
            update += 1
            seen += target.shape[0]
            running += float(loss.detach()) * target.shape[0]
            if update % 50 == 0:
                print(
                    json.dumps(
                        {
                            "update": update,
                            "loss_ema": running / max(1, seen),
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    ),
                    flush=True,
                )
            if update >= total_updates:
                stop = True
                break
        metrics = validate(model, val_loader, taxonomy, device)
        record = {
            "epoch": epoch,
            "updates": update,
            "train_element_loss": running / max(1, seen),
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        print(json.dumps(record), flush=True)
        save_distribution_model(
            model,
            args.output_dir / "best.pt",
            taxonomy=taxonomy,
            extra={"training_args": vars(args), "val": record},
        )
        if stop:
            break


if __name__ == "__main__":
    main()
