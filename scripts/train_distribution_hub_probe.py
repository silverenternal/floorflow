#!/usr/bin/env python3
"""Train coupled hierarchical pixel/element heads on frozen contour features."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.coupling_loss import distribution_coupling_loss
from floorflow.distribution.element_loss import element_set_loss
from floorflow.distribution.element_targets import element_targets
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.hub import CoupledDistributionHub
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.pixel_distribution import pixel_distribution_loss
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initialize-pixel-head", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--pixel-grid", type=int, default=16)
    parser.add_argument("--element-grid", type=int, default=16)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--room-iou-weight", type=float, default=0.5)
    parser.add_argument("--tversky-weight", type=float, default=0.2)
    parser.add_argument("--element-weight", type=float, default=0.25)
    parser.add_argument("--coupling-weight", type=float, default=0.1)
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=32)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def contour_features(backbone, footprint, floor_ids):
    zero = torch.zeros(
        footprint.shape[0],
        backbone.target_channels,
        *footprint.shape[-2:],
        device=footprint.device,
        dtype=footprint.dtype,
    )
    with torch.no_grad():
        return backbone._encode_features(
            zero,
            torch.zeros(footprint.shape[0], device=footprint.device),
            footprint,
            floor_ids=floor_ids,
        )


def query_counts(queries: int) -> dict[str, int]:
    return {
        "wall": queries,
        "opening": queries,
        "room": queries,
        "furniture": queries,
        "support": max(1, queries // 2),
    }


@torch.inference_mode()
def validate(hub, backbone, loader, taxonomy, device):
    hub.eval()
    rows = []
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint[:, None].to(device)
        floor_ids = floor_ids.to(device)
        features = contour_features(backbone, footprint, floor_ids)
        output = hub(features, footprint=footprint)
        logits = output["pixel"]["logits"]
        prediction = logits.argmax(dim=1) * footprint[:, 0].long()
        room_ids = torch.tensor(taxonomy.room_ids, device=device)
        for index in range(target.shape[0]):
            suite = room_metric_suite(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index, 0].cpu().numpy(),
                taxonomy,
                predicted_room_support=torch.isin(
                    prediction[index],
                    room_ids,
                ).cpu().numpy(),
                predicted_instance_support=torch.isin(
                    prediction[index],
                    room_ids,
                ).cpu().numpy(),
                target_room_support=torch.isin(
                    target[index],
                    room_ids,
                ).cpu().numpy(),
            )
            instance = suite["matched_instance"]
            rows.append(
                {
                    "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
                    "instance_geometric_iou": instance["matched_geometric_iou"],
                    "instance_room_f1": instance["instance_f1_at_threshold"],
                    "instance_type_accuracy": instance["matched_type_accuracy"],
                }
            )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    backbone, config, taxonomy, _ = load_checkpoint_model(
        args.checkpoint,
        device,
        taxonomy=None,
    )
    backbone.eval().requires_grad_(False)
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    hub = CoupledDistributionHub(
        backbone.output.in_channels,
        hierarchy,
        hidden_channels=args.hidden_channels,
        pixel_grid=args.pixel_grid,
        element_grid=args.element_grid,
        queries_per_group=query_counts(args.queries),
    ).to(device)
    if args.initialize_pixel_head is not None:
        payload = torch.load(
            args.initialize_pixel_head,
            map_location="cpu",
            weights_only=False,
        )
        hub.pixel.base.load_state_dict(payload["model"], strict=False)
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=False,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    positive = torch.from_numpy(
        load_positive_weights(
            args.cache_root,
            taxonomy.num_target_channels,
            20.0,
        )
    ).to(device)
    class_weights = torch.cat([torch.ones(1, device=device), positive])
    optimizer = torch.optim.AdamW(
        hub.parameters(),
        lr=args.lr,
        weight_decay=1.0e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        hub.train()
        totals = {
            "pixel": 0.0,
            "element": 0.0,
            "coupling": 0.0,
            "total": 0.0,
        }
        seen = 0
        for target, footprint, floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint[:, None].to(device)
            floor_ids = floor_ids.to(device)
            features = contour_features(backbone, footprint, floor_ids)
            output = hub(features, footprint=footprint)
            pixel_loss, pixel_components = pixel_distribution_loss(
                output["pixel"]["logits"],
                target,
                footprint,
                taxonomy,
                class_weights=class_weights,
                ce_weight=args.ce_weight,
                room_iou_weight=args.room_iou_weight,
                tversky_weight=args.tversky_weight,
            )
            targets = element_targets(
                target,
                taxonomy,
                hierarchy,
                max_queries=query_counts(args.queries),
            )
            element_loss, _element_components = element_set_loss(
                output["elements"],
                targets,
            )
            coupling_loss, _coupling_components = distribution_coupling_loss(
                output["pixel"],
                output["elements"],
                hierarchy,
                footprint=footprint,
            )
            loss = (
                pixel_loss
                + args.element_weight * element_loss
                + args.coupling_weight * coupling_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hub.parameters(), 1.0)
            optimizer.step()
            batch = target.shape[0]
            totals["pixel"] += float(pixel_loss.detach()) * batch
            totals["element"] += float(element_loss.detach()) * batch
            totals["coupling"] += float(coupling_loss.detach()) * batch
            totals["total"] += float(loss.detach()) * batch
            seen += batch
        metrics = validate(
            hub,
            backbone,
            val_loader,
            taxonomy,
            device,
        )
        record = {
            "epoch": epoch,
            **{f"train_{key}": value / max(1, seen) for key, value in totals.items()},
            "seconds": time.perf_counter() - started,
            **metrics,
            "condition_mode": "contour_only",
            "architecture": "hierarchical_pixel_element_hub",
        }
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "pixel_head": hub.pixel.state_dict(),
            "element_head": hub.element.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": vars(args),
            "metrics": metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["room_macro_iou"] > best:
            best = metrics["room_macro_iou"]
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
