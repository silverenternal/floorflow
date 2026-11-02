#!/usr/bin/env python3
"""Train a contour-only Trinity room-latent flow model on a fixed probe."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.model.trinity_contour_room import (
    ContourTrinityRoomModel,
    FeatureTrinityRoomModel,
)
from floorflow.objectives.trinity_constraints import trinity_constraint_loss
from floorflow.representation.trinity_latent import (
    render_room_boxes,
    room_batch_from_labels,
)
from floorflow.training.trinity_flow import (
    clamp_room_latent,
    gather_matched_room_targets,
    make_trinity_flow,
    match_room_tokens,
    prepare_room_flow,
)
from floorflow.training.checkpoint import load_checkpoint_model
from trinity.decode import z_to_xywh
from trinity.models.backbone import DenoiserCond


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-train", type=int, default=1024)
    parser.add_argument("--limit-val", type=int, default=32)
    parser.add_argument("--room-tokens", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument(
        "--count-source",
        choices=("predicted", "oracle"),
        default="predicted",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition-checkpoint", type=Path, default=None)
    return parser.parse_args()


def build_room_batch(target, footprint, taxonomy, room_tokens):
    return room_batch_from_labels(
        target,
        taxonomy,
        footprint,
        max_rooms=room_tokens,
    )


def masked_flow_loss(prediction, target, token_mask):
    mask = token_mask.to(prediction.dtype).unsqueeze(-1)
    error = F.smooth_l1_loss(prediction * mask, target * mask, reduction="none")
    return error.sum() / mask.expand_as(error).sum().clamp_min(1.0)


def condition_input(model, backbone, footprint, floor_ids):
    if backbone is None:
        return footprint
    zero = torch.zeros(
        footprint.shape[0],
        backbone.target_channels,
        *footprint.shape[-2:],
        device=footprint.device,
        dtype=footprint.dtype,
    )
    with torch.no_grad():
        features = backbone._encode_features(
            zero,
            torch.zeros(footprint.shape[0], device=footprint.device),
            footprint,
            floor_ids=floor_ids,
        )
    return features.detach()


@torch.inference_mode()
def sample_latents(model, footprint, framing, backbone=None, floor_ids=None, steps=8):
    source = condition_input(model, backbone, footprint, floor_ids)
    features, _routing = model.encoder(source)
    condition = DenoiserCond(features=features)
    logits = model.token_head(features).squeeze(-1)
    log_area = model.area_head(features).squeeze(-1)
    classes = model.class_head(features).argmax(dim=-1)
    z = torch.randn(
        footprint.shape[0],
        model.room_tokens,
        3,
        device=footprint.device,
    )
    for step in range(steps):
        t0 = 1.0 - step / steps
        t1 = 1.0 - (step + 1) / steps
        time = torch.full(
            (footprint.shape[0],),
            t0,
            device=footprint.device,
        )
        target = model.denoiser(z, time, condition)
        x0 = framing.x0_from_target(z, target, time)
        z = z + (t1 - t0) * framing.x0_to_velocity(z, x0, time)
    final_time = torch.full(
        (footprint.shape[0],),
        1.0e-3,
        device=footprint.device,
    )
    target = model.denoiser(z, final_time, condition)
    x0 = framing.x0_from_target(z, target, final_time)
    return x0, logits, log_area, classes


@torch.inference_mode()
def validate(model, loader, taxonomy, framing, args, backbone=None):
    model.eval()
    room_ids = torch.tensor(taxonomy.room_ids, device=args.device)
    rows = []
    for target, footprint, floor_ids, _names in loader:
        target = target.to(args.device)
        footprint = footprint.to(args.device)
        z0_hat, token_logits, log_area, classes = sample_latents(
            model,
            footprint,
            framing,
            backbone=backbone,
            floor_ids=floor_ids.to(args.device),
        )
        target_batch = build_room_batch(
            target,
            footprint,
            taxonomy,
            args.room_tokens,
        )
        token_mask = (
            target_batch.token_mask
            if args.count_source == "oracle"
            else token_logits.sigmoid() > 0.5
        )
        empty = ~token_mask.any(dim=1)
        if empty.any():
            top = token_logits.argmax(dim=1)
            token_mask[empty, top[empty]] = True
        areas = log_area.exp().clamp_min(1.0)
        scale = areas.masked_fill(~token_mask, 0.0).sum(dim=1).sqrt()
        boxes = z_to_xywh(z0_hat, areas, scale)
        labels = render_room_boxes(
            boxes,
            classes,
            token_mask,
            footprint,
            taxonomy,
        )
        for index in range(target.shape[0]):
            support = torch.isin(labels[index], room_ids)
            suite = room_metric_suite(
                labels[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=support.cpu().numpy(),
                predicted_instance_support=support.cpu().numpy(),
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
    device = torch.device(args.device)
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
    loader_args = {
        "batch_size": args.batch_size,
        "workers": args.workers,
    }
    train_loader = DataLoader(
        train,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    backbone = None
    if args.condition_checkpoint is not None:
        backbone, _config, _taxonomy, _ = load_checkpoint_model(
            args.condition_checkpoint,
            device,
            taxonomy=taxonomy,
        )
        backbone.eval().requires_grad_(False)
        model = FeatureTrinityRoomModel(
            backbone.output.in_channels,
            hidden=args.hidden,
            depth=args.depth,
            heads=args.heads,
            room_tokens=args.room_tokens,
            num_room_classes=len(taxonomy.room_ids),
        ).to(device)
    else:
        model = ContourTrinityRoomModel(
            hidden=args.hidden,
            depth=args.depth,
            heads=args.heads,
            room_tokens=args.room_tokens,
            num_room_classes=len(taxonomy.room_ids),
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    framing, time_sampler = make_trinity_flow()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        totals = {"loss": 0.0, "flow": 0.0, "physics": 0.0, "class": 0.0}
        seen = 0
        for target, footprint, floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            floor_ids = floor_ids.to(device)
            batch = build_room_batch(
                target,
                footprint,
                taxonomy,
                args.room_tokens,
            )
            flow = prepare_room_flow(
                batch.z,
                batch.token_mask,
                framing=framing,
                time_sampler=time_sampler,
            )
            source = condition_input(
                model,
                backbone,
                footprint,
                floor_ids,
            )
            prediction, auxiliary = model.forward_with_aux(
                flow.x_t,
                flow.time,
                source,
            )
            x0_hat = clamp_room_latent(
                framing.x0_from_target(flow.x_t, prediction, flow.time),
                batch.token_mask,
            )
            matching = match_room_tokens(
                x0_hat,
                auxiliary["class_logits"],
                auxiliary["log_area"],
                batch,
            )
            matched_mask = matching["matched_mask"]
            matched_flow_target = gather_matched_room_targets(
                flow.target,
                matching["target_for_prediction"],
            )
            matched_classes = gather_matched_room_targets(
                batch.room_classes,
                matching["target_for_prediction"],
            )
            matched_areas = gather_matched_room_targets(
                batch.area_targets,
                matching["target_for_prediction"],
            )
            flow_loss = masked_flow_loss(
                prediction,
                matched_flow_target,
                matched_mask,
            )
            physics, _physics_components = trinity_constraint_loss(
                x0_hat,
                batch,
                time=flow.time,
            )
            valid_tokens = matched_mask
            if bool(valid_tokens.any()):
                class_loss = F.cross_entropy(
                    auxiliary["class_logits"][valid_tokens],
                    matched_classes[valid_tokens],
                )
                area_loss = F.smooth_l1_loss(
                    auxiliary["log_area"][valid_tokens],
                    torch.log1p(matched_areas[valid_tokens]),
                )
            else:
                class_loss = auxiliary["class_logits"].sum() * 0.0
                area_loss = auxiliary["log_area"].sum() * 0.0
            token_loss = F.binary_cross_entropy_with_logits(
                auxiliary["token_logits"],
                matched_mask.float(),
            )
            loss = (
                flow_loss
                + args.physics_weight * physics
                + 0.2 * class_loss
                + 0.2 * token_loss
                + 0.1 * area_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach()) * target.shape[0]
            totals["flow"] += float(flow_loss.detach()) * target.shape[0]
            totals["physics"] += float(physics.detach()) * target.shape[0]
            totals["class"] += float(class_loss.detach()) * target.shape[0]
            seen += target.shape[0]
        metrics = validate(
            model,
            val_loader,
            taxonomy,
            framing,
            args,
            backbone=backbone,
        )
        record = {
            "epoch": epoch,
            **{f"train_{key}": value / max(1, seen) for key, value in totals.items()},
            "seconds": time.perf_counter() - started,
            "count_source": args.count_source,
            "physics_weight": args.physics_weight,
            **metrics,
        }
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": vars(args),
            "condition_backbone_checkpoint": (
                str(args.condition_checkpoint)
                if args.condition_checkpoint is not None
                else None
            ),
            "metrics": metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["room_macro_iou"] > best:
            best = metrics["room_macro_iou"]
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
