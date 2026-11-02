#!/usr/bin/env python3
"""Train a standalone full pixel-distribution head on contour features."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.masked_channel_distribution import (
    MaskedChannelDistributionHead,
)
from floorflow.heads.pixel_distribution import (
    PixelDistributionHead,
    pixel_distribution_loss,
)
from floorflow.model.multiscale_pixel_decoder import MultiScalePixelDecoder
from floorflow.model.pmt_mask_decoder import PMTMaskDecoderAdapter
from floorflow.model.pretrained_vit import FrozenViTPatchFeatures
from floorflow.objectives.gradient_balancer import GradientNormBalancer
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--room-iou-weight", type=float, default=0.5)
    parser.add_argument("--tversky-weight", type=float, default=0.2)
    parser.add_argument("--dynamic-loss-weights", action="store_true")
    parser.add_argument(
        "--head-type",
        choices=("conv", "masked_channel", "pmt"),
        default="conv",
    )
    parser.add_argument("--pmt-blocks", type=int, default=2)
    parser.add_argument("--pmt-grid", type=int, default=16)
    parser.add_argument("--pmt-mask-dropout", type=float, default=0.25)
    parser.add_argument("--pmt-upscale-steps", type=int, default=2)
    parser.add_argument("--pmt-pixel-refiner", action="store_true")
    parser.add_argument("--multiscale-pixel", action="store_true")
    parser.add_argument("--vit-backbone", type=str, default=None)
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


def features_for(backbone, footprint, floor_ids, vit_adapter=None):
    if vit_adapter is not None:
        return vit_adapter(footprint)
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


@torch.inference_mode()
def validate(
    model,
    backbone,
    loader,
    taxonomy,
    device,
    pixel_decoder=None,
    vit_adapter=None,
):
    model.eval()
    rows = []
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        features = features_for(backbone, footprint, floor_ids, vit_adapter)
        if pixel_decoder is not None:
            features, _levels = pixel_decoder(features)
        logits = F.interpolate(
            model(features)["logits"],
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prediction = logits.argmax(dim=1)
        prediction = prediction * footprint.bool().long()
        room_ids = torch.tensor(taxonomy.room_ids, device=device)
        for index in range(target.shape[0]):
            support = torch.isin(prediction[index], room_ids)
            suite = room_metric_suite(
                prediction[index].cpu().numpy(),
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
    set_seed(args.seed)
    device = torch.device(args.device)
    backbone, config, taxonomy, _ = load_checkpoint_model(
        args.checkpoint,
        device,
        taxonomy=None,
    )
    backbone.eval().requires_grad_(False)
    vit_adapter = (
        FrozenViTPatchFeatures(
            backbone_name=args.vit_backbone,
            image_size=int(config.get("image_size", 256)),
        ).to(device)
        if args.vit_backbone
        else None
    )
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
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    pixel_decoder = (
        MultiScalePixelDecoder(
            backbone.output.in_channels,
            hidden_channels=args.hidden_channels,
            levels=3,
        ).to(device)
        if args.multiscale_pixel
        else None
    )
    head_input_channels = (
        args.hidden_channels
        if pixel_decoder is not None
        else (
            vit_adapter.channels
            if vit_adapter is not None
            else backbone.output.in_channels
        )
    )
    if args.head_type == "pmt":
        model = PMTMaskDecoderAdapter(
            head_input_channels,
            taxonomy.num_classes,
            hidden_channels=args.hidden_channels,
            heads=4,
            blocks=args.pmt_blocks,
            grid=args.pmt_grid,
            mask_dropout=args.pmt_mask_dropout,
            upscale_steps=args.pmt_upscale_steps,
            pixel_refiner=args.pmt_pixel_refiner,
        ).to(device)
    elif args.head_type == "masked_channel":
        model = MaskedChannelDistributionHead(
            head_input_channels,
            taxonomy.num_classes,
            hidden_channels=args.hidden_channels,
            heads=4,
            layers=2,
        ).to(device)
    else:
        model = PixelDistributionHead(
            head_input_channels,
            taxonomy.num_classes,
            hidden_channels=args.hidden_channels,
            room_ids=taxonomy.room_ids,
        ).to(device)
    positive = torch.from_numpy(
        load_positive_weights(
            args.cache_root,
            taxonomy.num_target_channels,
            20.0,
        )
    ).to(device)
    class_weights = torch.cat([torch.ones(1, device=device), positive])
    parameters = list(model.parameters())
    if pixel_decoder is not None:
        parameters.extend(pixel_decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1.0e-4)
    balancer = (
        GradientNormBalancer(
            {
                "ce": args.ce_weight,
                "room_iou": args.room_iou_weight,
                "tversky": args.tversky_weight,
            },
            reference="ce",
            update_interval=20,
        )
        if args.dynamic_loss_weights
        else None
    )
    global_step = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        totals = {
            "loss": 0.0,
            "ce": 0.0,
            "soft_iou_loss": 0.0,
            "tversky_loss": 0.0,
        }
        seen = 0
        for target, footprint, floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            floor_ids = floor_ids.to(device)
            features = features_for(backbone, footprint, floor_ids, vit_adapter)
            if pixel_decoder is not None:
                features, _levels = pixel_decoder(features)
            output = model(features)
            output_logits = F.interpolate(
                output["logits"],
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss, components = pixel_distribution_loss(
                output_logits,
                target,
                footprint,
                taxonomy,
                class_weights=class_weights,
                ce_weight=args.ce_weight,
                room_iou_weight=args.room_iou_weight,
                tversky_weight=args.tversky_weight,
            )
            if balancer is not None:
                loss_terms = {
                    "ce": components["ce_term"],
                    "room_iou": components["room_iou_term"],
                    "tversky": components["tversky_term"],
                }
                balancer.maybe_update(loss_terms, parameters, global_step)
                loss = balancer.weighted_loss(loss_terms)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            totals["loss"] += float(loss.detach()) * target.shape[0]
            totals["ce"] += float(components["pixel_distribution_ce"]) * target.shape[0]
            totals["soft_iou_loss"] += (
                float(components["room_soft_iou_loss"]) * target.shape[0]
            )
            totals["tversky_loss"] += (
                float(components["room_tversky_loss"]) * target.shape[0]
            )
            seen += target.shape[0]
        metrics = validate(
            model,
            backbone,
            val_loader,
            taxonomy,
            device,
            pixel_decoder=pixel_decoder,
            vit_adapter=vit_adapter,
        )
        record = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(1, seen),
            "train_ce": totals["ce"] / max(1, seen),
            "train_room_soft_iou_loss": (
                totals["soft_iou_loss"] / max(1, seen)
            ),
            "train_room_tversky_loss": (
                totals["tversky_loss"] / max(1, seen)
            ),
            "seconds": time.perf_counter() - started,
            **metrics,
            "condition_mode": "contour_only",
            "head_type": args.head_type,
            "dynamic_loss_weights": args.dynamic_loss_weights,
            "loss_weights": {
                "ce": args.ce_weight,
                "room_iou": args.room_iou_weight,
                "tversky": args.tversky_weight,
            },
        }
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": vars(args),
            "metrics": metrics,
            "vit_backbone": args.vit_backbone,
        }
        if pixel_decoder is not None:
            checkpoint["pixel_decoder"] = pixel_decoder.state_dict()
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["room_macro_iou"] > best:
            best = metrics["room_macro_iou"]
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
