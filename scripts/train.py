#!/usr/bin/env python3
"""Train the boundary-conditioned floorplan flow model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.flow.cfm_backend import FLOW_MATCHERS
from floorflow.data.data_balance import class_presence_weights, make_weighted_sampler, read_dataset_labels
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.flow.core import flow_matching_loss
from floorflow.model.backbone import FloorplanFlowModel
from floorflow.training.training import ModelEMA, ScalarEMA, WarmupCosineScheduler, auxiliary_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/train/floorflow")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--category-loss-weight", type=float, default=0.20)
    parser.add_argument("--exterior-loss-weight", type=float, default=0.10)
    parser.add_argument("--exterior-band-width", type=int, default=7)
    parser.add_argument("--no-balanced-object-loss", action="store_true")
    parser.add_argument("--max-positive-weight", type=float, default=20.0)
    parser.add_argument("--time-sampling", choices=("uniform", "logit_normal"), default="uniform")
    parser.add_argument("--flow-matcher", choices=FLOW_MATCHERS, default="torchcfm")
    parser.add_argument("--flow-path-sigma", type=float, default=0.0)
    parser.add_argument("--objective-mode", choices=("velocity", "simplex"), default="velocity")
    parser.add_argument("--condition-dropout", type=float, default=0.10)
    parser.add_argument(
        "--disable-semantic-channel-attention",
        action="store_true",
        help="disable cross-channel semantic attention for ablation",
    )
    parser.add_argument("--disable-semantic-relation-bias", action="store_true")
    parser.add_argument(
        "--disable-slot-mixing",
        action="store_true",
        help="disable multiflow-style semantic slot decomposition and gated slot attention",
    )
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--disable-dynamic-output-head", action="store_true")
    parser.add_argument("--disable-topology-aux", action="store_true")
    parser.add_argument("--topology-aux-weight", type=float, default=0.10)
    parser.add_argument("--disable-semantic-endpoint-head", action="store_true")
    parser.add_argument("--semantic-head-weight", type=float, default=0.50)
    parser.add_argument("--direct-semantic-weight", type=float, default=0.50)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.10)
    parser.add_argument("--wall-loss-weight", type=float, default=0.10)
    parser.add_argument("--room-partition-weight", type=float, default=0.0)
    parser.add_argument("--enable-room-partition-head", action="store_true")
    parser.add_argument("--room-balanced-sampling", action="store_true")
    parser.add_argument("--layoutbridge-refinement", action="store_true")
    parser.add_argument("--layoutbridge-hidden", type=int, default=64)
    parser.add_argument("--frequency-low-weight", type=float, default=0.0)
    parser.add_argument("--frequency-high-weight", type=float, default=0.0)
    parser.add_argument("--room-mask-weight", type=float, default=0.0)
    parser.add_argument("--room-mask-queries", type=int, default=32)
    parser.add_argument("--room-mask-resolution", type=int, default=64)
    parser.add_argument("--disable-room-mask-partition-prior", action="store_true")
    parser.add_argument("--room-mask-partition-weight", type=float, default=0.0)
    parser.add_argument("--structured-object-weight", type=float, default=0.0)
    parser.add_argument("--semantic-anchor-weight", type=float, default=0.0)
    parser.add_argument("--aux-warmup-epochs", type=int, default=0)
    parser.add_argument("--backbone-lr-scale", type=float, default=1.0)
    parser.add_argument("--advanced-streams", action="store_true")
    parser.add_argument("--disable-fourier-conditioning", action="store_true")
    parser.add_argument("--output-kernels", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--noise-repeats", type=int, default=2)
    parser.add_argument("--lr-warmup-fraction", type=float, default=0.08)
    parser.add_argument("--min-lr-ratio", type=float, default=0.15)
    parser.add_argument("--loss-ema-decay", type=float, default=0.92)
    parser.add_argument("--model-ema-decay", type=float, default=0.999)
    parser.add_argument("--preload-data", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--initialize-from", type=Path, default=None, help="load compatible tensors from another checkpoint")
    parser.add_argument("--no-write-cache", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
    class_names: tuple[str, ...],
    taxonomy: dict,
    ema_model: torch.nn.Module | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "class_names": class_names,
            "taxonomy": taxonomy,
            "ema_model": ema_model.state_dict() if ema_model is not None else None,
            "config": vars(args),
        },
        path,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    taxonomy=None,
    aux_scale: float = 1.0,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device=device, non_blocking=True)
        footprint = footprint.to(device=device, non_blocking=True)
        floor_ids = floor_ids.to(device=device, non_blocking=True)
        loss, components = flow_matching_loss(
            model,
            target,
            footprint,
            category_loss_weight=args.category_loss_weight,
            time_sampling=args.time_sampling,
            exterior_loss_weight=args.exterior_loss_weight,
            exterior_band_width=args.exterior_band_width,
            balanced_object_loss=not args.no_balanced_object_loss,
            max_positive_weight=args.max_positive_weight,
            floor_ids=floor_ids,
            taxonomy=taxonomy,
            topology_aux_weight=args.topology_aux_weight,
            objective_mode=args.objective_mode,
            semantic_head_weight=args.semantic_head_weight,
            direct_semantic_weight=args.direct_semantic_weight,
            boundary_loss_weight=args.boundary_loss_weight,
            wall_loss_weight=args.wall_loss_weight,
            room_partition_weight=aux_scale * args.room_partition_weight,
            frequency_low_weight=aux_scale * args.frequency_low_weight,
            frequency_high_weight=aux_scale * args.frequency_high_weight,
            room_mask_weight=aux_scale * args.room_mask_weight,
            room_mask_partition_weight=aux_scale * args.room_mask_partition_weight,
            structured_object_weight=aux_scale * args.structured_object_weight,
            semantic_anchor_weight=aux_scale * args.semantic_anchor_weight,
            flow_matcher=args.flow_matcher,
            flow_path_sigma=args.flow_path_sigma,
        )
        del loss
        batch = target.shape[0]
        for key, value in components.items():
            totals.setdefault(key, 0.0)
            totals[key] += float(value.detach().cpu()) * batch
        count += batch
    return {key: value / max(1, count) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({**vars(args), "output_dir": str(args.output_dir)}, default=str, indent=2), flush=True)

    train_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=True,
        limit=args.limit_train,
        seed=args.seed,
        write_cache=not args.no_write_cache,
        preload=args.preload_data,
    )
    val_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_val,
        seed=args.seed,
        write_cache=not args.no_write_cache,
        preload=args.preload_data,
    )
    if train_dataset.taxonomy.class_names != val_dataset.taxonomy.class_names:
        raise ValueError("train/val taxonomy mismatch; rebuild the cache with one manifest")
    target_channels = train_dataset.taxonomy.num_target_channels
    class_names = train_dataset.taxonomy.class_names
    taxonomy_payload = train_dataset.taxonomy.to_json()
    positive_weights = torch.from_numpy(
        load_positive_weights(args.cache_root, target_channels, args.max_positive_weight)
    ).to(device)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    sampler = None
    if args.room_balanced_sampling:
        train_dataset.preload()
        sample_weights = class_presence_weights(
            read_dataset_labels(train_dataset),
            train_dataset.taxonomy.room_ids,
        )
        sampler = make_weighted_sampler(sample_weights)
    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
        **loader_args,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)

    model = FloorplanFlowModel(
        target_channels=target_channels,
        base_channels=args.base_channels,
        levels=args.levels,
        condition_dropout=args.condition_dropout,
        semantic_channel_attention=not args.disable_semantic_channel_attention,
        semantic_relation_bias=not args.disable_semantic_relation_bias,
        slot_mixing=not args.disable_slot_mixing,
        num_slots=args.num_slots,
        dynamic_output_head=not args.disable_dynamic_output_head,
        output_kernels=tuple(args.output_kernels),
        topology_aux=not args.disable_topology_aux,
        num_rooms=len(train_dataset.taxonomy.room_ids),
        semantic_endpoint_head=not args.disable_semantic_endpoint_head,
        channel_relations=train_dataset.taxonomy.channel_prior(),
        advanced_streams=args.advanced_streams,
        fourier_conditioning=not args.disable_fourier_conditioning,
        room_partition_head=args.enable_room_partition_head or args.room_partition_weight > 0,
        num_room_classes=len(train_dataset.taxonomy.room_ids),
        layoutbridge_refinement=args.layoutbridge_refinement,
        layoutbridge_hidden=args.layoutbridge_hidden,
        room_mask_head=args.room_mask_weight > 0 or args.room_mask_partition_weight > 0,
        room_mask_queries=args.room_mask_queries,
        room_mask_resolution=args.room_mask_resolution,
        room_mask_use_partition=not args.disable_room_mask_partition_prior,
        structured_object_head=args.structured_object_weight > 0,
        semantic_taxonomy=train_dataset.taxonomy,
        semantic_anchor_fusion=args.semantic_anchor_weight > 0,
    ).to(device)
    auxiliary_prefixes = (
        "room_partition.",
        "room_mask_decoder.",
        "structured_objects.",
        "semantic_anchor_fusion.",
        "dual_stream_injection.",
        "anisotropic_velocity.",
        "layout_stream_mixer.",
        "spatial_velocity.",
    )
    auxiliary_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        target = auxiliary_parameters if name.startswith(auxiliary_prefixes) else backbone_parameters
        target.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.lr * args.backbone_lr_scale},
            {"params": auxiliary_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=max(1, updates_per_epoch * args.epochs),
        base_lr=args.lr,
        warmup_fraction=args.lr_warmup_fraction,
        min_lr_ratio=args.min_lr_ratio,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    ema_model = ModelEMA(model, decay=args.model_ema_decay)
    loss_ema = ScalarEMA(decay=args.loss_ema_decay)
    start_epoch = 0
    best_val = math.inf
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("ema_model") is not None:
            ema_model.module.load_state_dict(checkpoint["ema_model"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val = float(checkpoint.get("best_val", math.inf))
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)
    elif args.initialize_from is not None:
        source = torch.load(args.initialize_from, map_location="cpu", weights_only=False)["model"]
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in source.items()
            if key in current and current[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
        print(
            f"initialized {len(compatible)}/{len(current)} tensors from {args.initialize_from} "
            "with the remaining heads random",
            flush=True,
        )

    log_path = args.output_dir / "train.jsonl"
    accumulation_steps = max(1, args.gradient_accumulation_steps)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        started = time.time()
        aux_scale = auxiliary_weight(1.0, epoch, args.aux_warmup_epochs)
        running = {"loss": 0.0, "velocity_loss": 0.0, "category_loss": 0.0}
        seen = 0
        accumulated = 0
        current_lr = scheduler.lr_at(scheduler.step_index + 1)
        optimizer.zero_grad(set_to_none=True)
        for repeat in range(max(1, args.noise_repeats)):
            for step, (target, footprint, floor_ids, _names) in enumerate(train_loader):
                target = target.to(device=device, non_blocking=True)
                footprint = footprint.to(device=device, non_blocking=True)
                floor_ids = floor_ids.to(device=device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=args.amp and device.type == "cuda",
                ):
                    loss, components = flow_matching_loss(
                        model,
                        target,
                        footprint,
                        category_loss_weight=args.category_loss_weight,
                        time_sampling=args.time_sampling,
                        exterior_loss_weight=args.exterior_loss_weight,
                        exterior_band_width=args.exterior_band_width,
                        balanced_object_loss=not args.no_balanced_object_loss,
                        max_positive_weight=args.max_positive_weight,
                        positive_class_weights=positive_weights,
                        floor_ids=floor_ids,
                        taxonomy=train_dataset.taxonomy,
                        topology_aux_weight=args.topology_aux_weight,
                        objective_mode=args.objective_mode,
                        semantic_head_weight=args.semantic_head_weight,
                        direct_semantic_weight=args.direct_semantic_weight,
                        boundary_loss_weight=args.boundary_loss_weight,
                        wall_loss_weight=args.wall_loss_weight,
                        room_partition_weight=aux_scale * args.room_partition_weight,
                        frequency_low_weight=aux_scale * args.frequency_low_weight,
                        frequency_high_weight=aux_scale * args.frequency_high_weight,
                        room_mask_weight=aux_scale * args.room_mask_weight,
                        room_mask_partition_weight=aux_scale * args.room_mask_partition_weight,
                        structured_object_weight=aux_scale * args.structured_object_weight,
                        semantic_anchor_weight=aux_scale * args.semantic_anchor_weight,
                        flow_matcher=args.flow_matcher,
                        flow_path_sigma=args.flow_path_sigma,
                    )
                scaler.scale(loss / accumulation_steps).backward()
                accumulated += 1
                is_last = repeat == max(1, args.noise_repeats) - 1 and step == len(train_loader) - 1
                if accumulated >= accumulation_steps or is_last:
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    current_lr = scheduler.step()
                    ema_model.update(model)
                    accumulated = 0
                batch = target.shape[0]
                loss_value = float(loss.detach().cpu())
                ema_value = loss_ema.update(loss_value)
                for key in components:
                    running.setdefault(key, 0.0)
                for key in running:
                    value = components.get(key)
                    if value is not None:
                        running[key] += float(value.detach().cpu()) * batch
                seen += batch
                if step % 20 == 0:
                    print(
                        f"epoch={epoch} repeat={repeat} step={step}/{len(train_loader)} "
                        f"loss={running['loss'] / max(1, seen):.5f} "
                        f"ema={ema_value:.5f} lr={current_lr:.2e}",
                        flush=True,
                    )
        train_metrics = {key: value / max(1, seen) for key, value in running.items()}
        val_metrics = validate(
            model,
            val_loader,
            device,
            args,
            taxonomy=train_dataset.taxonomy,
            aux_scale=1.0,
        )
        elapsed = time.time() - started
        record = {
            "epoch": epoch,
            "seconds": elapsed,
            "samples_per_second": seen / max(elapsed, 1.0e-6),
            "lr": current_lr,
            "aux_scale": aux_scale,
            "loss_ema": loss_ema.value,
            "train": train_metrics,
            "val": val_metrics,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            best_val=best_val,
            args=args,
            class_names=class_names,
            taxonomy=taxonomy_payload,
            ema_model=ema_model.module,
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                best_val=best_val,
                args=args,
                class_names=class_names,
                taxonomy=taxonomy_payload,
                ema_model=ema_model.module,
            )
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            torch.save(
                {
                    "model": ema_model.state_dict(),
                    "class_names": class_names,
                    "taxonomy": taxonomy_payload,
                    "config": vars(args),
                    "epoch": epoch,
                    "best_val": best_val,
                },
                args.output_dir / "ema.pt",
            )


if __name__ == "__main__":
    main()
