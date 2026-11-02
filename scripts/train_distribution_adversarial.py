#!/usr/bin/env python3
"""Adversarially adapt the contour-conditioned pixel distribution head."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.data.dynamic_curriculum import (
    DataDynamicsTracker,
    data_map_weights,
    load_data_map,
    write_data_map,
)
from floorflow.data.stratified_sampling import (
    CurriculumStratifiedSampler,
    EpochRandomSubsetSampler,
    StratifiedWeightDataset,
    load_floorplan_strata,
)
from floorflow.distribution.adversarial import (
    DistributionAdversarialTrainer,
    DynamicFeatureMaskDiscriminator,
)
from floorflow.distribution.checkpoint import (
    restore_distribution_model,
    save_distribution_model,
)
from floorflow.heads.pixel_distribution import pixel_distribution_loss
from floorflow.heads.room_partition import (
    batch_room_partition_target,
    room_partition_loss,
)
from floorflow.training.distribution_runtime import set_training_seed
from floorflow.training.distribution_validation import (
    validate_distribution_room_iou,
)
from floorflow.training.training import ModelEMA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dynamic-head-variant",
        choices=("fixed", "mixture", "generated", "query"),
        default=None,
    )
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
    parser.add_argument("--limit-train", type=int, default=1024)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--backbone-lr", type=float, default=1.0e-6)
    parser.add_argument("--class-weight-cap", type=float, default=20.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--partition-loss-weight", type=float, default=0.0)
    parser.add_argument("--partition-lr", type=float, default=1.0e-4)
    parser.add_argument("--adversarial-lr", type=float, default=1.0e-4)
    parser.add_argument("--adversarial-weight", type=float, default=0.02)
    parser.add_argument("--r1-weight", type=float, default=0.01)
    parser.add_argument("--critic-interval", type=int, default=4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="cosine",
    )
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--selection", choices=("raw", "ema"), default="raw")
    parser.add_argument("--stratified-index", type=Path, default=None)
    parser.add_argument("--sample-weight-index", type=Path, default=None)
    parser.add_argument("--sampler-samples", type=int, default=None)
    parser.add_argument("--rare-boost", type=float, default=2.0)
    parser.add_argument("--curriculum-strength", type=float, default=1.0)
    parser.add_argument("--data-map-strength", type=float, default=0.0)
    parser.add_argument("--data-map-warmup-epochs", type=int, default=1)
    parser.add_argument("--data-map-floor", type=float, default=0.75)
    parser.add_argument("--data-map-ceiling", type=float, default=1.5)
    parser.add_argument("--data-map-task-focus", type=float, default=1.0)
    parser.add_argument("--data-map-index", type=Path, default=None)
    parser.add_argument(
        "--data-map-guard",
        choices=("none", "sparse"),
        default="none",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def real_distribution(target, channels):
    one_hot = F.one_hot(
        target.long(),
        num_classes=channels + 1,
    ).permute(0, 3, 1, 2).float()
    return one_hot[:, 1:]


def main():
    args = parse_args()
    set_training_seed(args.seed)
    device = torch.device(args.device)
    model, taxonomy, _ = restore_distribution_model(
        args.checkpoint,
        device,
        dynamic_head_variant=args.dynamic_head_variant,
    )
    ema = ModelEMA(model, decay=args.ema_decay)
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
    base_train = train.base if isinstance(train, StratifiedWeightDataset) else train
    sample_names = [
        f"{sample.relative_case}__{sample.floor_id}"
        for sample in base_train.samples
    ]
    data_tracker = DataDynamicsTracker(sample_names, taxonomy)
    dynamic_weights = None
    dynamic_eligible = None
    if args.data_map_guard == "sparse":
        if args.sample_weight_index is None:
            raise ValueError("sparse data-map guard requires --sample-weight-index")
        guarded_entries = load_floorplan_strata(
            args.sample_weight_index,
            "train",
        )
        dynamic_eligible = {
            entry.name.replace("/", "__", 1): (
                entry.room_count_bin > 0 and entry.instance_count_bin > 1
            )
            for entry in guarded_entries
        }
    if args.data_map_index is not None:
        dynamic_weights = data_map_weights(
            load_data_map(args.data_map_index),
            strength=args.data_map_strength,
            floor=args.data_map_floor,
            ceiling=args.data_map_ceiling,
            task_focus=args.data_map_task_focus,
            eligible=dynamic_eligible,
        )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    if args.sample_weight_index is not None:
        entries = load_floorplan_strata(args.sample_weight_index, "train")
        by_name = {entry.name: entry for entry in entries}
        selected = [
            by_name[f"{sample.relative_case}/{sample.floor_id}"]
            for sample in train.samples
        ]
        train = StratifiedWeightDataset(train, selected)
    sampler = None
    if args.stratified_index is not None:
        entries = load_floorplan_strata(args.stratified_index, "train")
        by_name = {entry.name: entry for entry in entries}
        selected = [
            by_name[f"{sample.relative_case}/{sample.floor_id}"]
            for sample in train.samples
        ]
        sampler = CurriculumStratifiedSampler(
            selected,
            num_samples=args.sampler_samples or len(selected),
            rare_boost=args.rare_boost,
            curriculum_strength=args.curriculum_strength,
        )
        train_loader = DataLoader(train, sampler=sampler, **loader_args)
    elif args.sampler_samples:
        sampler = EpochRandomSubsetSampler(
            len(train),
            args.sampler_samples,
            seed=0,
        )
        train_loader = DataLoader(train, sampler=sampler, **loader_args)
    else:
        train_loader = DataLoader(train, shuffle=True, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    class_weights = torch.cat(
        [
            torch.ones(1, device=device),
            torch.from_numpy(
                load_positive_weights(
                    args.cache_root,
                    taxonomy.num_target_channels,
                    args.class_weight_cap,
                )
            ).to(device),
        ]
    )
    discriminator = DynamicFeatureMaskDiscriminator(
        model.target_channels,
        mask_ratio=args.mask_ratio,
    ).to(device)
    adversarial = DistributionAdversarialTrainer(
        discriminator,
        lr=args.adversarial_lr,
        generator_weight=args.adversarial_weight,
        r1_weight=args.r1_weight,
    )
    head_parameters = [
        parameter
        for name, parameter in model.hub.pixel.named_parameters()
        if parameter.requires_grad
    ]
    backbone_parameters = []
    partition_parameters = []
    if args.unfreeze_backbone:
        model.base.requires_grad_(True)
        backbone_parameters = [
            parameter
            for parameter in model.base.parameters()
            if parameter.requires_grad
        ]
    else:
        model.base.requires_grad_(False)
        if args.partition_loss_weight > 0:
            if not getattr(model.base, "use_room_partition_head", False):
                raise ValueError(
                    "partition loss requires a checkpoint room-partition head"
                )
            model.base.room_partition.requires_grad_(True)
            partition_parameters = list(model.base.room_partition.parameters())
    parameter_groups = [
        {"params": head_parameters, "lr": args.lr},
    ]
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": args.backbone_lr}
        )
    if partition_parameters and not backbone_parameters:
        parameter_groups.append(
            {"params": partition_parameters, "lr": args.partition_lr}
        )
    parameters = head_parameters + backbone_parameters + partition_parameters
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=1.0e-4,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.lr * 0.05,
        )
        if args.lr_schedule == "cosine"
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        if sampler is not None:
            if isinstance(sampler, CurriculumStratifiedSampler):
                sampler.set_epoch(epoch, args.epochs)
            else:
                sampler.set_epoch(epoch)
        started = time.perf_counter()
        model.train()
        discriminator.train()
        totals = {
            "loss": 0.0,
            "pixel": 0.0,
            "partition": 0.0,
            "generator_adversarial": 0.0,
        }
        seen = 0
        for step, batch in enumerate(train_loader):
            sample_weights = batch[4] if len(batch) == 5 else None
            target, footprint, floor_ids, _names = batch[:4]
            target = target.to(device, non_blocking=True)
            footprint = footprint[:, None].to(device, non_blocking=True)
            floor_ids = floor_ids.to(device, non_blocking=True)
            if dynamic_weights is not None:
                dynamic_batch_weights = torch.tensor(
                    [dynamic_weights.get(name, 1.0) for name in _names],
                    device=device,
                    dtype=torch.float32,
                )
                sample_weights = (
                    dynamic_batch_weights
                    if sample_weights is None
                    else sample_weights.to(device) * dynamic_batch_weights
                )
            style_ids = (
                torch.arange(target.shape[0], device=device)
                + step * target.shape[0]
            ) % max(1, model.num_styles)
            initial = model.contour_initial_state(
                footprint,
                floor_ids=floor_ids,
                style_ids=style_ids,
            )
            fake = initial["pixel"]["probability"][:, 1:]
            data_tracker.update(
                _names,
                initial["pixel"]["logits"],
                target,
                footprint,
            )
            pixel_loss, _ = pixel_distribution_loss(
                initial["pixel"]["logits"],
                target,
                footprint,
                taxonomy,
                class_weights=class_weights,
                room_iou_weight=0.5,
                tversky_weight=0.2,
                sample_weights=sample_weights,
            )
            partition_loss = pixel_loss.new_zeros(())
            if args.partition_loss_weight > 0:
                partition_target = batch_room_partition_target(
                    target,
                    footprint[:, 0],
                    taxonomy,
                )
                room_class_weights = class_weights[
                    torch.tensor(
                        taxonomy.room_ids,
                        device=device,
                        dtype=torch.long,
                    )
                ]
                room_class_weights = room_class_weights / room_class_weights.mean()
                partition_loss = room_partition_loss(
                    initial["room_partition_logits"],
                    partition_target,
                    class_weights=room_class_weights,
                    dice_weight=0.5,
                    tversky_weight=0.2,
                )["loss"]
            adversarial_loss, adversarial_components = (
                adversarial.generator_loss(fake, footprint)
            )
            loss = (
                pixel_loss
                + float(args.partition_loss_weight) * partition_loss
                + adversarial_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            ema.update(model)
            if step % max(1, args.critic_interval) == 0:
                discriminator_metrics = adversarial.discriminator_step(
                    real_distribution(target, model.target_channels),
                    fake.detach(),
                    footprint,
                )
            totals["loss"] += float(loss.detach()) * target.shape[0]
            totals["pixel"] += float(pixel_loss.detach()) * target.shape[0]
            totals["partition"] += float(partition_loss.detach()) * target.shape[0]
            totals["generator_adversarial"] += float(
                adversarial_loss.detach()
            ) * target.shape[0]
            seen += target.shape[0]
            if step % 25 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            **{
                                key: float(value.detach())
                                for key, value in {
                                    **adversarial_components,
                                    **discriminator_metrics,
                                }.items()
                            },
                        }
                    ),
                    flush=True,
                )
        if scheduler is not None:
            scheduler.step()
        selected_model = ema.module if args.selection == "ema" else model
        validation = validate_distribution_room_iou(
            selected_model,
            val_loader,
            taxonomy,
            device,
            default_style_id=0,
        )
        data_map_summary = write_data_map(
            data_tracker.snapshot(),
            args.output_dir / f"data_map_epoch_{epoch}.jsonl",
            epoch=epoch,
        )
        if epoch + 1 >= args.data_map_warmup_epochs:
            dynamic_weights = data_map_weights(
                data_tracker.snapshot(),
                strength=args.data_map_strength,
                floor=args.data_map_floor,
                ceiling=args.data_map_ceiling,
                task_focus=args.data_map_task_focus,
                eligible=dynamic_eligible,
            )
        record = {
            "epoch": epoch,
            **{
                f"train_{key}": value / max(1, seen)
                for key, value in totals.items()
            },
            **{
                f"adversarial_{key}": float(value.detach())
                for key, value in discriminator_metrics.items()
            },
            "seconds": time.perf_counter() - started,
            "data_map": data_map_summary,
            **validation,
        }
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {"training_args": vars(args), "adversarial_metrics": record}
        save_distribution_model(
            model,
            args.output_dir / "last.pt",
            taxonomy=taxonomy,
            extra=payload,
        )
        save_distribution_model(
            ema.module,
            args.output_dir / "ema_last.pt",
            taxonomy=taxonomy,
            extra=payload,
        )
        torch.save(
            adversarial.state_dict(),
            args.output_dir / "adversarial_last.pt",
        )
        if validation["room_macro_iou"] > best:
            best = validation["room_macro_iou"]
            save_distribution_model(
                selected_model,
                args.output_dir / "best.pt",
                taxonomy=taxonomy,
                extra=payload,
            )
            torch.save(
                adversarial.state_dict(),
                args.output_dir / "adversarial_best.pt",
            )


if __name__ == "__main__":
    main()
