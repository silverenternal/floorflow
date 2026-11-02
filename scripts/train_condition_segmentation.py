#!/usr/bin/env python3
"""Train a contour-conditioned structured floorplan decoder."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.data_balance import cached_class_presence_weights, make_weighted_sampler
from floorflow.data.dataset import CubiCasaFloorplanDataset, CubiCasaGeometryDataset
from floorflow.data.synthetic_data import SyntheticFloorplanDataset
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.condition_evaluation import evaluate
from floorflow.flow.condition_training import (
    build_condition_flow_sample,
    condition_velocity_loss,
)
from floorflow.heads.room_program import room_presence_target
from floorflow.heads.structured_objects import initialize_object_priors
from floorflow.objectives.condition_objective import (
    ConditionObjectiveWeights,
    condition_layout_loss,
)
from floorflow.objectives.loss_audit import (
    ObjectiveTerm,
    contribution_rows,
    gradient_contribution_rows,
)
from floorflow.recursion.inward_flow import multi_stage_weights
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.training.cli import parse_args
from floorflow.training.condition_model import build_condition_model
from floorflow.training.io import (
    checkpoint_has_module,
    memory_index_by_name,
    write_json,
)
from floorflow.training.optimizer import (
    freeze_except_hierarchy_heads,
    make_optimizer,
)
from floorflow.training.runtime_acceleration import RuntimeConfig, configure_torch_runtime
from floorflow.training.training import ModelEMA, WarmupCosineScheduler


def build_model(args: argparse.Namespace, taxonomy: FloorplanTaxonomy):
    return build_condition_model(args, taxonomy)

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    precision = args.precision or ("fp16" if args.amp else ("bf16" if device.type == "cuda" else "fp32"))
    runtime = RuntimeConfig(
        precision=precision,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        channels_last=args.channels_last,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    configure_torch_runtime(device)
    if args.geometry_cache_root is not None:
        train = CubiCasaGeometryDataset(
            args.data_root,
            "train",
            args.geometry_cache_root,
            augment=True,
            limit=args.limit_train,
        )
        val = CubiCasaGeometryDataset(
            args.data_root,
            "val",
            args.geometry_cache_root,
            augment=False,
            limit=args.limit_val,
        )
    elif args.synthetic_cache is not None:
        train = SyntheticFloorplanDataset(args.synthetic_cache, "train", augment=True)
        val = SyntheticFloorplanDataset(args.synthetic_cache, "val", augment=False)
    else:
        train = CubiCasaFloorplanDataset(
            args.data_root,
            "train",
            image_size=args.image_size,
            cache_root=args.cache_root,
            augment=True,
            limit=args.limit_train,
        )
        val = CubiCasaFloorplanDataset(
            args.data_root,
            "val",
            image_size=args.image_size,
            cache_root=args.cache_root,
            augment=False,
            limit=args.limit_val,
        )
    taxonomy = train.taxonomy
    retrieval_memory = None
    retrieval_lookup = {}
    if args.retrieval_prior_weight > 0:
        retrieval_memory = ContourPartitionMemory.from_cache(
            args.cache_root,
            top_k=args.retrieval_top_k,
            metric=args.retrieval_metric,
        )
        retrieval_memory.precompute_partitions(
            taxonomy,
            workers=args.retrieval_precompute_workers,
        )
        retrieval_lookup = memory_index_by_name(retrieval_memory, args.cache_root)
    positive_np = load_positive_weights(args.cache_root, taxonomy.num_target_channels, 20.0)
    positive_weights = torch.from_numpy(positive_np).to(device)
    room_weights = positive_weights[
        torch.tensor([class_id - 1 for class_id in taxonomy.room_ids], device=device)
    ]
    room_weights = room_weights / room_weights.mean()
    model = build_model(args, taxonomy)
    has_object_state = (
        args.object_initialize_from is not None
        or checkpoint_has_module(args.initialize_from, "structured_objects.")
    )
    if not has_object_state:
        frequency_path = args.cache_root / "class_frequency.json"
        if frequency_path.is_file():
            frequency = json.loads(frequency_path.read_text(encoding="utf-8"))
            object_priors = initialize_object_priors(
                model.structured_objects,
                frequency["pixel_counts"],
            )
            print(f"initialized object priors: {object_priors}", flush=True)
    model = runtime.prepare_model(model.to(device))
    if args.train_hierarchy_heads_only:
        freeze_except_hierarchy_heads(model)
    optimizer = make_optimizer(model, args)
    sampler = None
    if not args.disable_room_sampler:
        sample_weights = cached_class_presence_weights(
            train,
            taxonomy.room_ids,
            power=args.sampler_power,
        )
        sampler = make_weighted_sampler(sample_weights)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        **runtime.dataloader_options(device, args.workers),
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        **runtime.dataloader_options(device, args.workers),
    )
    updates_per_epoch = math.ceil(len(train_loader))
    scheduler = (
        WarmupCosineScheduler(
            optimizer,
            total_steps=max(1, updates_per_epoch * args.epochs),
            base_lr=args.head_lr,
            warmup_fraction=args.lr_warmup_fraction,
            min_lr_ratio=args.min_lr_ratio,
        )
        if args.lr_schedule == "cosine"
        else None
    )
    objective_weights = ConditionObjectiveWeights(
        room_boundary=args.room_boundary_weight,
        room_tversky=args.room_tversky_weight,
        room_group=args.room_group_weight,
        room_group_head=args.room_group_head_weight,
        room_program=args.room_program_weight,
        room_hierarchy=args.room_hierarchy_weight,
        room_refine=args.room_refine_weight,
        room_transport=args.room_transport_weight,
        room_polygon=args.room_polygon_weight,
        room_mask=args.room_mask_weight,
        room_mask_partition=args.room_mask_partition_weight,
        opening=args.opening_weight,
        furniture=args.furniture_weight,
        wall=args.wall_weight,
        geometry=args.geometry_weight,
        structural_consistency=args.structural_consistency_weight,
    )
    scaler = runtime.make_scaler(device)
    ema = None if args.disable_ema else ModelEMA(model, decay=args.ema_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_room = -1.0
    best_balanced = -math.inf
    best_hierarchy = -1.0
    partition_reference = -1.0
    global_step = 0
    for epoch in range(args.epochs):
        curriculum_progress = epoch / max(1, args.epochs - 1)
        stage_weights = multi_stage_weights(curriculum_progress)
        hierarchy_ready = (
            not args.enable_hierarchical_rooms
            or best_hierarchy >= partition_reference + args.hierarchy_injection_margin
        )
        injection_enabled = (
            not args.enable_hierarchical_rooms
            or (
                epoch >= max(0, args.hierarchy_injection_warmup_epochs)
                and hierarchy_ready
            )
        )
        if args.enable_hierarchical_rooms:
            injection_parameters = (
                "hierarchical_scale",
                "coarse_room_scale",
                "hierarchical_program_scale",
                "hierarchical_rooms.graph.output_scale",
                "hierarchical_rooms.refiner.residual_scale",
            )
            for name, parameter in model.named_parameters():
                if name in injection_parameters:
                    parameter.requires_grad_(injection_enabled)
        epoch_started = time.perf_counter()
        object_aux_scale = min(
            1.0,
            max(0.0, epoch / max(1, args.object_loss_warmup_epochs)),
        )
        room_stage = stage_weights.room if args.stage_curriculum else 1.0
        structure_stage = (
            stage_weights.structure if args.stage_curriculum else 1.0
        )
        detail_stage = stage_weights.detail if args.stage_curriculum else 1.0
        epoch_weights = ConditionObjectiveWeights(
            room_boundary=args.room_boundary_weight * room_stage,
            room_tversky=args.room_tversky_weight * room_stage,
            room_group=args.room_group_weight,
            room_group_head=args.room_group_head_weight,
            room_program=args.room_program_weight,
            room_hierarchy=args.room_hierarchy_weight,
            room_refine=args.room_refine_weight,
            room_transport=args.room_transport_weight,
            room_polygon=args.room_polygon_weight,
            room_mask=args.room_mask_weight,
            room_mask_partition=args.room_mask_partition_weight,
            opening=(
                args.opening_weight
                * object_aux_scale
                * detail_stage
            ),
            furniture=(
                args.furniture_weight
                * object_aux_scale
                * detail_stage
            ),
            wall=(
                args.wall_weight
                * object_aux_scale
                * structure_stage
            ),
            geometry=args.geometry_weight,
            structural_consistency=args.structural_consistency_weight,
        )
        model.train()
        running = {
            "loss": 0.0,
            "room_partition": 0.0,
            "room_group": 0.0,
            "room_group_head": 0.0,
            "room_boundary": 0.0,
            "room_program": 0.0,
            "room_hierarchy": 0.0,
            "room_polygon": 0.0,
            "condition_velocity": 0.0,
            "opening": 0.0,
            "furniture": 0.0,
            "wall": 0.0,
        }
        seen = 0
        for batch in train_loader:
            if len(batch) == 6:
                target, footprint, floor_ids, names, room_support, occlusion_mask = batch
                room_support = room_support.to(device, non_blocking=True)
            else:
                target, footprint, floor_ids, names = batch
                room_support = None
                occlusion_mask = None
            target = target.to(device, non_blocking=True)
            footprint = footprint.to(device, non_blocking=True)
            floor_ids = floor_ids.to(device, non_blocking=True)
            if args.condition_flow_weight > 0:
                flow_sample = build_condition_flow_sample(
                    target,
                    footprint,
                    channels=model.target_channels,
                    matcher=args.condition_flow_matcher,
                    time_sampling=args.condition_time_sampling,
                    noise_scale=args.condition_flow_noise_scale,
                    sigma=args.condition_flow_sigma,
                    inward_progress=(
                        curriculum_progress
                        if args.contour_inward_curriculum
                        else None
                    ),
                    inward_band_width=args.contour_band_width,
                )
                model_input = flow_sample.location
                model_time = flow_sample.time
            else:
                flow_sample = None
                model_input = torch.zeros(
                    target.shape[0],
                    model.target_channels,
                    *target.shape[-2:],
                    device=device,
                )
                model_time = torch.zeros(target.shape[0], device=device)
            room_prior_logits = None
            if retrieval_memory is not None and args.retrieval_prior_weight > 0:
                excluded = torch.tensor(
                    [retrieval_lookup[name] for name in names],
                    dtype=torch.long,
                )
                room_prior_logits = retrieval_memory.retrieve_logits(
                    footprint.detach().cpu().numpy(),
                    taxonomy,
                    device,
                    top_k=args.retrieval_top_k,
                    exclude_indices=excluded,
                )
            with runtime.autocast(device):
                room_program = (
                    room_presence_target(target, taxonomy)
                    if args.enable_program_conditioning
                    else None
                )
                velocity, aux = model.forward_with_aux(
                    model_input,
                    model_time,
                    footprint,
                    floor_ids=floor_ids,
                    detach_object_features=not args.share_backbone_with_objects,
                    layout_prior_logits=room_prior_logits,
                    room_program=room_program,
                )
                loss, components, terms = condition_layout_loss(
                    aux,
                    target,
                    footprint,
                    taxonomy,
                    room_class_weights=room_weights,
                    positive_class_weights=positive_weights,
                    weights=epoch_weights,
                    room_prior_logits=room_prior_logits,
                    room_prior_weight=args.retrieval_prior_weight,
                    room_support=room_support,
                    occlusion_mask=occlusion_mask,
                )
                if flow_sample is not None:
                    velocity_loss = condition_velocity_loss(
                        velocity,
                        flow_sample.velocity,
                        footprint,
                        weight_map=flow_sample.weight_map,
                    )
                    loss = (
                        loss
                        + float(args.condition_flow_weight) * velocity_loss
                    )
                    terms.append(
                        ObjectiveTerm(
                            "condition_velocity",
                            velocity_loss,
                            args.condition_flow_weight,
                        )
                    )
                    components["condition_velocity"] = velocity_loss.detach()
            if global_step < args.gradient_audit_steps:
                gradient_rows = gradient_contribution_rows(
                    terms,
                    model.parameters(),
                    primary="room_partition",
                )
                with (args.output_dir / "gradient_audit.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": global_step, "rows": gradient_rows}) + "\n")
            value_rows = contribution_rows(terms)
            with (args.output_dir / "loss_audit.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epoch, "step": global_step, "rows": value_rows}) + "\n")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            running["loss"] += float(loss.detach().cpu()) * target.shape[0]
            for key in running:
                if key != "loss" and key in components:
                    running[key] += float(components[key].cpu()) * target.shape[0]
            seen += target.shape[0]
        eval_model = model if ema is None else ema.module
        validate_now = (epoch + 1) % max(1, args.validate_every) == 0 or epoch == args.epochs - 1
        val_metrics = None
        rows = []
        if validate_now:
            val_metrics, rows = evaluate(
                eval_model,
                val_loader,
                taxonomy,
                device,
                room_weights,
                positive_weights,
                objective_weights,
                retrieval_memory,
                args.retrieval_prior_weight,
                args.retrieval_eval_top_k,
                runtime=runtime,
                flow_steps=args.condition_validation_steps,
                flow_noise_scale=args.condition_flow_noise_scale,
                program_conditioning=args.enable_program_conditioning,
            )
        train_seconds = time.perf_counter() - epoch_started
        record = {
            "epoch": epoch,
            **{f"train_{key}": value / max(1, seen) for key, value in running.items()},
            "train_seconds": train_seconds,
            "train_samples_per_second": seen / max(train_seconds, 1.0e-6),
            "validation_performed": validate_now,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "object_aux_scale": object_aux_scale,
            "ema_enabled": ema is not None,
            "runtime": runtime.manifest(),
            "hierarchy_injection_enabled": injection_enabled,
            "hierarchy_injection_ready": hierarchy_ready,
        }
        if val_metrics is not None:
            record["val_loss"] = val_metrics.pop("loss")
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if validate_now:
            write_json(args.output_dir / "val_per_case_latest.json", rows)
        checkpoint = {
            "model": eval_model.state_dict(),
            "raw_model": model.state_dict(),
            "ema_model": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "taxonomy": taxonomy.to_json(),
            "epoch": epoch,
            "val": record,
            "config": {
                "image_size": args.image_size,
                "base_channels": args.base_channels,
                "levels": args.levels,
                "synthetic_cache": (
                    None if args.synthetic_cache is None else str(args.synthetic_cache)
                ),
                "objective_mode": "condition_structured",
                "room_partition_semantic_prior": True,
                "hierarchical_rooms": args.enable_hierarchical_rooms,
                "hierarchical_hidden": args.hierarchical_hidden,
                "hierarchical_tokens": args.hierarchical_tokens,
                "hierarchical_coarse_resolution": args.hierarchical_coarse_resolution,
                "room_transport_weight": args.room_transport_weight,
                "room_refine_weight": args.room_refine_weight,
                "condition_flow_weight": args.condition_flow_weight,
                "condition_flow_noise_scale": args.condition_flow_noise_scale,
                "condition_flow_matcher": args.condition_flow_matcher,
                "condition_flow_sigma": args.condition_flow_sigma,
                "condition_time_sampling": args.condition_time_sampling,
                "flow_preset": args.flow_preset,
                "condition_validation_steps": args.condition_validation_steps,
                "room_mask_head": (
                    args.enable_room_mask_decoder
                    or args.room_mask_weight > 0
                    or args.room_mask_partition_weight > 0
                ),
                "room_mask_queries": args.room_mask_queries,
                "room_mask_resolution": args.room_mask_resolution,
                "polygon_loop_head": args.enable_polygon_loop,
                "polygon_queries": args.polygon_queries,
                "polygon_points": args.polygon_points,
                "polygon_hidden": args.polygon_hidden,
                "geometry_heads": args.enable_geometry_heads,
                "geometry_hidden": args.geometry_hidden,
                "geometry_experts": args.geometry_experts,
                "geometry_top_k": args.geometry_top_k,
                "family_streams": args.enable_family_streams,
                "family_mita_attention": args.enable_family_mita,
                "program_conditioning": args.enable_program_conditioning,
                "program_hidden": args.program_hidden,
                "geometry_weight": args.geometry_weight,
                "coarse_to_face_cfm": args.enable_coarse_to_face_cfm,
                "coarse_face_hidden": args.coarse_face_hidden,
                "coarse_face_max_faces": args.coarse_face_max_faces,
                "train_hierarchy_heads_only": args.train_hierarchy_heads_only,
                "hierarchy_injection_warmup_epochs": args.hierarchy_injection_warmup_epochs,
                "hierarchy_injection_margin": args.hierarchy_injection_margin,
                "selection_metric": args.selection_metric,
                "objective_weights": vars(objective_weights),
                "object_loss_warmup_epochs": args.object_loss_warmup_epochs,
                "object_backbone_gradient_isolation": not args.share_backbone_with_objects,
                "retrieval_prior_weight": args.retrieval_prior_weight,
                "retrieval_top_k": args.retrieval_top_k,
                "retrieval_eval_top_k": args.retrieval_eval_top_k,
                "retrieval_metric": args.retrieval_metric,
                "retrieval_memory_fingerprint": (
                    None if retrieval_memory is None else retrieval_memory.fingerprint
                ),
                "ema_decay": None if ema is None else args.ema_decay,
                "lr_schedule": args.lr_schedule,
                "runtime": runtime.manifest(),
            },
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_metrics is not None:
            selection_value = float(val_metrics.get(args.selection_metric, -math.inf))
            if selection_value > best_room:
                best_room = selection_value
                torch.save(checkpoint, args.output_dir / "best_room_iou.pt")
                torch.save(checkpoint, args.output_dir / "best.pt")
                write_json(args.output_dir / "best_room_iou_metrics.json", {"record": record, "cases": rows})
            if val_metrics.get("refined_room_iou", -1.0) > best_hierarchy:
                best_hierarchy = float(val_metrics["refined_room_iou"])
                torch.save(checkpoint, args.output_dir / "best_hierarchy_iou.pt")
            if (
                not injection_enabled
                and val_metrics["partition_room_macro_iou"] > partition_reference
            ):
                partition_reference = float(
                    val_metrics["partition_room_macro_iou"]
                )
            if val_metrics["selection_score"] > best_balanced:
                best_balanced = val_metrics["selection_score"]
                torch.save(checkpoint, args.output_dir / "best_balanced.pt")


if __name__ == "__main__":
    main()
