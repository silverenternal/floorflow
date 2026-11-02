#!/usr/bin/env python3
"""Jointly train contour→distribution→CFM with no target-derived controls."""

from __future__ import annotations

import json
import time

import torch
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.distribution.cfm_model import DistributionCFMModel
from floorflow.distribution.checkpoint import (
    restore_distribution_model,
    save_distribution_model,
)
from floorflow.distribution.element_loss import element_set_loss
from floorflow.distribution.element_targets import element_targets
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.joint_loss import joint_distribution_cfm_loss
from floorflow.distribution.style_semantic import (
    auxiliary_style_losses,
    load_style_semantic_prototypes,
)
from floorflow.flow.masked_control import sample_control_mask
from floorflow.flow.presets import resolve_flow_preset
from floorflow.model.cfm_structure_adapter import enable_cfm_structure
from floorflow.objectives.gradient_balancer import GradientNormBalancer
from floorflow.training.cfm_convergence import (
    CFMConvergenceConfig,
    effective_updates,
    make_joint_ema,
    make_joint_optimizer,
    make_joint_scheduler,
    make_style_optimizer,
)
from floorflow.training.checkpoint import (
    load_checkpoint_model,
    preserve_checkpoint_num_styles,
)
from floorflow.training.distribution_cli import parse_distribution_args
from floorflow.training.distribution_data import distribution_dataset
from floorflow.training.distribution_manifest import build_joint_run_manifest
from floorflow.training.distribution_runtime import (
    apply_style_dropout,
    configure_flow_args,
    distribution_query_counts,
    resolve_style_assignments,
    set_training_seed,
    style_ids_for_names,
)
from floorflow.training.distribution_validation import (
    validate_distribution_room_iou,
)
from floorflow.training.gradient_health import clip_gradients_fail_closed
from floorflow.training.loss_metric_guard import LossMetricGuard
from floorflow.training.runtime_acceleration import (
    RuntimeConfig,
    configure_torch_runtime,
)
from floorflow.training.training import ScalarEMA


def main() -> None:
    args = parse_distribution_args()
    if args.style_dropout < 0.0 or args.style_dropout >= 1.0: raise SystemExit("--style-dropout must be in [0, 1)")
    if args.style_orthogonality_weight < 0.0: raise SystemExit("--style-orthogonality-weight must be non-negative")
    preset = resolve_flow_preset(args.flow_preset)
    style_assignments = resolve_style_assignments(args)
    style_prototypes = None
    if args.resume is None and args.initialize_distribution is None and args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --resume or --initialize-distribution is set")
    backend_version = configure_flow_args(args, preset)
    set_training_seed(args.seed)
    device = torch.device(args.device)
    if args.style_manifest is not None:
        style_prototypes = load_style_semantic_prototypes(
            args.style_manifest,
            device,
        )
    configure_torch_runtime(device)
    runtime = RuntimeConfig.from_args(args)
    if args.resume is not None:
        model, taxonomy, payload = restore_distribution_model(
            args.resume,
            device,
            hierarchy_support_strength=args.hierarchy_support_strength,
            bridge_mode=args.bridge_mode,
        )
        base = model.base
        config = {"image_size": int(payload.get("training_args", {}).get("image_size", 256))}
        if "training_state" not in payload:
            raise SystemExit("resume checkpoint has no optimizer/scheduler/EMA state")
    elif args.initialize_distribution is not None:
        args.num_styles = preserve_checkpoint_num_styles(
            args.initialize_distribution,
            args.num_styles,
        )
        model, taxonomy, payload = restore_distribution_model(
            args.initialize_distribution, device,
            num_styles=args.num_styles, style_rank=args.style_rank,
            hierarchy_support_strength=args.hierarchy_support_strength,
            element_blend_init=args.element_blend_init,
            element_feedback_init=args.element_feedback_init,
            dynamic_head_variant=args.dynamic_head_variant,
            dynamic_element_queries=args.dynamic_element_queries,
            bridge_mode=args.bridge_mode,
        )
        base = model.base
        config = {"image_size": int(payload.get("training_args", {}).get("image_size", 256))}
    else:
        base, config, taxonomy, _ = load_checkpoint_model(
            args.checkpoint,
            device,
            taxonomy=None,
        )
        model = DistributionCFMModel(
            base,
            ChannelHierarchy.from_taxonomy(taxonomy),
            hidden_channels=args.hidden_channels,
            pixel_grid=args.pixel_grid,
            element_grid=args.element_grid,
            queries_per_group=distribution_query_counts(args.queries),
            element_mask_version=args.element_mask_version,
            num_styles=args.num_styles,
            style_rank=args.style_rank,
            hierarchy_support_strength=float(
                args.hierarchy_support_strength or 0.0
            ),
            element_blend_init=float(args.element_blend_init or 0.0),
            element_feedback_init=float(args.element_feedback_init or 0.0),
            dynamic_head_variant=args.dynamic_head_variant,
            bridge_mode=args.bridge_mode,
            dynamic_element_queries=args.dynamic_element_queries,
        ).to(device)
    if args.cfm_structure_fusion:
        enable_cfm_structure(
            model.base,
            taxonomy,
            static_condition_encoder=args.static_condition_encoder,
            typed_family_streams=args.typed_family_streams,
            multiscale_structure_injection=(
                args.multiscale_structure_injection
            ),
        )
    model = runtime.prepare_model(model)
    for parameter in model.hub.element.parent_reliability.blend.values():
        parameter.data.fill_(float(args.parent_reliability_init))
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    train = distribution_dataset(
        args,
        "train",
        taxonomy,
        image_size=int(config.get("image_size", 256)),
        augment=True,
        limit=args.limit_train,
    )
    val = distribution_dataset(
        args,
        "val",
        taxonomy,
        image_size=int(config.get("image_size", 256)),
        augment=False,
        limit=args.limit_val,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = runtime.dataloader_options(device, args.workers)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    positive = torch.from_numpy(
        load_positive_weights(
            args.cache_root,
            taxonomy.num_target_channels,
            args.positive_weight_clip,
        )
    ).to(device)
    class_weights = torch.cat([torch.ones(1, device=device), positive])
    updates_per_epoch = effective_updates(len(train_loader), args.gradient_accumulation_steps)
    joint_updates = max(1, int(args.epochs * updates_per_epoch))
    total_updates = max(
        1,
        args.max_steps
        if args.max_steps > 0
        else args.element_warmup_updates + joint_updates,
    )
    scheduled_joint_updates = max(1, total_updates - args.element_warmup_updates)
    warmup_fraction = (
        0.0
        if args.lr_schedule == "constant"
        else min(0.5, args.warmup_steps / scheduled_joint_updates)
    )
    convergence = CFMConvergenceConfig(
        base_lr=args.base_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_fraction=warmup_fraction,
        min_lr_ratio=(
            1.0 if args.lr_schedule == "constant" else args.min_lr_ratio
        ),
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        fused_adamw=args.fused_adamw,
    )
    optimizer = make_joint_optimizer(model, convergence)
    style_optimizer = make_style_optimizer(
        model,
        args.style_lr,
        args.weight_decay,
    )
    balancer = GradientNormBalancer(
        {
            "flow": args.flow_weight,
            "pixel": args.pixel_weight,
            "element": args.element_weight,
            "coupling": args.coupling_weight,
            "source": args.source_weight,
        },
        reference="flow",
        update_interval=args.gradient_balance_interval,
        minimum_scale=args.gradient_balance_min_scale,
        maximum_scale=args.gradient_balance_max_scale,
    )
    element_optimizer = torch.optim.AdamW(
        [
            parameter
            for name, parameter in model.hub.element.named_parameters()
            if "style_adapter" not in name
        ],
        lr=args.element_warmup_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = make_joint_scheduler(optimizer, convergence, scheduled_joint_updates)
    scaler = runtime.make_scaler(device)
    ema = make_joint_ema(model, convergence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    metric_guard = LossMetricGuard(
        patience=args.loss_metric_patience,
        loss_drop=args.loss_metric_drop,
        metric_drop=args.validation_metric_drop,
    )
    global_step = 0
    start_epoch = 0
    loss_ema = ScalarEMA(decay=0.92)
    if args.resume is not None:
        state = payload["training_state"]
        optimizer.load_state_dict(state["optimizer"])
        if "element_optimizer" in state:
            element_optimizer.load_state_dict(state["element_optimizer"])
        if style_optimizer is not None and "style_optimizer" in state:
            style_optimizer.load_state_dict(state["style_optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        ema.module.load_state_dict(state["ema_model"])
        start_epoch = int(state.get("epoch", -1)) + 1
        global_step = int(state.get("global_step", 0))
        best = float(state.get("best", -1.0))
        loss_ema.value = state.get("loss_ema")
        if "gradient_balancer" in state:
            balancer.load_state_dict(state["gradient_balancer"])
        if "metric_guard" in state:
            metric_guard.load_state_dict(state["metric_guard"])
    scheduler.total_steps = scheduled_joint_updates
    scheduler.warmup_steps = min(
        scheduler.total_steps,
        round(scheduler.total_steps * convergence.warmup_fraction),
    )
    scheduler.apply(scheduler.step_index + 1)
    manifest = build_joint_run_manifest(
        args=args,
        model=model,
        runtime=runtime,
        convergence=convergence,
        flow_preset=preset.name,
        flow_backend=(
            "legacy" if args.flow_matcher == "legacy" else "torchcfm"
        ),
        flow_backend_version=str(backend_version),
        batches_per_epoch=len(train_loader),
        updates_per_epoch=updates_per_epoch,
        total_updates=total_updates,
        joint_updates=scheduled_joint_updates,
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    epoch = start_epoch
    while global_step < total_updates:
        started = time.perf_counter()
        model.train()
        totals = {"loss": 0.0, "flow": 0.0, "pixel": 0.0, "element": 0.0}
        seen = 0
        accumulated = 0
        optimizer.zero_grad(set_to_none=True)
        if style_optimizer is not None:
            style_optimizer.zero_grad(set_to_none=True)
        stop_training = False
        for target, footprint, floor_ids, _names in train_loader:
            target = target.to(device, non_blocking=True)
            footprint = footprint[:, None].to(device, non_blocking=True)
            floor_ids = floor_ids.to(device, non_blocking=True)
            style_ids = (
                style_ids_for_names(_names, style_assignments["train"], device)
                if style_assignments else None
            )
            style_ids = apply_style_dropout(
                style_ids,
                args.style_dropout,
                generator=torch.Generator(device=device).manual_seed(
                    args.seed + global_step
                ),
            )
            initial = model.contour_initial_state(
                footprint,
                floor_ids=floor_ids,
                style_ids=style_ids,
            )
            if global_step < args.element_warmup_updates:
                targets = element_targets(
                    target,
                    taxonomy,
                    hierarchy,
                    max_queries=distribution_query_counts(args.queries),
                )
                with runtime.autocast(device):
                    loss, components = element_set_loss(
                        initial["elements"],
                        targets,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite element warmup loss at update {global_step}"
                    )
                element_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if convergence.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.hub.element.parameters(),
                        convergence.grad_clip,
                    )
                element_optimizer.step()
                ema.update(model)
                global_step += 1
                loss_ema.update(float(loss.detach()))
                batch = target.shape[0]
                seen += batch
                totals["loss"] += float(loss.detach()) * batch
                totals["element"] += float(
                    components["element_set_loss"]
                ) * batch
                if global_step >= total_updates:
                    stop_training = True
                    break
                continue
            source = initial["cfm"]["state"]
            if args.dynamic_source_mask:
                mask = sample_control_mask(
                    source,
                    footprint,
                    taxonomy,
                    ratio=None,
                    generator=torch.Generator(device=device).manual_seed(
                        args.seed + global_step
                    ),
                )
                source = torch.where(
                    mask.value,
                    torch.zeros_like(source),
                    source,
                )
            with runtime.autocast(device):
                loss, components = joint_distribution_cfm_loss(
                    model,
                    target,
                    source,
                    footprint,
                    taxonomy,
                    hierarchy,
                    floor_ids=floor_ids,
                    queries_per_group=distribution_query_counts(args.queries),
                    class_weights=class_weights,
                    flow_weight=args.flow_weight * balancer.scales["flow"],
                    pixel_weight=args.pixel_weight * balancer.scales["pixel"],
                    element_weight=args.element_weight * balancer.scales["element"],
                    coupling_weight=args.coupling_weight * balancer.scales["coupling"],
                    source_weight=args.source_weight * balancer.scales["source"],
                    family_iou_weight=args.family_iou_weight,
                    room_iou_weight=args.room_iou_weight,
                    tversky_weight=args.tversky_weight,
                    boundary_weight=args.boundary_weight,
                    structural_weight=args.structural_weight,
                    topology_closure_weight=args.topology_closure_weight,
                    dynamic_route_weight=args.dynamic_route_weight,
                    flow_endpoint_weight=args.flow_endpoint_weight,
                    flow_change_weight=args.flow_change_weight,
                    contour_output=initial,
                    flow_matcher=args.flow_matcher,
                    time_sampling=args.time_sampling,
                    sigma=args.flow_sigma,
                    time_sampling_mean=args.time_sampling_mean,
                    time_sampling_std=args.time_sampling_std,
                    minibatch_ot=args.minibatch_ot,
                    ot_token_grid=args.ot_token_grid,
                    style_ids=style_ids,
                )
            style_loss, style_components = auxiliary_style_losses(
                model,
                initial,
                hierarchy,
                footprint,
                style_ids,
                prototypes=style_prototypes,
                contrast_weight=args.style_contrast_weight,
                contrast_margin=args.style_contrast_margin,
                semantic_weight=args.style_semantic_weight,
                orthogonality_weight=args.style_orthogonality_weight,
            )
            loss = loss + style_loss
            components.update(style_components)
            components["joint_total_loss"] = loss.detach()
            balancer.maybe_update(
                {
                    "flow": components["_flow_term"],
                    "pixel": components["_pixel_term"],
                    "element": components["_element_term"],
                    "coupling": components["_coupling_term"],
                    "source": components["_source_term"],
                },
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                global_step,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite joint loss at epoch {epoch}, update {global_step}"
                )
            scaler.scale(
                loss / convergence.gradient_accumulation_steps
            ).backward()
            accumulated += 1
            batch = target.shape[0]
            seen += batch
            for key, name in (
                ("joint_total_loss", "loss"),
                ("joint_flow_loss", "flow"),
                ("joint_pixel_loss", "pixel"),
                ("joint_element_loss", "element"),
            ):
                totals[name] += float(components[key]) * batch
            is_last_batch = seen == len(train)
            if (
                accumulated >= convergence.gradient_accumulation_steps
                or is_last_batch
            ):
                if convergence.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    if style_optimizer is not None:
                        scaler.unscale_(style_optimizer)
                    clip_gradients_fail_closed(
                        model,
                        convergence.grad_clip,
                    )
                scaler.step(optimizer)
                if style_optimizer is not None:
                    scaler.step(style_optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if style_optimizer is not None:
                    style_optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                accumulated = 0
                global_step += 1
                loss_ema.update(float(loss.detach()))
                if global_step >= total_updates:
                    stop_training = True
                    break
        metrics = (
            {}
            if args.skip_validation
            else validate_distribution_room_iou(
                ema.module if args.selection == "ema" else model,
                val_loader,
                taxonomy,
                device,
                default_style_id=0 if style_assignments else None,
            )
        )
        elapsed = time.perf_counter() - started
        record = {
            "epoch": epoch,
            **{f"train_{key}": value / max(1, seen) for key, value in totals.items()},
            "seconds": elapsed,
            "samples_per_second": seen / max(elapsed, 1.0e-9),
            "global_step": global_step,
            "loss_ema": loss_ema.value,
            "gradient_balancer": balancer.state_dict(),
            **{f"val_{key}": value for key, value in metrics.items()},
            "condition_mode": "contour_only",
            "flow_matcher": args.flow_matcher,
            "flow_backend_version": backend_version,
            "time_sampling": args.time_sampling,
            "time_sampling_mean": args.time_sampling_mean,
            "time_sampling_std": args.time_sampling_std,
            "flow_sigma": args.flow_sigma,
            "learning_rate_scale": (
                scheduler.lr_at(scheduler.step_index)
                / max(args.head_lr, 1.0e-12)
            ),
            "selection": args.selection,
            "phase": (
                "element_warmup"
                if global_step < args.element_warmup_updates
                else "joint"
            ),
        }
        if metrics:
            record["metric_guard"] = metric_guard.update(
                record["train_loss"],
                metrics["room_macro_iou"],
            )
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if metrics and metrics["room_macro_iou"] > best:
            best = metrics["room_macro_iou"]
        training_state = {
            "optimizer": optimizer.state_dict(),
            "element_optimizer": element_optimizer.state_dict(),
            "style_optimizer": (
                style_optimizer.state_dict()
                if style_optimizer is not None
                else None
            ),
            "scheduler": scheduler.state_dict(),
            "ema_model": ema.module.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best": best,
            "loss_ema": loss_ema.value,
            "metric_guard": metric_guard.state_dict(),
        }
        extra = {
            "training_args": vars(args),
            "val": record,
            "training_state": training_state,
        }
        save_distribution_model(
            model,
            args.output_dir / "last.pt",
            taxonomy=taxonomy,
            extra=extra,
        )
        if metrics and metrics["room_macro_iou"] >= best:
            save_distribution_model(
                ema.module if args.selection == "ema" else model,
                args.output_dir / "best.pt",
                taxonomy=taxonomy,
                extra=extra,
            )
        if stop_training:
            break
        epoch += 1


if __name__ == "__main__":
    main()
