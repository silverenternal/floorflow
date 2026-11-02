#!/usr/bin/env python3
"""Train instance-query room controls over a native CubiCasa raster."""

from __future__ import annotations

import json

import torch

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.data.predicted_control_raster import (
    DualControlRasterDataset,
    PredictedControlRasterDataset,
)
from floorflow.heads.predicted_room_identity import room_identity_loss
from floorflow.heads.raster_control_bank import (
    RasterInstanceControl,
)
from floorflow.heads.room_mask_decoder import (
    room_mask_loss,
    room_mask_partition_loss,
)
from floorflow.heads.room_partition import room_partition_confidence
from floorflow.heads.room_program_set import room_program_set_loss
from floorflow.heads.room_query_set_head import room_query_set_loss
from floorflow.objectives.gradient_balancer import GradientNormBalancer
from floorflow.objectives.room_boundary_loss import room_boundary_loss
from floorflow.training.metric_accumulator import PackedMetricAccumulator
from floorflow.training.proposal_distillation import (
    ProposalTeacherIndex,
)
from floorflow.training.raster2seq_teacher_runtime import (
    raster2seq_teacher_loss,
)
from floorflow.training.raster_instance_cli import parse_args, validate_args
from floorflow.training.raster_instance_control import (
    configure_joint_slot_state_recovery,
    configure_room_class_recovery,
)
from floorflow.training.raster_instance_evaluation import (
    evaluate,
    identity_balance_score,
    selection_score,
)
from floorflow.training.raster_instance_setup import (
    build_loss_contract,
    build_raster_control_model,
    build_room_optimizer,
    compatible_initial_state,
    full_resolution_predictions,
    load_room_encoder,
    materialize_float_outputs,
    seed_everything,
    training_summary,
    weighted_loss_sum,
    write_partial_report,
)
from floorflow.training.room_sampling import (
    build_room_loaders,
    prepare_raster_dataset,
    raster_dataset_kwargs,
)
from floorflow.training.runtime_acceleration import (
    RuntimeConfig,
    configure_torch_runtime,
)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    configure_torch_runtime(
        device,
        tf32=args.tf32,
        triton_kernels=args.triton_kernels,
    )
    runtime = RuntimeConfig.from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dual_control_raster and args.control_raster_root is None:
        raise ValueError("--dual-control-raster requires --control-raster-root")
    if args.train_sampling == "curriculum" and args.train_pool_size is None:
        args.train_pool_size = max(args.limit_train * 4, args.limit_train)
    train_limit = None if args.full_train else args.limit_train
    dataset_class = CubiCasaRasterControlDataset
    if args.control_raster_root is not None:
        dataset_class = (
            DualControlRasterDataset
            if args.dual_control_raster
            else PredictedControlRasterDataset
        )
    dataset_kwargs = raster_dataset_kwargs(args)
    train_dataset = dataset_class(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=True,
        limit=None,
        **dataset_kwargs,
    )
    removed_train = prepare_raster_dataset(
        train_dataset,
        limit=(
            args.train_pool_size
            if args.train_sampling == "curriculum"
            else train_limit
        ),
        offset=args.train_offset,
        shuffle_seed=args.seed if args.train_sampling == "seeded" else None,
    )
    if removed_train:
        print(f"removed empty training targets: {removed_train}", flush=True)
    taxonomy = train_dataset.taxonomy
    val_dataset = dataset_class(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=None,
        **{**dataset_kwargs, "include_room_targets": False},
    )
    removed_val = prepare_raster_dataset(
        val_dataset,
        limit=args.limit_val,
        offset=args.val_offset,
    )
    if removed_val:
        print(f"removed empty validation targets: {removed_val}", flush=True)
    query_state_enabled = bool(args.query_set_head or args.joint_room_slot_state)
    teacher = (
        ProposalTeacherIndex(args.teacher_manifest, taxonomy=taxonomy)
        if args.teacher_manifest is not None
        else None
    )
    initial_payload = (
        torch.load(
            args.initialize_from,
            map_location="cpu",
            weights_only=False,
        )
        if args.initialize_from is not None
        else None
    )
    model = build_raster_control_model(args, taxonomy, device)
    load_room_encoder(model, args.room_checkpoint)
    if initial_payload is not None:
        state = initial_payload.get("model", initial_payload)
        current = model.state_dict()
        compatible = compatible_initial_state(
            state,
            current,
            exclude_room_encoder=args.preserve_room_encoder,
        )
        model.load_state_dict(compatible, strict=False)
        print(f"initialized {len(compatible)}/{len(current)} tensors from {args.initialize_from}", flush=True)
    if not args.train_encoder:
        model.room.requires_grad_(False)
    if args.class_only or args.slot_state_only:
        if args.teacher_only:
            raise ValueError("recovery-only modes cannot use --teacher-only")
        trained_names = (
            configure_joint_slot_state_recovery(model)
            if args.slot_state_only
            else configure_room_class_recovery(model)
        )
        print(f"recovery tensors: {len(trained_names)}", flush=True)
    model = runtime.prepare_model(model)
    scaler = runtime.make_scaler(device)
    positive = torch.from_numpy(
        load_positive_weights(
            args.cache_root,
            taxonomy.num_target_channels,
            20.0,
        )
    ).to(device)
    room_indices = torch.tensor(
        [class_id - 1 for class_id in taxonomy.room_ids],
        device=device,
    )
    room_weights = positive[room_indices]
    room_weights = room_weights / room_weights.mean().clamp_min(1.0e-6)
    optimizer = build_room_optimizer(model, args)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    loss_balancer = (
        GradientNormBalancer(
            {
                "room_mask": 1.0,
                "partition": args.partition_weight,
                "fused_identity": args.fused_identity_weight,
                **(
                    {"dense": args.dense_weight}
                    if args.train_encoder or args.dense_identity_refiner
                    else {}
                ),
                **(
                    {"query_set": args.query_set_weight}
                    if query_state_enabled and args.query_set_weight > 0.0
                    else {}
                ),
                **(
                    {"room_program": args.room_program_weight}
                    if args.room_program_head and args.room_program_weight > 0.0
                    else {}
                ),
                **({"teacher": args.teacher_weight} if teacher is not None else {}),
            },
            reference="room_mask",
            minimum_scale=args.gradient_balance_min_scale,
            maximum_scale=args.gradient_balance_max_scale,
            update_interval=args.gradient_balance_interval,
        )
        if args.dynamic_room_loss_weights and not args.teacher_only
        else None
    )
    if (
        loss_balancer is not None
        and initial_payload is not None
        and not args.reset_gradient_balancer
        and "gradient_balancer" in initial_payload
    ):
        try:
            loss_balancer.load_state_dict(
                initial_payload["gradient_balancer"]
            )
        except ValueError as error:
            print(f"gradient balancer warm-start skipped: {error}", flush=True)
    loss_contract = build_loss_contract(
        args,
        teacher_present=teacher is not None,
        dynamic_weights=loss_balancer is not None,
    )
    train_loader, val_loader, curriculum_sampler = build_room_loaders(
        args,
        train_dataset,
        val_dataset,
        taxonomy,
    )
    blend_weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    history = []
    best_score = -1.0
    best_weight = blend_weights[-1]
    global_step = 0
    for epoch in range(args.epochs):
        if curriculum_sampler is not None:
            curriculum_sampler.set_epoch(epoch, args.epochs)
        model.train()
        if not args.train_encoder:
            model.room.eval()
        losses = PackedMetricAccumulator()
        component_totals = PackedMetricAccumulator()
        component_count = 0
        seen = 0
        non_blocking = device.type == "cuda"
        for batch in train_loader:
            target, footprint, image, _floor_ids, names = batch[:5]
            target = target.to(device, non_blocking=non_blocking)
            footprint = footprint.to(device, non_blocking=non_blocking)
            image = image.to(device, non_blocking=non_blocking)
            if not args.teacher_only:
                classes, masks, valid = (
                    value.to(device, non_blocking=non_blocking)
                    for value in batch[5:8]
                )
                masks = masks.float()
                target_counts = batch[8]
            if runtime.channels_last:
                image = image.contiguous(memory_format=torch.channels_last)
            model.query_count_gate = args.query_count_gate and global_step >= args.query_gate_start_step
            with runtime.autocast(device):
                outputs = model(footprint, control_image=image)
            outputs = materialize_float_outputs(outputs)
            predictions = full_resolution_predictions(
                outputs,
                tuple(target.shape[-2:]),
            )
            components = {}
            if args.teacher_only:
                if teacher is None:
                    raise ValueError("--teacher-only requires --teacher-manifest")
                distill_loss, distill_components = raster2seq_teacher_loss(
                    teacher,
                    predictions,
                    list(names),
                    max_proposals=args.teacher_max_proposals,
                    device=device,
                    class_weight=args.teacher_class_weight,
                )
                loss = args.teacher_weight * distill_loss
                components.update(distill_components)
            else:
                mask_loss, components = room_mask_loss(
                    predictions,
                    classes,
                    masks,
                    valid,
                    target_counts=target_counts,
                    class_weights=room_weights,
                    boundary_weight=1.0,
                    assignment_class_weight=0.5,
                    assignment_mask_weight=2.0,
                    mask_loss_weight=2.0,
                    unmatched_mask_weight=args.unmatched_mask_weight,
                    mask_tversky_weight=args.mask_tversky_weight,
                    existence_loss_weight=(
                        0.0
                        if query_state_enabled
                        else args.existence_loss_weight
                    ),
                    query_separation_weight=args.query_separation_weight,
                    supervised_query_separation=(
                        args.query_separation_mode == "supervised_same_class"
                    ),
                    class_aware_query_separation=(
                        not args.query_separation_class_agnostic
                    ),
                    partition_consistency_weight=(
                        args.partition_consistency_weight
                    ),
                    partition_conflict_only=args.partition_conflict_only,
                    partition_label_smoothing=(
                        args.partition_label_smoothing
                    ),
                )
                partition_loss, partition_components = room_mask_partition_loss(
                    predictions,
                    target,
                    footprint,
                    taxonomy,
                    class_weights=room_weights,
                    pixel_weights=(
                        room_partition_confidence(
                            target,
                            footprint,
                            taxonomy,
                        )
                        if args.partition_confidence_weighting
                        else None
                    ),
                )
                fused_identity_loss, fused_components = room_identity_loss(
                    outputs["fused_logits"],
                    target,
                    footprint,
                    taxonomy.room_ids,
                    class_weights=room_weights,
                    dice_weight=0.5,
                    boundary_weight=0.5,
                )
                loss_terms = {
                    "room_mask": mask_loss,
                    "partition": partition_loss,
                    "fused_identity": fused_identity_loss,
                }
                if (
                    args.room_boundary_weight > 0.0
                    and "room_boundary_logits" in predictions
                ):
                    boundary_loss, boundary_components = room_boundary_loss(
                        predictions["room_boundary_logits"],
                        masks,
                        valid,
                    )
                    loss_terms["room_boundary"] = boundary_loss
                    components.update(boundary_components)
                elif args.room_boundary_weight > 0.0:
                    raise ValueError(
                        "room boundary loss requires --room-boundary-owner-refiner"
                    )
                components.update(partition_components)
                components.update(fused_components)
                if args.train_encoder or args.dense_identity_refiner:
                    dense_loss, dense_components = room_identity_loss(
                        outputs["dense_logits"],
                        target,
                        footprint,
                        taxonomy.room_ids,
                        class_weights=room_weights,
                        dice_weight=0.5,
                        boundary_weight=0.5,
                    )
                    loss_terms["dense"] = dense_loss
                    components.update(dense_components)
                if query_state_enabled and args.query_set_weight > 0.0:
                    query_set_loss, query_set_components = room_query_set_loss(
                        predictions,
                        classes,
                        valid,
                        target_masks=masks,
                        existence_weight=args.existence_loss_weight,
                        count_weight=args.query_count_weight,
                        class_count_weight=args.query_class_count_weight,
                        slot_class_anchor_weight=args.slot_class_anchor_weight,
                        support_assignment_weight=args.slot_owner_weight,
                    )
                    loss_terms["query_set"] = query_set_loss
                    components.update(query_set_components)
                if args.room_program_head and args.room_program_weight > 0.0:
                    program_loss, program_components = room_program_set_loss(
                        {
                            "class_logits": outputs[
                                "room_program_class_logits"
                            ],
                            "presence_logits": outputs[
                                "room_program_presence_logits"
                            ],
                            "cardinality_logits": outputs[
                                "room_program_cardinality_logits"
                            ],
                        },
                        classes,
                        valid,
                        class_weights=room_weights,
                        cardinality_weight=(
                            args.room_program_cardinality_weight
                        ),
                    )
                    loss_terms["room_program"] = program_loss
                    components.update(program_components)
                if teacher is not None:
                    distill_loss, distill_components = raster2seq_teacher_loss(
                        teacher,
                        predictions,
                        list(names),
                        max_proposals=args.teacher_max_proposals,
                        device=device,
                        class_weight=args.teacher_class_weight,
                    )
                    loss_terms["teacher"] = distill_loss
                    components.update(distill_components)
                if loss_balancer is None:
                    loss = weighted_loss_sum(loss_terms, args)
                else:
                    loss_balancer.maybe_update(
                        loss_terms,
                        trainable_parameters,
                        global_step,
                    )
                    loss = loss_balancer.weighted_loss(loss_terms)
            component_totals.add(components)
            losses.add(
                {"loss": loss},
                weight=float(target.shape[0]),
            )
            component_count += 1
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.gradient_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            seen += target.shape[0]
        metrics = evaluate(
            model,
            val_loader,
            taxonomy,
            device,
            blend_weights,
            runtime=runtime,
        )
        if args.selection_mode == "dense_relative_identity":
            reference = metrics[0.0]
            current_weight, current_metrics = max(
                metrics.items(),
                key=lambda item: identity_balance_score(
                    item[1],
                    reference,
                    count_weight=args.selection_count_weight,
                ),
            )
            current_score = identity_balance_score(
                current_metrics,
                reference,
                count_weight=args.selection_count_weight,
            )
        else:
            current_weight, current_metrics = max(
                metrics.items(),
                key=lambda item: selection_score(
                    item[1],
                    count_weight=args.selection_count_weight,
                ),
            )
            current_score = selection_score(
                current_metrics,
                count_weight=args.selection_count_weight,
            )
        row = {
            "epoch": epoch,
            "train_loss": losses.finalize(seen)["loss"],
            "train_components": component_totals.finalize(component_count),
            "best_blend": current_weight,
            "selection_mode": args.selection_mode,
            "best_selection_score": current_score,
            "blends": {str(key): value for key, value in metrics.items()},
        }
        if loss_balancer is not None:
            row["gradient_balancer_scales"] = dict(loss_balancer.scales)
            row["gradient_balancer_norm_ema"] = dict(loss_balancer.norm_ema)
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": vars(args),
            "metrics": current_metrics,
            "blend": current_weight,
            "loss_contract": loss_contract,
            "query_count_gate_effective": bool(model.query_count_gate),
        }
        if loss_balancer is not None:
            state["gradient_balancer"] = loss_balancer.state_dict()
        torch.save(state, args.output_dir / "last.pt")
        if current_score > best_score:
            best_score = current_score
            best_weight = current_weight
            torch.save(state, args.output_dir / "best.pt")
        write_partial_report(args.output_dir, history, args)
    report = {
        "status": "complete",
        "scope": (
            "native-raster teacher-only mask pretraining"
            if args.teacher_only
            else (
                "native-raster frozen-proposal class recovery"
                if args.class_only
                else "native-raster instance-query room control"
            )
        ),
        "input_protocol": "native_cubicasa_raster",
        "target_used_at_inference": False,
        "worker_room_targets": not args.teacher_only,
        "train_cases": len(train_dataset),
        "validation_cases": len(val_dataset),
        "best_blend": best_weight,
        "best_selection_score": best_score,
        "history": history,
        "config": vars(args),
        "teacher_distillation": (
            {
                "manifest": str(args.teacher_manifest),
                "weight": args.teacher_weight,
                "max_proposals": args.teacher_max_proposals,
                "class_labels_copied": args.teacher_class_weight > 0.0,
                "class_loss_weight": args.teacher_class_weight,
            }
            if teacher is not None
            else None
        ),
        "gradient_balancer": (
            loss_balancer.state_dict()
            if loss_balancer is not None
            else None
        ),
        "loss_contract": loss_contract,
        "runtime": runtime.manifest(),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(training_summary(report), indent=2, default=str))


if __name__ == "__main__":
    main()
