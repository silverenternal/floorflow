#!/usr/bin/env python3
"""Train all-channel prediction on real FloorFlow predicted controls."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.dynamic_object_instance import (
    object_instance_targets,
)
from floorflow.heads.pixel_distribution import pixel_distribution_loss
from floorflow.heads.room_mask_decoder import (
    room_mask_loss,
)
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.model.predicted_channel_model import PredictedChannelModel
from floorflow.objectives.cad_stage_loss import cad_stage_loss
from floorflow.objectives.furniture_vector_refine import (
    furniture_containment_loss,
)
from floorflow.objectives.routing_regularization import (
    typed_routing_regularization,
)
from floorflow.training.cad_furniture_assignment import (
    furniture_room_assignment_loss,
)
from floorflow.training.proposal_distillation import ProposalTeacherIndex
from floorflow.training.room_capability_training import (
    instance_room_losses,
)
from floorflow.training.typed_channel_cli import (
    model_config,
    parse_args,
    seed_everything,
)
from floorflow.training.typed_channel_eval import (
    accumulate_train_component_totals,
    empty_train_component_totals,
    materialize_train_component_totals,
    selection_score,
    train_component_values,
)
from floorflow.training.typed_channel_feature_cache import (
    ensure_feature_cache_manifest,
    extract_frozen_features,
    read_frozen_batch,
    typed_channel_cache_fingerprint,
    write_frozen_batch,
)
from floorflow.training.typed_channel_init import (
    load_typed_channel_initialization,
)
from floorflow.training.typed_channel_instance_loss_kwargs import (
    typed_channel_instance_loss_kwargs,
)
from floorflow.training.typed_channel_loss_balance import (
    build_typed_channel_balancer,
)
from floorflow.training.typed_channel_route_metrics import typed_channel_module_gradient_metrics
from floorflow.training.typed_channel_scopes import (
    configure_typed_channel_scope,
    make_typed_channel_optimizer_from_args,
)
from floorflow.training.typed_channel_validation import validate_typed_channel


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index not in {None, 0}:
        raise ValueError("typed channel training is restricted to cuda:0")
    non_blocking = device.type == "cuda"
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    train = RasterPredictedControlDataset(
        args.predicted_root,
        "train",
        args.cache_root,
        args.data_root,
        image_size=args.image_size,
        limit=args.limit_train,
    )
    val = RasterPredictedControlDataset(
        args.predicted_root,
        "val",
        args.cache_root,
        args.data_root,
        image_size=args.image_size,
        limit=args.limit_val,
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "prefetch_factor": 2 if args.workers > 0 else None,
    }
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    config = model_config(args)
    model = PredictedChannelModel(taxonomy, **config).to(device)
    init_report = (
        load_typed_channel_initialization(model, args.init_checkpoint) if args.init_checkpoint is not None else None
    )
    scope_report = configure_typed_channel_scope(model, args.train_scope)
    if args.feature_cache_root is not None and args.train_scope in {
        "full",
        "field",
    }:
        raise ValueError("feature cache requires a frozen field scope")
    if args.feature_cache_root is not None:
        ensure_feature_cache_manifest(
            args.feature_cache_root,
            typed_channel_cache_fingerprint(args, config),
        )
    stage_runtime = CadStageRuntime(
        taxonomy,
        device,
        room_checkpoint=args.room_checkpoint,
        structure_checkpoint=args.structure_checkpoint,
        centerline_checkpoint=args.centerline_checkpoint,
        opening_checkpoint=args.opening_checkpoint,
        room_blend=args.room_blend,
        structure_size=128,
    )
    seed_everything(args.seed)
    optimizer = make_typed_channel_optimizer_from_args(model, args)
    teacher_index = (
        ProposalTeacherIndex(args.teacher_proposal_root / "train" / "manifest.json")
        if args.use_teacher_room_proposals
        else None
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, len(train_loader) * args.epochs),
        eta_min=args.lr * 0.05,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(args.amp and args.amp_dtype == "float16" and device.type == "cuda"),
    )
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    positive = torch.from_numpy(
        load_positive_weights(
            args.cache_root,
            taxonomy.num_target_channels,
            20.0,
        )
    ).to(device)
    class_weights = torch.cat([torch.ones(1, device=device), positive])
    if not args.no_normalize_class_weights:
        class_weights = class_weights / class_weights.mean().clamp_min(1.0e-6)
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    balancer = build_typed_channel_balancer(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.jsonl").write_text("", encoding="utf-8")
    history = []
    best_score = float("-inf")
    best_furniture_score = float("-inf")
    global_step = 0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        loss_total = torch.zeros((), device=device)
        component_totals = empty_train_component_totals(device)
        seen = 0
        for target, footprint, floor_ids, names, image in train_loader:
            target = target.to(device, non_blocking=non_blocking)
            footprint = footprint.to(device, non_blocking=non_blocking)
            image = image.to(device, non_blocking=non_blocking)
            floor_ids = floor_ids.to(device, non_blocking=non_blocking)
            cached = (
                read_frozen_batch(
                    args.feature_cache_root,
                    "train",
                    names,
                    device,
                )
                if args.feature_cache_root is not None
                else None
            )
            if cached is None:
                controls = _predicted_controls(
                    args.predicted_root,
                    "train",
                    names,
                    args,
                    device,
                )
                controls, stage_diagnostics = stage_runtime(
                    controls,
                    footprint,
                    image,
                    floor_ids,
                )
                frozen_features = extract_frozen_features(
                    model,
                    controls,
                    footprint,
                )
                if args.feature_cache_root is not None:
                    write_frozen_batch(
                        args.feature_cache_root,
                        "train",
                        names,
                        controls,
                        frozen_features,
                    )
            else:
                controls, frozen_features = cached
                stage_diagnostics = {}
            accumulate_train_component_totals(
                component_totals,
                {f"stage_{key}": value for key, value in stage_diagnostics.items()},
                target.shape[0],
            )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=args.amp and device.type == "cuda",
            ):
                output = model(
                    controls,
                    footprint,
                    cached_features=frozen_features,
                )
                logits = output["logits"]
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                aux_zero = logits.sum() * 0.0
                instance_room_loss = partition_loss = aux_zero
                query_loss = assignment_loss = containment_loss = aux_zero
                loss, components = pixel_distribution_loss(
                    logits,
                    target,
                    footprint,
                    taxonomy,
                    class_weights=class_weights,
                    ce_weight=args.ce_weight,
                    room_iou_weight=args.room_iou_weight,
                    tversky_weight=args.tversky_weight,
                    boundary_weight=args.boundary_weight,
                    family_iou_weight=args.family_iou_weight,
                )
                stage_result = cad_stage_loss(
                    logits,
                    target,
                    footprint,
                    taxonomy,
                    wall_precision_weight=args.wall_precision_weight,
                    opening_precision_weight=args.opening_precision_weight,
                    opening_host_weight=args.opening_host_weight,
                    opening_family_weight=args.opening_family_weight,
                    opening_class_weights=(
                        class_weights[
                            torch.tensor(
                                taxonomy.opening_ids,
                                device=device,
                            )
                        ]
                        if args.opening_type_class_balance
                        else None
                    ),
                    return_loss_terms=balancer is not None,
                )
                (
                    stage_loss,
                    stage_components,
                    *stage_loss_terms,
                ) = stage_result
                components.update(stage_components)
                loss = loss + args.stage_family_weight * stage_loss
                routing_loss, routing_components = typed_routing_regularization(model, output)
                components.update(routing_components)
                loss = loss + args.routing_regularization_weight * routing_loss
                if args.enable_instance_room_decoder and args.instance_room_weight > 0.0:
                    teacher = None
                    if teacher_index is not None:
                        teacher = teacher_index.load(
                            names,
                            image_size=256,
                            max_proposals=args.teacher_max_proposals,
                            device=device,
                        )
                    instance_result = instance_room_losses(
                        output,
                        target,
                        footprint,
                        taxonomy,
                        max_rooms=args.room_instance_queries,
                        teacher=teacher,
                        program=controls["program"],
                        **typed_channel_instance_loss_kwargs(
                            args,
                            return_loss_terms=balancer is not None,
                        ),
                    )
                    (
                        instance_room_loss,
                        partition_loss,
                        instance_components,
                        *raw_loss_terms,
                    ) = instance_result
                    components.update(instance_components)
                    loss = (
                        loss
                        + args.instance_room_weight * instance_room_loss
                        + args.instance_room_partition_weight * partition_loss
                    )
                components["furniture_query_loss"] = query_loss.detach()
                if args.enable_furniture_queries and args.furniture_query_weight > 0.0:
                    classes, masks, valid = object_instance_targets(
                        target,
                        taxonomy.furniture_ids,
                        max_instances=args.furniture_max_queries,
                    )
                    query_prediction = {
                        "class_logits": output["furniture_query_class_logits"],
                        "mask_logits": torch.nn.functional.interpolate(
                            output["furniture_query_mask_logits"],
                            size=target.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                    }
                    query_loss, query_components = room_mask_loss(
                        query_prediction,
                        classes,
                        masks,
                        valid,
                        boundary_weight=2.0,
                        mask_pos_weight=args.furniture_query_mask_pos_weight,
                        unmatched_mask_weight=(args.furniture_query_unmatched_mask_weight),
                        mask_tversky_weight=(args.furniture_query_tversky_weight),
                    )
                    components.update(query_components)
                    components["furniture_query_loss"] = query_loss.detach()
                    loss = loss + args.furniture_query_weight * query_loss
                    if args.furniture_room_assignment and args.furniture_room_assignment_weight > 0.0:
                        assignment_loss, assignment_components = furniture_room_assignment_loss(
                            output,
                            classes,
                            masks,
                            valid,
                            target,
                            taxonomy,
                        )
                        components.update(assignment_components)
                        components["furniture_room_assignment_loss"] = assignment_loss.detach()
                        loss = loss + args.furniture_room_assignment_weight * assignment_loss
                    if args.furniture_containment_weight > 0.0:
                        containment_loss = furniture_containment_loss(
                            output["furniture_query_mask_logits"],
                            torch.isin(target, room_ids),
                        )
                        components["furniture_query_containment"] = containment_loss.detach()
                        loss = loss + args.furniture_containment_weight * containment_loss
                zero_component = query_loss.detach() * 0.0
                for name in (
                    "room_mask_class_loss",
                    "room_mask_dice_bce_loss",
                    "room_mask_unmatched_loss",
                    "room_mask_tversky_loss",
                    "furniture_query_containment",
                    "furniture_room_assignment_loss",
                    "furniture_room_assignment_accuracy",
                    "instance_room_class",
                    "instance_room_mask",
                    "instance_room_mask_centroid",
                    "instance_room_unmatched",
                    "instance_room_tversky",
                    "instance_room_reference_anchor",
                    "instance_room_partition",
                    "instance_room_centroid_error",
                    "instance_room_area",
                ):
                    components.setdefault(name, zero_component)
                if balancer is not None:
                    terms = {
                        "ce": components["ce_term"],
                        "room_iou": components["room_iou_term"],
                        "tversky": components["tversky_term"],
                        "boundary": components["boundary_term"],
                        "family_iou": components["family_iou_term"],
                        "routing": routing_loss,
                        "furniture_query": query_loss,
                        "instance_room": (raw_loss_terms[0]["instance_room"] if raw_loss_terms else instance_room_loss),
                        "instance_partition": partition_loss,
                        "furniture_assignment": assignment_loss,
                        "furniture_containment": containment_loss,
                    }
                    if stage_loss_terms:
                        terms.update(stage_loss_terms[0])
                    if raw_loss_terms:
                        terms.update(
                            {name: value for name, value in raw_loss_terms[0].items() if name != "instance_room"}
                        )
                    balancer.maybe_update(
                        terms,
                        list(model.parameters()),
                        global_step,
                    )
                    loss = balancer.weighted_loss(terms)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            loss_total += loss.detach().float() * target.shape[0]
            accumulate_train_component_totals(
                component_totals,
                train_component_values(components),
                target.shape[0],
            )
            seen += target.shape[0]
        atom_proxy_rows = []
        metrics = validate_typed_channel(
            model,
            val_loader,
            taxonomy,
            stage_runtime,
            _predicted_controls,
            args,
            device,
            epoch=epoch,
            row_sink=atom_proxy_rows,
        )
        score = selection_score(
            metrics,
            furniture_weight=args.selection_furniture_weight,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(loss_total.cpu()) / max(1, seen),
            "selection_score": score,
            "loss_components": materialize_train_component_totals(
                component_totals,
                seen,
            ),
            "loss_scales": dict(balancer.scales) if balancer is not None else None,
            "loss_balance_reference": balancer.reference if balancer is not None else None,
            "loss_gradient_diagnostics": balancer.diagnostics() if balancer is not None else None,
            "module_gradient_norms": typed_channel_module_gradient_metrics(model),
            **metrics,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if atom_proxy_rows:
            with (args.output_dir / "atom_proxy_rows.jsonl").open(
                "a",
                encoding="utf-8",
            ) as handle:
                for row in atom_proxy_rows:
                    handle.write(json.dumps(row) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "config": {**vars(args), **config},
            "taxonomy": taxonomy.to_json(),
            "epoch": epoch,
            "metrics": metrics,
            "loss_balancer": balancer.state_dict() if balancer is not None else None,
            "train_scope": scope_report,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(checkpoint, args.output_dir / "best.pt")
        if metrics["furniture_f1"] > best_furniture_score:
            best_furniture_score = metrics["furniture_f1"]
            torch.save(
                checkpoint,
                args.output_dir / "best_furniture.pt",
            )
        if args.save_epoch_checkpoints:
            torch.save(checkpoint, args.output_dir / f"epoch_{epoch:02d}.pt")
    report = {
        "schema_version": "typed-channel-attention-v1",
        "scope": "real predicted-control all-channel prediction",
        "test_opened": False,
        "target_used_at_inference": False,
        "train_scope": scope_report,
        "initialization": init_report,
        "best_selection_score": best_score,
        "selected_epoch": int(max(history, key=lambda row: row["selection_score"])["epoch"]),
        "best_furniture_f1": best_furniture_score,
        "selected_furniture_epoch": int(max(history, key=lambda row: row["furniture_f1"])["epoch"]),
        "config": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            **config,
        },
        "history": history,
        "atom_proxy_rows": str(args.output_dir / "atom_proxy_rows.jsonl"),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
