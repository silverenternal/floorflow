#!/usr/bin/env python3
"""Train guided MiTA-CFM refinement from coarse CAD controls."""

from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.flow.core import target_to_continuous
from floorflow.objectives.cfm_structural import (
    CFMStructuralWeights,
    cfm_structural_loss,
    predicted_x1,
)
from floorflow.objectives.gradient_balancer import GradientNormBalancer
from floorflow.flow.guided_control import guided_source_from_labels
from floorflow.flow.mask_policies import (
    AdversarialMaskScheduler,
    AttributeMaskPolicy,
    CompositeMaskPolicy,
    SalienceMaskPolicy,
    SpatialBlockMaskPolicy,
    make_mask_policy,
)
from floorflow.model.mask_discriminator import (
    MaskDifficultyDiscriminator,
    difficulty_rank_loss,
)
from floorflow.heads.room_partition import batch_room_partition_target, room_partition_loss
from floorflow.heads.room_mask_decoder import (
    room_mask_loss,
    room_mask_partition_loss,
    room_mask_targets,
)
from floorflow.model.sparse_control_attention import (
    MaskedControlFlow,
)
from floorflow.model.learned_mask_generator import LearnedMaskGenerator
from floorflow.training.mask_curriculum import MaskRatioCurriculum
from floorflow.training.dynamic_policy import RoomDynamicPolicy
from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.training.distribution_integration import (
    attach_distribution_controller,
    distribution_checkpoint_fields,
    load_distribution_head_state,
)
from floorflow.training.guided_arguments import parse_args
from floorflow.training.guided_validation import validate_guided
from floorflow.training.room_instance_integration import attach_room_mask_head


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_payload = torch.load(
        args.initialize_from,
        map_location=device,
        weights_only=False,
    )
    model, config, taxonomy, _state = load_checkpoint_model(
        args.initialize_from,
        device,
        taxonomy=None,
        family_attention_impl=(
            None
            if args.family_attention_impl == "checkpoint"
            else args.family_attention_impl
        ),
    )
    if args.enable_room_mask_head:
        model = attach_room_mask_head(
            model,
            taxonomy,
            queries=args.room_mask_queries,
            hidden_dim=args.room_mask_hidden,
            resolution=args.room_mask_resolution,
        )
    distribution_head_state = checkpoint_payload.get("distribution_head")
    gate_state = checkpoint_payload.get("distribution_gate")
    if distribution_head_state is None and args.distribution_head_checkpoint is not None:
        distribution_head, _distribution_payload = load_distribution_head_state(
            args.distribution_head_checkpoint,
            model.output.in_channels,
            taxonomy.num_classes,
            args.distribution_hidden,
            taxonomy.room_ids,
        )
        distribution_head_state = distribution_head.state_dict()
    model = attach_distribution_controller(
        model,
        taxonomy,
        head_state=distribution_head_state,
        gate_state=gate_state,
        hidden_channels=args.distribution_hidden,
        gate_hidden=args.distribution_gate_hidden,
    )
    if args.sparse_control_attention:
        model = MaskedControlFlow(
            model,
            attention_type=args.sparse_attention_type,
        ).to(device)
        if "sparse_attention" in checkpoint_payload:
            model.sparse_attention.load_state_dict(
                checkpoint_payload["sparse_attention"],
                strict=False,
            )
    mask_generator = (
        LearnedMaskGenerator(model.target_channels).to(device)
        if args.learned_mask_generator
        else None
    )
    if mask_generator is not None and "mask_generator" in checkpoint_payload:
        mask_generator.load_state_dict(checkpoint_payload["mask_generator"])
    mask_generator_optimizer = (
        torch.optim.Adam(mask_generator.parameters(), lr=args.head_lr)
        if mask_generator is not None
        else None
    )
    image_size = int(config.get("image_size", 256))
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=image_size,
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=True,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=image_size,
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train, shuffle=True, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        normalized_name = name.removeprefix("base.")
        if normalized_name.startswith(
            (
                "room_partition.",
                "structured_objects.",
                "room_program.",
                "geometry_heads.",
                "semantic_endpoint.",
                "room_mask_decoder.",
                "distribution.",
                "distribution_velocity_gate.",
            )
        ):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=1.0e-4,
    )
    initial_missing = checkpoint_payload.get(
        "missing_token",
        torch.zeros(model.target_channels),
    )
    missing_token = torch.nn.Parameter(
        initial_missing.to(device=device, dtype=torch.float32)
    )
    missing_optimizer = torch.optim.Adam([missing_token], lr=args.head_lr)
    scheduler = (
        AdversarialMaskScheduler(
            (
                SpatialBlockMaskPolicy(),
                AttributeMaskPolicy(),
                SalienceMaskPolicy(),
                CompositeMaskPolicy(),
            )
        )
        if args.mask_policy == "adaptive"
        else None
    )
    if scheduler is not None and "mask_scheduler" in checkpoint_payload:
        scheduler.load_state_dict(checkpoint_payload["mask_scheduler"])
    fixed_mask_policy = (
        make_mask_policy(args.mask_policy)
        if args.mask_policy != "adaptive"
        else None
    )
    dynamic_policy = RoomDynamicPolicy(
        taxonomy,
        mask_generator=mask_generator,
        scheduler=scheduler,
        fixed_policy=fixed_mask_policy,
    )
    if "dynamic_policy" in checkpoint_payload:
        dynamic_policy.load_state_dict(checkpoint_payload["dynamic_policy"])
    mask_discriminator = (
        MaskDifficultyDiscriminator(model.target_channels).to(device)
        if args.mask_discriminator
        else None
    )
    if (
        mask_discriminator is not None
        and "mask_discriminator" in checkpoint_payload
    ):
        mask_discriminator.load_state_dict(
            checkpoint_payload["mask_discriminator"]
        )
    discriminator_optimizer = (
        torch.optim.Adam(
            mask_discriminator.parameters(),
            lr=args.head_lr,
        )
        if mask_discriminator is not None
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    structural_weights = CFMStructuralWeights(
        family_bce=args.x1_family_bce_weight,
        family_dice=args.x1_family_dice_weight,
        opening_wall=args.x1_opening_wall_weight,
        wall_boundary=args.x1_wall_boundary_weight,
    )
    balancer = GradientNormBalancer(
        {
            "flow": args.flow_weight,
            "partition": args.partition_weight,
            "semantic": args.semantic_weight,
            "x1_structural": args.x1_structural_weight,
        },
        update_interval=args.gradient_balance_interval,
    )
    if "gradient_balancer" in checkpoint_payload:
        balancer.load_state_dict(checkpoint_payload["gradient_balancer"])
    global_step = 0
    ratio_curriculum = MaskRatioCurriculum(
        minimum=args.mask_min,
        maximum=args.mask_max,
        probe_ratio=args.probe_ratio,
        zero_control_probability=args.zero_control_probability,
        probe_probability=args.probe_probability,
    )
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        totals = {
            "loss": 0.0,
            "flow": 0.0,
            "partition": 0.0,
            "semantic": 0.0,
            "x1_structural": 0.0,
            "room_instance": 0.0,
            "route_balance": 0.0,
            "generator_adversarial": 0.0,
        }
        seen = 0
        progress = (epoch + 1) / max(1, args.epochs)
        for target, footprint, floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            floor_ids = floor_ids.to(device)
            full_source = guided_source_from_labels(target, footprint, taxonomy)
            ratio = 0.0
            if args.contour_only_training:
                ratio = 1.0
            elif args.dynamic_mask:
                ratio = ratio_curriculum.sample(progress, device=device)
            prepared = dynamic_policy.propose(
                full_source,
                footprint,
                missing_token,
                missing_ratio=ratio,
                floor_ids=floor_ids,
            ).condition
            source = (
                prepared.soft_control
                if prepared.soft_control is not None
                else prepared.control
            )
            control_mask = prepared.missing_mask
            generator_adv_grads = None
            generator_adv_value = torch.zeros((), device=device)
            x1 = target_to_continuous(target, model.target_channels)
            flow_time = torch.distributions.Beta(1.5, 1.0).sample(
                (target.shape[0],)
            ).to(device)
            time_view = flow_time[:, None, None, None]
            x_t = (1.0 - time_view) * source + time_view * x1
            velocity_target = x1 - source
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.precision == "bf16" and device.type == "cuda",
            ):
                if isinstance(model, MaskedControlFlow):
                    velocity, auxiliary = model.forward_with_aux(
                        x_t,
                        flow_time,
                        footprint,
                        floor_ids=floor_ids,
                        control=source,
                        control_mask=control_mask,
                    )
                else:
                    velocity, auxiliary = model.forward_with_aux(
                        x_t,
                        flow_time,
                        footprint,
                        floor_ids=floor_ids,
                    )
                mask = footprint[:, None].float()
                flow_loss = (
                    (velocity.float() - velocity_target).pow(2) * mask
                ).sum() / mask.sum().clamp_min(1.0) / velocity.shape[1]
                partition_target = batch_room_partition_target(
                    target,
                    footprint,
                    taxonomy,
                    fill_occluded=False,
                )
                partition = room_partition_loss(
                    auxiliary["room_partition_logits"],
                    partition_target,
                )["loss"]
                room_instance_loss = partition.detach() * 0.0
                room_mask_predictions = auxiliary.get("room_mask_predictions")
                if room_mask_predictions is not None:
                    target_classes, target_masks, target_valid = room_mask_targets(
                        target,
                        taxonomy,
                        max_rooms=room_mask_predictions["class_logits"].shape[1],
                    )
                    matched_loss, _matched_components = room_mask_loss(
                        room_mask_predictions,
                        target_classes,
                        target_masks,
                        target_valid,
                    )
                    room_instance_loss = (
                        args.room_mask_weight * matched_loss
                    )
                    if args.room_mask_partition_weight > 0:
                        dense_loss, _dense_components = room_mask_partition_loss(
                            room_mask_predictions,
                            target,
                            footprint,
                            taxonomy,
                        )
                        room_instance_loss = (
                            room_instance_loss
                            + args.room_mask_partition_weight * dense_loss
                        )
                semantic_logits = auxiliary["semantic_logits"]
                semantic = F.cross_entropy(
                    torch.cat(
                        [torch.zeros_like(semantic_logits[:, :1]), semantic_logits],
                        dim=1,
                    ),
                    target,
                )
                implied_x1 = predicted_x1(x_t, velocity, flow_time)
                structural, _structural_components = cfm_structural_loss(
                    implied_x1.float(),
                    x1.float(),
                    footprint[:, None].float(),
                    taxonomy,
                    weights=structural_weights,
                )
                loss_terms = {
                    "flow": flow_loss,
                    "partition": partition,
                    "semantic": semantic,
                    "x1_structural": structural,
                }
                balancer.maybe_update(
                    loss_terms,
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    global_step,
                )
                loss = balancer.weighted_loss(loss_terms)
                loss = loss + room_instance_loss
                route_balance = auxiliary.get("sparse_control", {}).get(
                    "route_balance_loss"
                )
                if route_balance is not None:
                    loss = loss + args.route_balance_weight * route_balance
                else:
                    route_balance = loss.detach() * 0.0
                if scheduler is not None:
                    if mask_discriminator is not None:
                        difficulty_map = (
                            (velocity.float() - velocity_target).pow(2).mean(dim=1, keepdim=True)
                            * footprint[:, None].float()
                        )
                        difficulty_score = difficulty_map.mean(dim=(1, 2, 3))
                        discriminator_logits, discriminator_diag = mask_discriminator(
                            control_mask.detach().float(),
                            difficulty_map.detach(),
                            footprint.detach(),
                        )
                        discriminator_loss, discriminator_metrics = difficulty_rank_loss(
                            discriminator_logits,
                            difficulty_score,
                        )
                        hardness = float(
                            discriminator_diag["predicted_hardness"].mean().detach()
                        )
                        totals["disc_spearman"] = totals.get(
                            "disc_spearman",
                            0.0,
                        ) + float(
                            discriminator_metrics["difficulty_spearman"]
                        ) * target.shape[0]
                        if (
                            mask_generator is not None
                            and prepared.soft_mask is not None
                        ):
                            generator_logits, _ = mask_discriminator(
                                prepared.soft_mask,
                                difficulty_map.detach(),
                                footprint.detach(),
                            )
                            generator_adv_loss = -generator_logits.sigmoid().mean()
                            generator_adv_value = generator_adv_loss.detach()
                            generator_adv_grads = torch.autograd.grad(
                                generator_adv_loss,
                                tuple(mask_generator.parameters()),
                                retain_graph=True,
                                allow_unused=True,
                            )
                    else:
                        discriminator_loss = None
                        hardness = float(
                            flow_loss.detach()
                            / (
                                flow_loss.detach()
                                + partition.detach()
                                + semantic.detach()
                                + 1.0e-6
                            )
                        )
                    if prepared.policy_name is not None:
                        dynamic_policy.update(
                            hardness,
                            policy_name=prepared.policy_name,
                        )
                else:
                    discriminator_loss = None
                if (
                    mask_generator is not None
                    and prepared.soft_mask is not None
                ):
                    footprint_area = footprint[:, None].float().expand_as(
                        prepared.soft_mask
                    )
                    soft_coverage = (
                        prepared.soft_mask * footprint_area
                    ).sum(dim=(1, 2, 3)) / footprint_area.sum(
                        dim=(1, 2, 3)
                    ).clamp_min(1.0)
                    coverage_loss = (
                        soft_coverage - float(ratio)
                    ).pow(2).mean()
                    loss = loss + args.mask_coverage_weight * coverage_loss
            optimizer.zero_grad(set_to_none=True)
            missing_optimizer.zero_grad(set_to_none=True)
            if mask_generator_optimizer is not None:
                mask_generator_optimizer.zero_grad(set_to_none=True)
                if generator_adv_grads is not None:
                    for parameter, gradient in zip(
                        mask_generator.parameters(),
                        generator_adv_grads,
                    ):
                        if gradient is None:
                            continue
                        weighted_gradient = args.generator_adversarial_weight * gradient
                        parameter.grad = (
                            weighted_gradient
                            if parameter.grad is None
                            else parameter.grad + weighted_gradient
                        )
            if discriminator_optimizer is not None:
                discriminator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if discriminator_loss is not None:
                discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            missing_optimizer.step()
            if mask_generator_optimizer is not None:
                mask_generator_optimizer.step()
            if discriminator_optimizer is not None:
                discriminator_optimizer.step()
            global_step += 1
            for name, value in (
                ("loss", loss),
                ("flow", flow_loss),
                ("partition", partition),
                ("semantic", semantic),
                ("x1_structural", structural),
                ("room_instance", room_instance_loss),
                ("route_balance", route_balance),
                (
                    "generator_adversarial",
                    generator_adv_value,
                ),
            ):
                totals[name] += float(value.detach()) * target.shape[0]
            seen += target.shape[0]
        metrics = validate_guided(
            model,
            val_loader,
            taxonomy,
            device,
            args.validation_steps,
            missing_token=missing_token if args.dynamic_mask else None,
            mask_ratio=(
                1.0
                if args.contour_only_training
                else args.validation_mask_ratio
            ),
            mask_policy=fixed_mask_policy or AttributeMaskPolicy(),
            mask_generator=mask_generator,
        )
        record = {
            "epoch": epoch,
            **{f"train_{key}": value / max(1, seen) for key, value in totals.items()},
            "seconds": time.perf_counter() - started,
            **metrics,
        }
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            **distribution_checkpoint_fields(model),
            "taxonomy": taxonomy.to_json(),
            "config": {**vars(args), "image_size": image_size},
            "val": record,
            "epoch": epoch,
            "missing_token": missing_token.detach().cpu(),
        }
        if scheduler is not None:
            checkpoint["mask_scheduler"] = scheduler.state_dict()
        if mask_generator is not None:
            checkpoint["mask_generator"] = mask_generator.state_dict()
        if mask_discriminator is not None:
            checkpoint["mask_discriminator"] = mask_discriminator.state_dict()
        checkpoint["dynamic_policy"] = dynamic_policy.state_dict()
        checkpoint["gradient_balancer"] = balancer.state_dict()
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["room_macro_iou"] > best:
            best = metrics["room_macro_iou"]
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
