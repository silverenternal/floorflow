"""Chip-flow objective and schedule kept separate from training orchestration."""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch
import torch.nn.functional as F

from floorflow.chip_flow.auxiliary_objectives import (
    relation_gate_objective,
    structural_relation_objective,
    training_clean_layout,
)
from floorflow.chip_flow.batched import (
    batch_grouping_proxy,
    batch_physics_terms,
    batch_routing_proxy_loss,
)
from floorflow.chip_flow.flow_adapter import (
    decode_for_batch,
    sparse_loss_terms,
)
from floorflow.chip_flow.hard_constraint_projector import (
    project_hard_constraints_latent,
)
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
)
from floorflow.chip_flow.hierarchical_flow_runtime import (
    block_global_state,
    hierarchical_training_state,
    hierarchy_flow_loss,
    interpolate_hierarchy,
)
from floorflow.chip_flow.hierarchy_consistency import hierarchy_cycle_loss
from floorflow.chip_flow.legalizer_response import legalizer_response_loss
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.multi_objective_balance import normalized_loss_balance
from floorflow.chip_flow.nfe_consistency import hierarchy_nfe_consistency_loss
from floorflow.chip_flow.objective_schedules import (
    annealed_weight,
    delayed_ramp_weight,
    hierarchy_bbox_schedule,
    learning_rate,  # noqa: F401
    scheduled_proxy_weight,
    top_fraction_mean,
)
from floorflow.chip_flow.official_cost_surrogate import official_cost_objective
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.shape_mode_head import shape_mode_training_loss
from floorflow.chip_flow.slot_model import ChipSlotFlowModel
from floorflow.chip_flow.source_mixing import source_shelf_probability_at_step
from floorflow.chip_flow.training_endpoint import (
    apply_training_endpoint_contracts,
)


def supervised_loss(
    model: ChipSlotFlowModel,
    batch: dict,
    args: argparse.Namespace,
    device: torch.device,
    *,
    step: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if (
        isinstance(model, HierarchicalTokenFlowModel)
        and model.outline_conditioning
        and "outline_state" not in batch
    ):
        raise ValueError(
            "outline conditioning requires an explicit target-free "
            "batch['outline_state'] during training"
        )
    if isinstance(model, HierarchicalTokenFlowModel) and args.path_steps > 0:
        raise ValueError("typed hierarchy does not yet support path-step loss")
    size = batch["features"].shape[0]
    time_value = torch.sigmoid(args.time_std * torch.randn(size, device=device))
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        (
            clean_latent,
            clean_boxes,
            target_global,
            condition,
            context,
        ) = training_clean_layout(
            model,
            batch,
        )
        clean_is_identity = isinstance(model, HierarchicalTokenFlowModel)
        level_terms = {}
        hierarchy_target = None
        legalizer_response_terms = {}
        legalizer_response = torch.zeros((), device=device)
        legalizer_response_weight = 0.0
        nfe_consistency = torch.zeros((), device=device)
        nfe_terms: dict[str, torch.Tensor] = {}
        shape_mode = torch.zeros((), device=device)
        shape_mode_terms: dict[str, torch.Tensor] = {}
        if isinstance(model, HierarchicalTokenFlowModel):
            source_shelf_probability = source_shelf_probability_at_step(args, step)
            hierarchy_source, hierarchy_target = hierarchical_training_state(
                batch,
                context,
                source_mode=str(
                    getattr(args, "hierarchy_source_mode", "random")
                ),
                source_noise=float(
                    getattr(args, "hierarchy_source_noise", 0.0)
                ),
                source_levels=str(
                    getattr(args, "hierarchy_source_levels", "random")
                ),
                source_shelf_probability=float(source_shelf_probability),
            )
            interpolant = interpolate_hierarchy(
                hierarchy_source,
                hierarchy_target,
                time_value,
            )
            if model.self_conditioning:
                probability = float(
                    getattr(
                        args,
                        "self_conditioning_keep_probability",
                        0.5,
                    )
                )
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        "self-conditioning probability must be in [0, 1]"
                    )
                if float(torch.rand((), device=device)) < probability:
                    with torch.no_grad():
                        history_prediction = model(
                            interpolant,
                            time_value,
                            context,
                        )
                    interpolant = replace(
                        interpolant,
                        previous_block_state=(
                            history_prediction.block_state.detach()
                        ),
                    )
            prediction = model(interpolant, time_value, context)
            nfe_consistency, nfe_terms = hierarchy_nfe_consistency_loss(
                model, interpolant, time_value, prediction, hierarchy_target,
                context, args, step=step)
            legalizer_response_weight = delayed_ramp_weight(
                float(getattr(args, "legalizer_response_weight", 0.0)),
                int(getattr(args, "legalizer_response_delay_steps", 0)),
                int(getattr(args, "legalizer_response_anneal_steps", 0)),
                step,
            )
            if (
                legalizer_response_weight > 0.0
                and prediction.legalizer_displacement is not None
                and prediction.legalizer_rung_logits is not None
            ):
                response = legalizer_response_loss(
                    prediction.legalizer_displacement,
                    prediction.legalizer_rung_logits,
                    batch,
                    displacement_weight=float(
                        getattr(args, "legalizer_displacement_weight", 1.0)
                    ),
                    rung_weight=float(
                        getattr(args, "legalizer_rung_weight", 0.1)
                    ),
                    huber_beta=args.flow_huber_beta,
                )
                legalizer_response = response.total
                legalizer_response_terms = response.as_dict()
            flow, level_terms = hierarchy_flow_loss(
                prediction,
                hierarchy_target,
                huber_beta=args.flow_huber_beta,
                global_weight=args.hierarchy_global_weight,
                structural_weight=args.hierarchy_structural_weight,
                port_weight=args.hierarchy_port_weight,
                constraints=batch["constraints"],
                structural_mask=context.typed.structural_mask,
                port_incidence=context.typed.port_block_incidence,
                port_mask=context.typed.port_mask,
                structured=bool(
                    getattr(args, "structured_head_contract", False)
                ),
                forced_weight=float(
                    getattr(args, "hierarchy_forced_head_weight", 0.0)
                ),
            )
            shape_mode, shape_mode_terms = shape_mode_training_loss(
                prediction, hierarchy_target, args, (model, interpolant, context, batch)
            )
            endpoint = prediction.block_state
        else:
            source = torch.randn_like(target_global)
            time_view = time_value.view(
                -1,
                *([1] * (target_global.ndim - 1)),
            )
            interpolant = (
                (1.0 - time_view) * source + time_view * target_global
            )
            prediction = model(interpolant, time_value, condition)
            flow_target = (
                target_global
                if model.output_mode == "x0"
                else target_global - source
            )
            error = prediction - flow_target
            flow = (
                F.smooth_l1_loss(
                    error,
                    torch.zeros_like(error),
                    beta=args.flow_huber_beta,
                )
                if args.flow_huber_beta > 0
                else error.pow(2).mean()
            )
            endpoint = (
                prediction
                if model.output_mode == "x0"
                else interpolant + (1.0 - time_view) * prediction
            )
        latent, decode_diagnostics = decode_for_batch(
            model,
            endpoint,
            condition,
            context,
            batch,
        )
        sparse_loss, sparse_terms = sparse_loss_terms(
            model,
            context,
            decode_diagnostics,
            sparsity_weight=args.sparsity_weight,
            balance_weight=args.route_balance_weight,
        )
        if args.path_steps > 0 and args.path_weight > 0.0:
            path_state = source
            for path_index in range(args.path_steps):
                path_time = torch.full(
                    (size,),
                    path_index / args.path_steps,
                    device=device,
                )
                path_prediction = model(
                    path_state,
                    path_time,
                    condition,
                )
                path_velocity = (
                    (path_prediction - path_state)
                    / (1.0 - path_time[:, None]).clamp_min(1.0e-3)
                    if model.output_mode == "x0"
                    else path_prediction
                )
                path_state = (
                    path_state + path_velocity / args.path_steps
                )
            path_latent, _ = decode_for_batch(
                model,
                path_state,
                condition,
                context,
                batch,
            )
            path_latent = project_conditioned_latent(
                path_latent.float(),
                batch["target_latent"],
                batch["constraints"],
            )
            path_loss = F.smooth_l1_loss(
                path_latent,
                batch["target_latent"].float(),
                beta=0.1,
            )
        else:
            path_loss = torch.zeros((), device=device)
    raw_latent = project_conditioned_latent(
        latent.float(),
        batch["target_latent"],
        batch["constraints"],
    )
    endpoint_contract_terms: dict[str, torch.Tensor] = {}
    if hierarchy_target is not None:
        latent, endpoint_contract_loss, endpoint_contract_terms = (
            apply_training_endpoint_contracts(
                raw_latent,
                batch,
                args,
                step=step,
                device=device,
            )
        )
    else:
        latent = raw_latent
        endpoint_contract_loss = torch.zeros((), device=device)
    boxes = decode_boxes_batch(latent, batch["areas"])
    official_cost, official_terms = official_cost_objective(
        boxes, batch["boxes"].float(), batch, args, step=step)
    hierarchy_bbox = torch.zeros((), device=device)
    hierarchy_cycle = torch.zeros((), device=device)
    hierarchy_cycle_terms: dict[str, torch.Tensor] = {}
    projector_residual_loss = torch.zeros((), device=device)
    if hierarchy_target is not None:
        predicted_bbox = block_global_state(boxes.float(), batch["areas"])
        hierarchy_bbox = F.smooth_l1_loss(
            predicted_bbox,
            hierarchy_target.global_state.float(),
            beta=0.1,
        )
    if hierarchy_target is not None and float(
        getattr(args, "hierarchy_cycle_weight", 0.0)
    ) > 0.0:
        hierarchy_cycle, hierarchy_cycle_terms = hierarchy_cycle_loss(
            prediction,
            boxes,
            batch,
            context,
            huber_beta=args.flow_huber_beta,
        )
    if (
        hierarchy_target is not None
        and prediction.projector_residual is not None
        and prediction.block_base_state is not None
    ):
        base = prediction.block_base_state.float().detach()
        projected = project_hard_constraints_latent(
            base,
            batch["target_latent"],
            batch["constraints"],
            batch["areas"],
            iterations=int(args.projector_residual_iterations),
        )
        target_residual = projected - base
        projector_residual_loss = F.smooth_l1_loss(
            prediction.projector_residual.float(),
            target_residual,
            beta=args.flow_huber_beta,
        )
    terms = batch_physics_terms(
        boxes,
        batch["areas"],
        batch["constraints"],
        batch["b2b"],
        batch["p2b"],
        batch["pins"],
        batch["n_b2b"],
        batch["n_p2b"],
        batch["n_pins"],
    )
    clean_terms = (
        None
        if clean_is_identity
        else batch_physics_terms(
            clean_boxes.float(),
            batch["areas"],
            batch["constraints"],
            batch["b2b"],
            batch["p2b"],
            batch["pins"],
            batch["n_b2b"],
            batch["n_p2b"],
            batch["n_pins"],
        )
    )
    structured_head_contract = bool(
        getattr(args, "structured_head_contract", False)
    ) and hierarchy_target is not None
    decode = (
        raw_latent.sum() * 0.0
        if structured_head_contract
        else F.smooth_l1_loss(
            raw_latent,
            batch["target_latent"].float(),
            beta=0.1,
        )
    )
    clean = (
        torch.zeros((), device=device)
        if clean_is_identity
        else F.smooth_l1_loss(
            clean_latent.float(),
            batch["target_latent"].float(),
            beta=0.1,
        )
    )
    rho_decode = F.smooth_l1_loss(
        raw_latent[..., 2],
        batch["target_latent"][..., 2].float(),
        beta=0.1,
    )
    rho_clean = (
        torch.zeros((), device=device)
        if clean_is_identity
        else F.smooth_l1_loss(
            clean_latent[..., 2],
            batch["target_latent"][..., 2].float(),
            beta=0.1,
        )
    )
    overlap_scale = float(args.overlap_weight_scale) - 1.0
    physics_cases = terms["total"] + overlap_scale * 0.01 * terms["overlap"]
    clean_physics_cases = (
        torch.zeros_like(physics_cases)
        if clean_terms is None
        else clean_terms["total"] + overlap_scale * 0.01 * clean_terms["overlap"]
    )
    physics = physics_cases.mean()
    clean_physics = clean_physics_cases.mean()
    routing_proxy_weight = scheduled_proxy_weight(args, "routing_proxy", step)
    grouping_proxy_weight = scheduled_proxy_weight(args, "grouping_proxy", step)
    if grouping_proxy_weight > 0.0:
        grouping_proxy = batch_grouping_proxy(
            boxes,
            batch["constraints"],
        ).mean()
    else:
        grouping_proxy = torch.zeros((), device=device)
    structural_relation, relation_weight = structural_relation_objective(
        prediction,
        batch,
        args,
        device,
    )
    relation_gate_loss, relation_gate_weight = relation_gate_objective(
        model,
        args,
        device,
    )
    if routing_proxy_weight > 0.0:
        routing_proxy = batch_routing_proxy_loss(
            boxes,
            batch["areas"],
            batch["b2b"],
            batch["p2b"],
            batch["pins"],
            batch["n_b2b"],
            batch["n_p2b"],
            batch["n_pins"],
        ).mean()
    else:
        routing_proxy = torch.zeros((), device=device)
    bbox_weight = hierarchy_bbox_schedule(
        float(getattr(args, "hierarchy_bbox_weight", 0.0)),
        int(getattr(args, "hierarchy_bbox_decay_steps", 0)),
        step,
    )
    anneal_steps = int(getattr(args, "weight_anneal_steps", 0))
    anneal_mode = str(getattr(args, "weight_schedule", "linear"))
    physics_weight = annealed_weight(
        float(args.physics_weight),
        getattr(args, "physics_final_weight", None),
        step,
        anneal_steps,
        anneal_mode,
    )
    decode_weight = annealed_weight(
        float(args.decode_weight),
        getattr(args, "decode_final_weight", None),
        step,
        anneal_steps,
        anneal_mode,
    )
    clean_weight = annealed_weight(
        float(args.clean_weight),
        getattr(args, "clean_final_weight", None),
        step,
        anneal_steps,
        anneal_mode,
    )
    physics_tail_weight = delayed_ramp_weight(
        float(getattr(args, "physics_tail_weight", 0.0)),
        int(getattr(args, "physics_tail_delay_steps", 0)),
        int(getattr(args, "physics_tail_anneal_steps", 0)),
        step,
    )
    if physics_tail_weight > 0.0:
        fraction = float(getattr(args, "physics_tail_fraction", 0.25))
        physics_tail = top_fraction_mean(physics_cases, fraction)
        clean_physics_tail = top_fraction_mean(
            clean_physics_cases,
            fraction,
        )
    else:
        physics_tail = torch.zeros((), device=device)
        clean_physics_tail = torch.zeros((), device=device)
    loss = (
        flow
        + nfe_consistency
        + official_cost
        + physics_weight * physics
        + decode_weight * (decode + args.rho_weight * rho_decode)
        + clean_weight
        * (clean + args.rho_weight * rho_clean + clean_physics)
        + args.path_weight * path_loss
        + bbox_weight * hierarchy_bbox
        + float(getattr(args, "hierarchy_cycle_weight", 0.0))
        * hierarchy_cycle
        + float(getattr(args, "projector_residual_weight", 1.0))
        * projector_residual_loss
        + routing_proxy_weight * routing_proxy
        + grouping_proxy_weight * grouping_proxy
        + relation_weight * structural_relation
        + relation_gate_weight * relation_gate_loss
        + physics_tail_weight * (physics_tail + clean_physics_tail)
        + endpoint_contract_loss
        + legalizer_response_weight * legalizer_response
        + shape_mode
    )
    if sparse_terms:
        loss = loss + sparse_loss
    result_terms = {
        "flow": flow,
        "physics": physics,
        "decode": decode,
        "rho_decode": rho_decode,
        "clean": clean,
        "clean_physics": clean_physics,
        "physics_tail": physics_tail,
        "clean_physics_tail": clean_physics_tail,
        "physics_tail_current_weight": torch.tensor(physics_tail_weight, device=device),
        "path": path_loss,
        "hierarchy_bbox": hierarchy_bbox,
        "hierarchy_cycle": hierarchy_cycle,
        "projector_residual": projector_residual_loss,
        "routing_proxy": routing_proxy,
        "routing_proxy_current_weight": torch.tensor(routing_proxy_weight, device=device),
        "grouping_proxy": grouping_proxy,
        "structural_relation": structural_relation,
        "structural_relation_weight": torch.tensor(relation_weight, device=device),
        "relation_gate_regularization": relation_gate_loss,
        "relation_gate_current_weight": torch.tensor(relation_gate_weight, device=device),
        "grouping_proxy_current_weight": torch.tensor(grouping_proxy_weight, device=device),
        "hierarchy_bbox_current_weight": torch.tensor(bbox_weight, device=device),
        "physics_current_weight": torch.tensor(physics_weight, device=device),
        "decode_current_weight": torch.tensor(decode_weight, device=device),
        "clean_current_weight": torch.tensor(clean_weight, device=device),
        "legalizer_response_current_weight": torch.tensor(legalizer_response_weight, device=device),
        **level_terms, **nfe_terms,
        **official_terms,
        **hierarchy_cycle_terms,
        **endpoint_contract_terms,
        **legalizer_response_terms,
        **shape_mode_terms,
        **sparse_terms,
    }
    atom_terms = getattr(prediction, "atom_diagnostics", None) or {}
    result_terms.update({
        f"atom_{name.removeprefix('atom_')}": value.mean()
        for name, value in atom_terms.items() if name != "atom_aggregator_gate"})
    if str(getattr(args, "loss_balance_mode", "fixed")) == "normalized":
        loss, balance_terms = normalized_loss_balance(
            {
                "flow": flow,
                "physics": physics_weight * physics,
                "decode": decode_weight * decode,
                "clean": clean_weight * clean,
                "official_cost": official_cost,
                "endpoint": endpoint_contract_loss,
                "legalizer_response": legalizer_response_weight
                * legalizer_response,
                "shape_mode": shape_mode,
            },
            floor=float(getattr(args, "loss_balance_floor", 1.0e-6)),
        )
        result_terms.update(balance_terms)
    return loss, result_terms


