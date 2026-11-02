"""Batch adapters shared by chip-flow training and evaluation."""

from __future__ import annotations

import torch

from floorflow.chip_flow.graph_conditioning import (
    dense_b2b_adjacency,
    graph_conditioned_features,
)
from floorflow.chip_flow.guided_sampling import sample_hierarchy_guided_flow
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
)
from floorflow.chip_flow.hierarchical_flow_runtime import sample_hierarchy_flow
from floorflow.chip_flow.hierarchical_model import ChipHierarchicalSlotFlowModel
from floorflow.chip_flow.model import ChipBlockFlowModel
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.shape_mode_head import attach_shape_mode_head
from floorflow.chip_flow.slot_model import ChipSlotFlowModel
from floorflow.chip_flow.sparse_flow_model import ChipSparseHierarchyFlowModel
from floorflow.chip_flow.sparse_slot_flow import ChipSlotTransportFlowModel
from floorflow.chip_flow.trinity_token_flow import ChipTrinityTokenFlowModel


def graph_inputs(batch: dict) -> dict[str, torch.Tensor]:
    return {
        "constraints": batch["constraints"],
        "b2b": batch["b2b"],
        "p2b": batch["p2b"],
        "pins": batch["pins"],
        "n_b2b": batch["n_b2b"],
        "n_p2b": batch["n_p2b"],
        "n_pins": batch["n_pins"],
    }


def graph_adapter_inputs(
    model: ChipSlotFlowModel,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cached = batch.get("_graph_adapter_inputs")
    if cached is not None and cached[0] is model:
        return cached[1]
    if model.graph_features:
        features, adjacency, key_mask = graph_conditioned_features(batch)
    else:
        features = batch["features"]
        adjacency = dense_b2b_adjacency(
            batch["b2b"],
            batch["n_b2b"],
            features.shape[1],
        )
        key_mask = torch.ones(
            features.shape[:2],
            device=features.device,
            dtype=torch.bool,
        )
    if model.training and model.wire_dropout > 0.0:
        keep = torch.rand(
            features.shape[:2],
            device=features.device,
        ).ge(model.wire_dropout)
        adjacency = adjacency * keep[:, :, None] * keep[:, None, :]
    if model.training and model.constraint_dropout > 0.0 and model.graph_features:
        features = features.clone()
        keep = torch.rand_like(features[..., 3:9]).ge(model.constraint_dropout)
        features[..., 3:9] *= keep
        keep = torch.rand_like(features[..., 16:19]).ge(
            model.constraint_dropout
        )
        features[..., 16:19] *= keep
    batch["_graph_adapter_inputs"] = (
        model,
        (features, adjacency, key_mask),
    )
    return features, adjacency, key_mask


def build_chip_model(config: dict) -> torch.nn.Module:
    values = dict(config)
    architecture = values.pop("architecture", "global_slot")
    shape_modes = int(values.pop("shape_modes", 0))
    shape_mode_width = values.pop("shape_mode_width", None)
    shape_mode_max_delta = float(values.pop("shape_mode_max_delta", 0.5))
    shape_mode_center_scale = float(
        values.pop("shape_mode_center_scale", 0.0)
    )
    shape_mode_top_fractions = values.pop(
        "shape_mode_top_fractions",
        None,
    )
    shape_mode_apply = bool(values.pop("shape_mode_apply", False))
    shape_mode_selection = str(
        values.pop("shape_mode_selection", "learned")
    )
    if "relation_specific_attention" in values:
        values["relation_specific"] = bool(
            values.pop("relation_specific_attention")
        )
    if "structural_relation_head" in values:
        values["structural_relations"] = bool(
            values.pop("structural_relation_head")
        )
    if architecture == "typed_hierarchy":
        values.pop("output_mode", None)
    classes = {
        "flat_token": ChipBlockFlowModel,
        "hierarchical_slot": ChipHierarchicalSlotFlowModel,
        "graph_slot": ChipSlotFlowModel,
        "sparse_hierarchy": ChipSparseHierarchyFlowModel,
        "slot_transport": ChipSlotTransportFlowModel,
        "slot_residual": ChipSlotTransportFlowModel,
        "trinity_token": ChipTrinityTokenFlowModel,
        "typed_hierarchy": HierarchicalTokenFlowModel,
    }
    model = classes.get(architecture, ChipSlotFlowModel)(**values)
    if architecture == "typed_hierarchy" and shape_modes > 0:
        attach_shape_mode_head(
            model,
            modes=shape_modes,
            width=shape_mode_width,
            max_delta=shape_mode_max_delta,
            center_scale=shape_mode_center_scale,
            mode_top_fractions=shape_mode_top_fractions,
            apply=shape_mode_apply,
            selection=shape_mode_selection,
        )
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """Return all trainable and non-trainable tensor parameters."""

    return sum(parameter.numel() for parameter in model.parameters())


def parameter_matched_global_slot_config(
    reference_config: dict,
    target_parameters: int,
    *,
    tolerance: float = 0.02,
) -> tuple[dict, int, float]:
    """Choose a global-slot width closest to a sparse-hierarchy budget."""

    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    base = ChipSlotFlowModel(
        feature_dim=int(reference_config.get("feature_dim", 10)),
        hidden=int(reference_config.get("hidden", 192)),
        global_dim=int(
            reference_config.get(
                "global_dim",
                reference_config.get("hidden", 192),
            )
        ),
        target_layers=int(reference_config.get("target_layers", 4)),
        condition_layers=int(reference_config.get("condition_layers", 2)),
        decoder_layers=int(reference_config.get("decoder_layers", 4)),
        heads=int(reference_config.get("heads", 6)),
    )
    base_hidden = base.hidden
    base_global = base.global_dim
    heads = int(reference_config.get("heads", 6))
    hidden_start = max(heads, (base_hidden - 32) // heads * heads)
    hidden_stop = base_hidden + 64
    hidden_values = range(hidden_start, hidden_stop + 1, heads)
    global_values = (
        max(1, base_global - 16),
        max(1, base_global - 8),
        base_global,
        base_global + 8,
        base_global + 16,
    )
    best: tuple[float, dict, int] | None = None
    for hidden in hidden_values:
        for global_dim in global_values:
            config = {
                "architecture": "global_slot",
                "feature_dim": int(reference_config.get("feature_dim", 10)),
                "hidden": hidden,
                "global_dim": global_dim,
                "target_layers": int(reference_config.get("target_layers", 4)),
                "condition_layers": int(
                    reference_config.get("condition_layers", 2)
                ),
                "decoder_layers": int(
                    reference_config.get("decoder_layers", 4)
                ),
                "heads": heads,
            }
            parameters = count_parameters(build_chip_model(config))
            relative_error = abs(parameters - target_parameters) / target_parameters
            if best is None or relative_error < best[0]:
                best = (relative_error, config, parameters)
    if best is None:
        raise RuntimeError("no global-slot candidates were generated")
    relative_error, config, parameters = best
    if relative_error > tolerance:
        raise ValueError(
            "closest global-slot candidate exceeds the parameter tolerance: "
            f"{relative_error:.4f} > {tolerance:.4f}"
        )
    return config, parameters, relative_error


def encode_target_and_condition(
    model: torch.nn.Module,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor, object | None]:
    if isinstance(model, ChipSlotFlowModel) and model.graph_input_enabled:
        features, adjacency, key_mask = graph_adapter_inputs(model, batch)
        target = model.encode_target(
            batch["target_latent"],
            features,
            adjacency,
            key_mask,
        )
        condition = model.encode_condition(
            features,
            adjacency,
            key_mask,
        )
        return target, condition, None
    if isinstance(model, ChipSlotTransportFlowModel):
        graph = graph_inputs(batch)
        context = model.encode_condition_context(batch["features"], **graph)
        target = model.encode_target(
            batch["target_latent"],
            batch["features"],
            **graph,
        )
        if model.residual_slots:
            target = model.to_residual_target(target, context)
        return target, model.condition_vector(context), context
    if isinstance(model, ChipSparseHierarchyFlowModel):
        graph = graph_inputs(batch)
        context = model.encode_condition_context(batch["features"], **graph)
        target = model.encode_target(
            batch["target_latent"],
            batch["features"],
            **graph,
        )
        return target, context.global_state, context
    if isinstance(model, ChipTrinityTokenFlowModel):
        context = model.prepare_context(batch)
        return batch["target_latent"].float(), context, context
    if isinstance(model, HierarchicalTokenFlowModel):
        context = model.prepare_context(batch)
        return batch["target_latent"].float(), context, context
    target = model.encode_target(
        batch["target_latent"],
        batch["features"],
    )
    condition = model.encode_condition(batch["features"])
    return target, condition, None


def decode_for_batch(
    model: torch.nn.Module,
    state: torch.Tensor,
    condition: torch.Tensor,
    context: object | None,
    batch: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(model, ChipSlotFlowModel) and model.graph_input_enabled:
        features, adjacency, key_mask = graph_adapter_inputs(model, batch)
        return model.decode(
            state,
            condition,
            features,
            adjacency,
            key_mask,
        ), {}
    if isinstance(model, ChipSlotTransportFlowModel):
        if context is None:
            raise ValueError("slot transport requires a condition context")
        return model.decode_transport(
            state,
            context,
            batch["features"],
            **graph_inputs(batch),
        )
    if isinstance(model, ChipSparseHierarchyFlowModel):
        if context is None:
            raise ValueError("sparse hierarchy requires a condition context")
        return model.decode_with_diagnostics(
            state,
            context,
            batch["features"],
        )
    if isinstance(model, ChipTrinityTokenFlowModel):
        if context is None:
            raise ValueError("Trinity token flow requires a condition context")
        return model.decode(state, context)
    if isinstance(model, HierarchicalTokenFlowModel):
        return state.float(), {}
    return model.decode(state, condition, batch["features"]), {}


def sample_for_batch(
    model: torch.nn.Module,
    batch: dict,
    *,
    sampling_steps: int,
    generator: torch.Generator,
    guidance_scale: float = 0.0,
    guidance_energy: str = "routing",
    projection_mode: str = "post_step",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(model, ChipSlotFlowModel) and model.graph_input_enabled:
        features, adjacency, key_mask = graph_adapter_inputs(model, batch)
        latent = model.sample(
            features,
            sampling_steps,
            adjacency=adjacency,
            key_mask=key_mask,
            generator=generator,
        )
        return latent, {}
    if isinstance(model, ChipSlotTransportFlowModel):
        return model.sample(
            batch["features"],
            sampling_steps,
            generator=generator,
            **graph_inputs(batch),
        )
    if isinstance(model, ChipSparseHierarchyFlowModel):
        return model.sample(
            batch["features"],
            sampling_steps,
            generator=generator,
            **graph_inputs(batch),
        )
    if isinstance(model, ChipTrinityTokenFlowModel):
        context = model.prepare_context(batch)
        latent = batch["source_latent"].float()
        for step in range(sampling_steps):
            time = torch.full(
                (latent.shape[0],),
                step / sampling_steps,
                device=latent.device,
            )
            prediction = model(latent, time, context)
            velocity = (prediction - latent) / (
                1.0 - time[:, None, None]
            ).clamp_min(1.0e-3)
            latent = latent + velocity / sampling_steps
            latent = project_conditioned_latent(
                latent,
                batch["target_latent"],
                batch["constraints"],
            )
        return latent, {}
    if isinstance(model, HierarchicalTokenFlowModel):
        if model.outline_conditioning and "outline_state" not in batch:
            raise ValueError(
                "outline conditioning requires an explicit target-free "
                "batch['outline_state'] during sampling"
            )
        if guidance_scale > 0.0:
            if projection_mode != "post_step":
                raise ValueError(
                    "guided sampling currently supports only post_step projection"
                )
            return sample_hierarchy_guided_flow(
                model,
                batch,
                sampling_steps=sampling_steps,
                generator=generator,
                guidance_scale=guidance_scale,
                energy_mode=guidance_energy,
            )
        return sample_hierarchy_flow(
            model,
            batch,
            sampling_steps=sampling_steps,
            generator=generator,
            projector=model.projector,
            projector_iterations=model.projector_iterations,
            projector_boundary_axes=model.projector_boundary_axes,
            projector_boundary_lock_corners=model.projector_boundary_lock_corners,
            projection_mode=projection_mode,
            coarse_commit_fraction=getattr(
                model,
                "coarse_commit_fraction",
                None,
            ),
            coarse_anchor_tolerance=getattr(
                model,
                "coarse_anchor_tolerance",
                0.05,
            ),
            coarse_anchor_strength=getattr(
                model,
                "coarse_anchor_strength",
                0.0,
            ),
            coarse_anchor_max_step=getattr(
                model,
                "coarse_anchor_max_step",
                0.1,
            ),
        )
    if isinstance(model, ChipSlotFlowModel):
        latent = model.sample(
            batch["features"],
            sampling_steps,
            generator=generator,
        )
        return latent, {}
    if isinstance(model, ChipBlockFlowModel):
        latent = batch["source_latent"].float()
        for step in range(sampling_steps):
            time = torch.full(
                (latent.shape[0],),
                step / sampling_steps,
                device=latent.device,
            )
            latent = latent + model(
                latent,
                time,
                batch["features"],
            ) / sampling_steps
            latent = project_conditioned_latent(
                latent,
                batch["target_latent"],
                batch["constraints"],
            )
        return latent, {}
    raise TypeError(f"unsupported chip model: {type(model).__name__}")


SPARSE_DIAGNOSTIC_NAMES = (
    "active_slot_count",
    "top_k",
    "router_budget_per_case",
    "router_budget_mean",
    "router_budget_min",
    "router_budget_max",
    "router_load",
    "router_usage_entropy",
    "attention_entropy",
)


def sparse_diagnostic_totals(
    diagnostics: dict[str, torch.Tensor],
    cases: int,
) -> dict[str, torch.Tensor]:
    """Accumulate sparse diagnostics on device without per-metric syncs."""

    return {
        name: diagnostics[name].float().mean() * cases
        for name in SPARSE_DIAGNOSTIC_NAMES
        if name in diagnostics
    }


def add_sparse_diagnostics(
    totals: dict[str, float],
    diagnostics: dict[str, torch.Tensor],
    cases: int,
) -> None:
    for name, value in sparse_diagnostic_totals(diagnostics, cases).items():
        totals[name] = totals.get(name, 0.0) + float(value)


def sparse_loss_terms(
    model: torch.nn.Module,
    context: object | None,
    diagnostics: dict[str, torch.Tensor],
    *,
    sparsity_weight: float,
    balance_weight: float,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    supported = (ChipSparseHierarchyFlowModel, ChipSlotTransportFlowModel)
    if not isinstance(model, supported) or context is None:
        return None, {}
    terms = model.sparse_regularization(context, diagnostics)
    terms["decode_active_slots"] = diagnostics["active_slot_count"].float().mean()
    loss = (
        float(sparsity_weight) * terms["budget"]
        + float(balance_weight) * terms["route_balance_total"]
    )
    return loss, terms


__all__ = [
    "add_sparse_diagnostics",
    "build_chip_model",
    "count_parameters",
    "decode_for_batch",
    "encode_target_and_condition",
    "graph_adapter_inputs",
    "graph_inputs",
    "parameter_matched_global_slot_config",
    "sample_for_batch",
    "sparse_diagnostic_totals",
    "sparse_loss_terms",
]
