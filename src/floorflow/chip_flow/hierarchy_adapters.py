"""Composable hierarchy adapter stages for the chip token-flow model."""

from __future__ import annotations

from typing import Any

import torch

from floorflow.chip_flow.coarse_to_fine_coupler import CoarseToFineCoupler
from floorflow.chip_flow.complexity_slot_adapter import ComplexitySlotAdapter
from floorflow.chip_flow.legalizer_response import apply_legalizer_displacement
from floorflow.chip_flow.ordered_group_staging import OrderedGroupStaging
from floorflow.chip_flow.port_consensus_coupler import PortConsensusCoupler
from floorflow.chip_flow.spatial_occupancy_coupler import SpatialOccupancyCoupler
from floorflow.chip_flow.structured_heads import StructuredHeadSuite

ATOM_BRANCH_NAMES = (
    "order_group",
    "coarse_to_fine",
    "port_consensus",
    "spatial_route",
    "complexity_tail",
    "hierarchy_coupling",
    "dynamic_graph",
)


def legalizer_action_radius(
    block_prediction: torch.Tensor,
    context: Any,
    *,
    max_displacement: float,
    pin_fraction: float = 0.1,
    edge_fraction: float = 0.25,
) -> torch.Tensor:
    """Bound final legalizer actions by local pin and net geometry."""

    batch, count, _ = block_prediction.shape
    bound = block_prediction.new_full((batch, count), float(max_displacement))
    centers = block_prediction[..., :2].float()
    pins = context.pin_positions
    incidence = context.typed.port_block_incidence
    if pins is not None and incidence is not None:
        distance = (
            centers[:, :, None] - pins.float()[:, None]
        ).abs().sum(dim=-1)
        valid = incidence.bool() & context.typed.port_mask[:, None]
        nearest = distance.masked_fill(~valid, float("inf")).amin(dim=-1)
        pin_bound = float(pin_fraction) * nearest
        bound = torch.minimum(
            bound,
            torch.where(torch.isfinite(pin_bound), pin_bound, bound),
        )
    edge_state = context.edge_state
    if edge_state is not None:
        source = edge_state.cache["b2b_source"]
        target = edge_state.cache["b2b_target"]
        if source.numel():
            flat_centers = centers.reshape(batch * count, 2)
            edge_distance = (
                flat_centers[source] - flat_centers[target]
            ).abs().sum(dim=-1)
            nearest = torch.full(
                (batch * count,),
                float("inf"),
                dtype=edge_distance.dtype,
                device=edge_distance.device,
            )
            nearest.scatter_reduce_(
                0,
                source,
                edge_distance,
                reduce="amin",
            )
            nearest.scatter_reduce_(
                0,
                target,
                edge_distance,
                reduce="amin",
            )
            nearest = nearest.reshape(batch, count)
            edge_bound = float(edge_fraction) * nearest
            bound = torch.minimum(
                bound,
                torch.where(torch.isfinite(edge_bound), edge_bound, bound),
            )
    return bound.clamp_min(0.0)


def install_optional_hierarchy_adapters(
    model: torch.nn.Module,
    hidden: int,
) -> None:
    """Attach the optional sparse adapters from model configuration."""

    model.order_staging = (
        OrderedGroupStaging(hidden, cluster_slots=4)
        if model.order_conditioning
        else None
    )
    model.coarse_to_fine = (
        CoarseToFineCoupler(hidden)
        if model.coarse_to_fine_conditioning
        else None
    )
    model.port_consensus = (
        PortConsensusCoupler(hidden)
        if model.port_consensus_conditioning or model.port_consensus_output
        else None
    )
    model.spatial_occupancy = (
        SpatialOccupancyCoupler(
            hidden,
            route_aware=model.spatial_route_field,
        )
        if model.spatial_occupancy_conditioning or model.spatial_route_field
        else None
    )
    model.complexity_slot = (
        ComplexitySlotAdapter(hidden)
        if model.complexity_slot_conditioning
        else None
    )


def apply_hierarchy_input_adapters(
    *,
    latent_residual: torch.Tensor,
    block_state: torch.Tensor,
    block_tokens: torch.Tensor,
    structural_tokens: torch.Tensor,
    structural_mask: torch.Tensor,
    port_state: torch.Tensor,
    port_prediction: torch.Tensor,
    port_tokens: torch.Tensor,
    port_block_incidence: torch.Tensor,
    port_mask: torch.Tensor,
    global_prediction: torch.Tensor,
    structural_prediction: torch.Tensor,
    time: torch.Tensor,
    time_token: torch.Tensor,
    context: Any,
    order_staging: torch.nn.Module | None,
    coarse_to_fine: torch.nn.Module | None,
    port_consensus: torch.nn.Module | None,
    port_consensus_output: bool,
    spatial_occupancy: torch.nn.Module | None,
    complexity_slot: torch.nn.Module | None,
    structured_heads: StructuredHeadSuite | None,
    dynamic_graph_input: bool = False,
    dynamic_relay: torch.nn.Module | None = None,
    dynamic_to_latent: torch.nn.Module | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Apply pre-backbone hierarchy adapters and same-call head routing."""

    port_output_residual = None
    branch_residuals = []
    branch_active = []
    atom_diagnostics = {}
    constraints = getattr(context, "constraints", None)
    zero_residual = torch.zeros_like(block_state)
    if order_staging is not None:
        if context.order_cluster_ids is None or context.order_cluster_mask is None:
            raise ValueError("order conditioning requires cluster ids and masks")
        order_residual, _ = order_staging(
            block_tokens,
            block_state,
            structural_tokens[:, :4],
            context.order_cluster_ids,
            context.order_cluster_mask,
            time,
        )
        branch_residuals.append(order_residual)
        branch_active.append(True)
    else:
        branch_residuals.append(zero_residual)
        branch_active.append(False)
    if coarse_to_fine is not None:
        if context.coarse_membership is None:
            raise ValueError(
                "coarse-to-fine conditioning requires structural membership"
            )
        coarse_residual, _ = coarse_to_fine(
            block_tokens,
            block_state,
            context.coarse_membership,
            structural_mask,
            global_prediction,
            structural_prediction,
            time,
        )
        branch_residuals.append(coarse_residual)
        branch_active.append(True)
    else:
        branch_residuals.append(zero_residual)
        branch_active.append(False)
    if structured_heads is not None and structured_heads.dual_semantic_ports is not None:
        # Port consensus must consume the refined L3 geometry rather than the
        # backbone's unrefined port prediction.
        connected = port_block_incidence.any(dim=1) & port_mask
        port_prediction = structured_heads.dual_semantic_ports(
            port_prediction,
            port_state,
            port_tokens,
            connected,
            port_mask,
            time_token,
        )
    if port_consensus is not None:
        port_residual, _ = port_consensus(
            block_tokens,
            block_state,
            port_prediction,
            port_block_incidence,
            port_mask,
            time,
            pin_positions=context.pin_positions,
            pin_mask=context.pin_mask,
        )
        if port_consensus_output:
            port_output_residual = port_residual
            branch_residuals.append(zero_residual)
            branch_active.append(False)
        else:
            branch_residuals.append(port_residual)
            branch_active.append(True)
    else:
        branch_residuals.append(zero_residual)
        branch_active.append(False)
    if spatial_occupancy is not None:
        if context.block_areas is None:
            raise ValueError("spatial occupancy requires per-block areas")
        edge_cache = (
            context.edge_state.cache
            if getattr(spatial_occupancy, "route_aware", False)
            else None
        )
        spatial_residual, _ = spatial_occupancy(
            block_tokens,
            block_state,
            context.block_areas,
            cache=edge_cache,
        )
        branch_residuals.append(spatial_residual)
        branch_active.append(True)
    else:
        branch_residuals.append(zero_residual)
        branch_active.append(False)
    if complexity_slot is not None:
        if context.complexity is None:
            raise ValueError("complexity slots require a case complexity estimate")
        slot_residual, _ = complexity_slot(
            block_tokens,
            block_state,
            context.complexity,
            time,
        )
        branch_residuals.append(slot_residual)
        branch_active.append(True)
    else:
        branch_residuals.append(zero_residual)
        branch_active.append(False)
    if structured_heads is not None:
        coupler = structured_heads.hierarchy_coupling
        if coupler is not None:
            if context.coarse_membership is None:
                raise ValueError(
                    "structured hierarchy coupling requires structural membership"
                )
            branch_residuals.append(coupler(
                block_tokens,
                block_state,
                context.coarse_membership,
                structural_mask,
                global_prediction,
                structural_prediction,
                structural_tokens,
                time,
            ))
            branch_active.append(True)
        else:
            branch_residuals.append(zero_residual)
            branch_active.append(False)
    else:
        branch_residuals.append(zero_residual)
        branch_active.append(False)
    if dynamic_graph_input:
        if dynamic_relay is None or context.edge_state is None:
            raise ValueError("dynamic graph input requires edge cache")
        dynamic_state = dynamic_relay(
            block_state,
            cache=context.edge_state.cache,
        )
        branch_residuals.append(dynamic_to_latent(dynamic_state))
        branch_active.append(True)
    else:
        branch_residuals.append(zero_residual)
        branch_active.append(False)
    branches = torch.stack(branch_residuals, dim=1)
    aggregator = getattr(structured_heads, "atom_aggregator", None)
    if aggregator is not None:
        if branches.shape[1] != aggregator.atom_count:
            raise ValueError(
                "trajectory branch count must match atom_aggregator.atom_count"
            )
        aggregated, atom_diagnostics = aggregator(
            branches,
            block_tokens,
            global_prediction,
            structural_tokens,
            structural_mask,
            port_tokens,
            port_mask,
            time,
            torch.tensor(
                branch_active,
                dtype=torch.bool,
                device=branches.device,
            )[None].expand(branches.shape[0], -1),
        )
    else:
        aggregated = branches.sum(dim=1)
    if constraints is not None:
        if constraints.shape != (*aggregated.shape[:2], 5):
            raise ValueError("aggregated residuals must match constraints")
        preplaced = constraints[..., 1].ne(0)
        fixed = constraints[..., 0].ne(0)
        aggregated = torch.cat(
            [
                aggregated[..., :2]
                * (~preplaced)[..., None].to(aggregated.dtype),
                aggregated[..., 2:]
                * (~(fixed | preplaced))[..., None].to(aggregated.dtype),
            ],
            dim=-1,
        )
    latent_residual = latent_residual + aggregated
    return (
        latent_residual,
        port_output_residual,
        port_prediction,
        atom_diagnostics,
    )


def apply_hierarchy_output_adapters(
    *,
    block_prediction: torch.Tensor,
    block_tokens: torch.Tensor,
    block_state: torch.Tensor,
    time: torch.Tensor,
    time_token: torch.Tensor,
    context: Any,
    structural_mask: torch.Tensor,
    global_bbox_consensus: torch.nn.Module | None,
    global_prediction: torch.Tensor | None,
    port_output_residual: torch.Tensor | None,
    boundary_anchor: torch.nn.Module | None,
    dynamic_relay: torch.nn.Module | None,
    projector_residual_head: torch.nn.Module | None,
    structured_heads: StructuredHeadSuite | None,
    apply_legalizer_response: bool = False,
    legalizer_response_strength: float = 1.0,
    legalizer_response_max_displacement: float = 0.1,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Apply post-backbone structured heads and established output adapters."""

    if structured_heads is not None and structured_heads.factorized_block is not None:
        constraints = getattr(context, "constraints", None)
        block_prediction = block_prediction + structured_heads.factorized_block(
            block_prediction,
            block_tokens,
            block_state,
            time_token,
            constraints=constraints,
        )
    if global_bbox_consensus is not None:
        if global_prediction is None or context.block_areas is None:
            raise ValueError(
                "global bbox consensus requires global prediction and block areas"
            )
        block_prediction, _ = global_bbox_consensus(
            block_prediction,
            global_prediction,
            context.block_areas,
            constraints=getattr(context, "constraints", None),
        )
    if port_output_residual is not None:
        block_prediction = block_prediction + port_output_residual
    if boundary_anchor is not None:
        if context.coarse_membership is None:
            raise ValueError("boundary anchor requires structural membership")
        boundary_residual, _ = boundary_anchor(
            block_tokens,
            block_prediction,
            context.coarse_membership[..., 5:9],
            time,
            areas=context.block_areas,
            constraints=getattr(context, "constraints", None),
        )
        block_prediction = block_prediction + boundary_residual
    if dynamic_relay is not None:
        if context.edge_state is None:
            raise ValueError("dynamic graph relay requires cached edge state")
        block_prediction = block_prediction + dynamic_relay(
            block_state,
            cache=context.edge_state.cache,
        )
    block_base_state = block_prediction
    projector_residual = None
    if projector_residual_head is not None:
        projector_residual = projector_residual_head(
            block_tokens,
            block_base_state,
            time_token,
        )
        block_prediction = block_prediction + projector_residual
    legalizer_displacement = None
    legalizer_rung_logits = None
    if structured_heads is not None:
        legalizer_displacement, legalizer_rung_logits = (
            structured_heads.predict_legalizer_response(
                block_tokens,
                block_state,
                block_prediction,
                time_token,
            )
        )
        if apply_legalizer_response:
            radius = legalizer_action_radius(
                block_prediction,
                context,
                max_displacement=legalizer_response_max_displacement,
            )
            block_prediction = apply_legalizer_displacement(
                block_prediction,
                legalizer_displacement,
                strength=legalizer_response_strength,
                max_displacement=legalizer_response_max_displacement,
                center_mask=(
                    context.constraints[..., 1].eq(0)
                    if context.constraints is not None
                    else None
                ),
                per_block_radius=radius,
            )
    return (
        block_prediction,
        block_base_state,
        projector_residual,
        legalizer_displacement,
        legalizer_rung_logits,
    )


__all__ = [
    "ATOM_BRANCH_NAMES",
    "apply_hierarchy_input_adapters",
    "apply_hierarchy_output_adapters",
    "install_optional_hierarchy_adapters",
    "legalizer_action_radius",
]
