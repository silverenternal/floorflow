"""Joint distribution and flow-matching objective."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.distribution.cfm_model import DistributionCFMModel
from floorflow.distribution.coupling_loss import distribution_coupling_loss
from floorflow.distribution.element_loss import element_set_loss
from floorflow.distribution.element_targets import element_targets
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.flow.cfm_backend import (
    sample_flow_batch,
)
from floorflow.flow.core import target_to_continuous
from floorflow.heads.algorithmic_dynamic_head import dynamic_routing_loss
from floorflow.heads.pixel_distribution import pixel_distribution_loss
from floorflow.objectives.furniture_vector_refine import (
    furniture_vector_loss,
)
from floorflow.objectives.structural_consistency import (
    structural_consistency_loss,
)
from floorflow.objectives.topology_closure import (
    low_confidence_gap_closure_loss,
)


def joint_distribution_cfm_loss(
    model: DistributionCFMModel,
    target: torch.Tensor,
    source: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    hierarchy: ChannelHierarchy,
    *,
    floor_ids: torch.Tensor | None = None,
    queries_per_group: dict[str, int] | None = None,
    class_weights: torch.Tensor | None = None,
    flow_weight: float = 1.0,
    pixel_weight: float = 1.0,
    element_weight: float = 0.25,
    coupling_weight: float = 0.1,
    source_weight: float = 0.1,
    family_iou_weight: float = 0.0,
    room_iou_weight: float = 0.5,
    tversky_weight: float = 0.2,
    boundary_weight: float = 0.0,
    structural_weight: float = 0.0,
    topology_closure_weight: float = 0.0,
    dynamic_route_weight: float = 0.0,
    flow_endpoint_weight: float = 0.0,
    flow_change_weight: float = 0.0,
    contour_output: Mapping[str, object] | None = None,
    flow_matcher: str = "torchcfm",
    time_sampling: str = "logit_normal",
    sigma: float = 0.05,
    time_sampling_mean: float = 0.0,
    time_sampling_std: float = 1.0,
    minibatch_ot: str = "token",
    ot_token_grid: int = 8,
    furniture_vector_weight: float = 0.05,
    furniture_family_weight: float = 1.25,
    style_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    x1 = target_to_continuous(target, model.target_channels)
    flow = sample_flow_batch(
        source,
        x1,
        backend=flow_matcher,
        sigma=sigma,
        time_sampling=time_sampling,
        time_sampling_mean=time_sampling_mean,
        time_sampling_std=time_sampling_std,
        minibatch_ot=minibatch_ot,
        token_grid=ot_token_grid,
    )
    time = flow.time
    x_t = flow.location
    velocity_target = flow.velocity
    velocity = model(
        x_t,
        time,
        footprint,
        floor_ids=floor_ids,
        style_ids=style_ids,
        source=source,
    )
    valid = footprint.float()
    change = (x1 - source).abs().amax(dim=1, keepdim=True)
    flow_weight_map = valid * (
        1.0 + float(flow_change_weight) * change
    )
    flow = (
        (velocity.float() - velocity_target).pow(2) * flow_weight_map
    ).sum() / flow_weight_map.sum().clamp_min(1.0) / model.target_channels
    predicted_x1 = x_t + (1.0 - time[:, None, None, None]) * velocity
    positive_error = -predicted_x1 * torch.nn.functional.logsigmoid(
        predicted_x1
    )
    negative_error = -(1.0 - x1) * torch.nn.functional.logsigmoid(
        -predicted_x1
    )
    if class_weights is not None:
        endpoint_weights = class_weights
        if endpoint_weights.numel() == model.target_channels + 1:
            endpoint_weights = endpoint_weights[1:]
        if endpoint_weights.numel() != model.target_channels:
            raise ValueError(
                "class_weights must match foreground or full class count"
            )
        positive_weight = endpoint_weights.to(
            device=x1.device,
            dtype=x1.dtype,
        ).view(1, -1, 1, 1)
        endpoint_error = positive_error * positive_weight + negative_error
    else:
        endpoint_error = positive_error + negative_error
    endpoint = (
        endpoint_error * valid
    ).sum() / valid.sum().clamp_min(1.0) / model.target_channels
    contour = (
        contour_output
        if contour_output is not None
        else model.contour_initial_state(
            footprint,
            floor_ids=floor_ids,
            style_ids=style_ids,
        )
    )
    pixel, pixel_components = pixel_distribution_loss(
        contour["pixel"]["logits"],
        target,
        footprint,
        taxonomy,
        class_weights=class_weights,
        family_iou_weight=family_iou_weight,
        room_iou_weight=room_iou_weight,
        tversky_weight=tversky_weight,
        boundary_weight=boundary_weight,
    )
    targets = element_targets(
        target,
        taxonomy,
        hierarchy,
        max_queries=queries_per_group,
    )
    element, element_components = element_set_loss(
        contour["elements"],
        targets,
        family_weights={"furniture": furniture_family_weight},
    )
    coupling, coupling_components = distribution_coupling_loss(
        contour["pixel"],
        contour["elements"],
        hierarchy,
        footprint=footprint,
    )
    structural = contour["pixel"]["logits"].sum() * 0.0
    structural_components: dict[str, torch.Tensor] = {}
    if structural_weight > 0.0:
        size = target.shape[-2:]
        elements = contour["elements"]
        objects = {
            "opening_logits": torch.nn.functional.interpolate(
                elements["opening"]["mask_logits"].float(),
                size=size,
                mode="bilinear",
                align_corners=False,
            ),
            "furniture_logits": torch.nn.functional.interpolate(
                elements["furniture"]["mask_logits"].float(),
                size=size,
                mode="bilinear",
                align_corners=False,
            ),
        }
        room_indices = hierarchy.groups["room"]
        structural, structural_components = structural_consistency_loss(
            contour["pixel"]["logits"][:, room_indices],
            objects,
            target,
            footprint,
            taxonomy,
            room_probability=contour["pixel"]["probability"][
                :, room_indices
            ].sum(dim=1),
        )
    topology_closure = contour["pixel"]["logits"].sum() * 0.0
    if topology_closure_weight > 0.0:
        wall_index = taxonomy.class_id("wall") - 1
        topology_closure = low_confidence_gap_closure_loss(
            contour["pixel"]["logits"][:, wall_index],
            target,
            footprint,
            hierarchy.room_ids,
        )
    source_loss = (
        (contour["cfm"]["state"] - x1).pow(2) * valid
    ).sum() / valid.sum().clamp_min(1.0) / model.target_channels
    route_loss = source_loss * 0.0
    route_components: dict[str, torch.Tensor] = {}
    route_weights = contour["pixel"].get("multi_scale_weights")
    if dynamic_route_weight > 0.0 and route_weights is not None:
        route_loss, route_components = dynamic_routing_loss(route_weights)
    furniture_vector = contour["pixel"]["logits"].sum() * 0.0
    furniture_components: dict[str, torch.Tensor] = {}
    if furniture_vector_weight > 0.0:
        furniture = contour["elements"]["furniture"]
        furniture_vector, furniture_components = furniture_vector_loss(
            furniture["type_logits"],
            furniture["mask_logits"],
            target,
            tuple(taxonomy.furniture_ids),
            footprint,
        )
    total = (
        float(flow_weight) * (flow + float(flow_endpoint_weight) * endpoint)
        + float(pixel_weight) * pixel
        + float(element_weight) * element
        + float(coupling_weight) * coupling
        + float(structural_weight) * structural
        + float(topology_closure_weight) * topology_closure
        + float(source_weight) * source_loss
        + float(dynamic_route_weight) * route_loss
        + float(furniture_vector_weight) * furniture_vector
    )
    components = {
        "_flow_term": flow + float(flow_endpoint_weight) * endpoint,
        "_pixel_term": pixel,
        "_element_term": element,
        "_coupling_term": coupling,
        "_structural_term": structural,
        "_topology_closure_term": topology_closure,
        "_source_term": source_loss,
        "_dynamic_route_term": route_loss,
        "_furniture_vector_term": furniture_vector,
        "joint_flow_loss": flow.detach(),
        "joint_endpoint_loss": endpoint.detach(),
        "joint_pixel_loss": pixel.detach(),
        "joint_element_loss": element.detach(),
        "joint_coupling_loss": coupling.detach(),
        "joint_structural_loss": structural.detach(),
        "joint_topology_closure_loss": topology_closure.detach(),
        "joint_source_loss": source_loss.detach(),
        "joint_dynamic_route_loss": route_loss.detach(),
        "joint_furniture_vector_loss": furniture_vector.detach(),
        "joint_total_loss": total.detach(),
        **pixel_components,
        **element_components,
        **coupling_components,
        **structural_components,
        **route_components,
        **furniture_components,
    }
    return total, components


__all__ = ["joint_distribution_cfm_loss"]
