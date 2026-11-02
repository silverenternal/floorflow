"""Objective for contour-to-floorplan structured generation.

The room partition is the primary objective because deployment renders a room
label at every interior pixel. Walls, openings, and furniture are independent
overlay heads, rather than channels competing with rooms in one 65-way softmax.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from floorflow.geometry.interior_wall import interior_wall_loss
from floorflow.heads.geometry_heads import (
    build_geometry_targets,
    geometry_heatmap_loss,
    opening_candidate_loss,
    structural_candidate_mask,
)
from floorflow.objectives.loss_audit import ObjectiveTerm, compose_objective
from floorflow.objectives.observable_room_loss import observable_room_loss
from floorflow.heads.polygon_loop import polygon_loop_loss
from floorflow.representation.polygon_targets import polygon_instance_targets
from floorflow.heads.room_partition import (
    batch_room_partition_target,
    room_partition_confidence,
    room_partition_loss,
    room_partition_metrics,
)
from floorflow.heads.room_groups import room_class_group_index, room_group_loss
from floorflow.heads.room_mask_decoder import (
    room_mask_loss,
    room_mask_partition_loss,
    room_mask_targets,
)
from floorflow.heads.room_program import room_program_loss
from floorflow.heads.structured_objects import group_ce_dice_loss
from floorflow.objectives.structural_consistency import structural_consistency_loss
from floorflow.objectives.semantic_losses import boundary_distance_loss
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.objectives.token_optimal_transport import (
    room_instance_token_targets,
    room_token_transport_loss,
)


@dataclass(frozen=True)
class ConditionObjectiveWeights:
    """Fixed objective weights, normalized around the dense room partition."""

    room_partition: float = 1.0
    room_tversky: float = 0.40
    room_group: float = 0.0
    room_group_head: float = 0.25
    room_boundary: float = 0.15
    room_boundary_distance: float = 0.10
    room_program: float = 0.05
    room_hierarchy: float = 0.50
    room_refine: float = 0.25
    room_transport: float = 0.20
    room_polygon: float = 0.10
    room_mask: float = 0.0
    room_mask_partition: float = 0.0
    opening: float = 0.20
    furniture: float = 0.20
    wall: float = 0.10
    geometry: float = 0.10
    structural_consistency: float = 0.10
    observable_room: float = 0.25
    visible_room_pixel_weight: float = 1.0
    pseudo_furniture_weight: float = 1.0
    pseudo_wall_weight: float = 1.0
    pseudo_opening_weight: float = 1.0
    pseudo_other_weight: float = 1.0


@torch.no_grad()
def partition_room_labels(logits: torch.Tensor, taxonomy: FloorplanTaxonomy) -> torch.Tensor:
    room_ids = torch.tensor(taxonomy.room_ids, device=logits.device, dtype=torch.long)
    return room_ids[logits.argmax(dim=1)]


@torch.no_grad()
def room_macro_iou_from_labels(
    prediction: torch.Tensor,
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> float:
    """Macro IoU over room classes present in the target drawing."""

    values = []
    for class_id in taxonomy.room_ids:
        expected = target == class_id
        if not expected.any():
            continue
        predicted = prediction == class_id
        intersection = (predicted & expected).sum().float()
        union = (predicted | expected).sum().float()
        values.append(float(intersection / union.clamp_min(1.0)))
    return sum(values) / max(1, len(values))


@torch.no_grad()
def room_area_coverage_from_labels(
    prediction: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> float:
    room_ids = torch.tensor(taxonomy.room_ids, device=prediction.device, dtype=prediction.dtype)
    inside = footprint.bool()
    room_pixels = torch.isin(prediction, room_ids) & inside
    return float(room_pixels.sum().float() / inside.sum().clamp_min(1).float())


def partition_boundary_regularizer(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float = 0.25,
) -> torch.Tensor:
    """Soft total-variation boundary loss for the dense room partition.

    Target class transitions should have high probability disagreement; pixels
    inside one room should have low disagreement. This acts on the actual room
    partition instead of applying a foreground boundary loss to all 64 classes.
    """

    valid = target >= 0
    if not valid.any() or logits.shape[1] < 2:
        return logits.sum() * 0.0
    probabilities = logits.softmax(dim=1)
    horizontal_tv = (probabilities[:, :, :, 1:] - probabilities[:, :, :, :-1]).abs().sum(dim=1)
    vertical_tv = (probabilities[:, :, 1:, :] - probabilities[:, :, :-1, :]).abs().sum(dim=1)
    horizontal_target = (
        (target[:, :, 1:] != target[:, :, :-1])
        & valid[:, :, 1:]
        & valid[:, :, :-1]
    )
    vertical_target = (
        (target[:, 1:, :] != target[:, :-1, :])
        & valid[:, 1:, :]
        & valid[:, :-1, :]
    )
    boundary_hinge = logits.sum() * 0.0
    if horizontal_target.any():
        boundary_hinge = boundary_hinge + F.relu(
            float(margin) - horizontal_tv[horizontal_target]
        ).mean()
    if vertical_target.any():
        boundary_hinge = boundary_hinge + F.relu(
            float(margin) - vertical_tv[vertical_target]
        ).mean()
    interior = ~horizontal_target & valid[:, :, 1:] & valid[:, :, :-1]
    interior_penalty = logits.sum() * 0.0
    if interior.any():
        interior_penalty = interior_penalty + horizontal_tv[interior].mean()
    interior = ~vertical_target & valid[:, 1:, :] & valid[:, :-1, :]
    if interior.any():
        interior_penalty = interior_penalty + vertical_tv[interior].mean()
    return 0.5 * (boundary_hinge + interior_penalty)


def condition_layout_loss(
    aux: dict[str, torch.Tensor | None],
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    room_class_weights: torch.Tensor | None = None,
    positive_class_weights: torch.Tensor | None = None,
    weights: ConditionObjectiveWeights | None = None,
    room_prior_logits: torch.Tensor | None = None,
    room_prior_weight: float = 0.0,
    room_support: torch.Tensor | None = None,
    occlusion_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[ObjectiveTerm]]:
    """Compute the deployment-aligned contour-to-floorplan objective."""

    weights = weights or ConditionObjectiveWeights()
    partition_logits = aux.get("room_partition_logits")
    objects = aux.get("structured_objects")
    if partition_logits is None:
        raise ValueError("condition objective requires room_partition_logits")
    if objects is None:
        raise ValueError("condition objective requires structured object heads")
    if room_prior_logits is not None and room_prior_weight > 0:
        if room_prior_logits.shape != partition_logits.shape:
            raise ValueError("room prior logits must match the partition logits")
        partition_logits = partition_logits + float(room_prior_weight) * room_prior_logits
    if room_support is None:
        partition_target = batch_room_partition_target(
            target,
            footprint,
            taxonomy,
            fill_occluded=False,
        )
        partition_pixel_weights = room_partition_confidence(
            target,
            footprint,
            taxonomy,
            visible_weight=weights.visible_room_pixel_weight,
            furniture_weight=weights.pseudo_furniture_weight,
            wall_weight=weights.pseudo_wall_weight,
            opening_weight=weights.pseudo_opening_weight,
            other_weight=weights.pseudo_other_weight,
        )
    else:
        room_ids = torch.tensor(
            taxonomy.room_ids,
            device=target.device,
            dtype=torch.long,
        )
        lookup = torch.full(
            (taxonomy.num_classes,),
            -100,
            device=target.device,
            dtype=torch.long,
        )
        lookup[room_ids] = torch.arange(len(room_ids), device=target.device)
        partition_target = lookup[target]
        partition_target = torch.where(
            footprint.bool(),
            partition_target,
            torch.full_like(partition_target, -100),
        )
        partition_pixel_weights = (partition_target >= 0).float()
    partition = room_partition_loss(
        partition_logits,
        partition_target,
        class_weights=room_class_weights,
        dice_weight=1.0,
        tversky_weight=weights.room_tversky,
        pixel_weights=partition_pixel_weights,
    )
    hierarchy_components = {}
    hierarchy_loss = partition["loss"] * 0.0
    hierarchy_predictions = aux.get("hierarchical_rooms")
    if (
        isinstance(hierarchy_predictions, dict)
        and hierarchy_predictions.get("coarse_room_logits") is not None
        and weights.room_hierarchy > 0
    ):
        coarse_logits = hierarchy_predictions["coarse_room_logits"]
        coarse_target = F.interpolate(
            partition_target.float()[:, None],
            size=coarse_logits.shape[-2:],
            mode="nearest",
        )[:, 0].long()
        coarse_pixels = F.adaptive_max_pool2d(
            partition_pixel_weights[:, None],
            coarse_logits.shape[-2:],
        )[:, 0]
        coarse_parts = room_partition_loss(
            coarse_logits,
            coarse_target,
            class_weights=room_class_weights,
            dice_weight=1.0,
            tversky_weight=weights.room_tversky,
            pixel_weights=coarse_pixels,
        )
        hierarchy_loss = float(weights.room_hierarchy) * coarse_parts["loss"]
        hierarchy_components = {
            "room_hierarchy": hierarchy_loss.detach(),
            "room_hierarchy_loss": hierarchy_loss.detach(),
            "room_hierarchy_ce": coarse_parts["partition_ce"],
            "room_hierarchy_dice": coarse_parts["partition_dice"],
        }
        if weights.room_refine > 0:
            refined_parts = room_partition_loss(
                hierarchy_predictions["refined_room_logits"],
                coarse_target,
                class_weights=room_class_weights,
                dice_weight=1.0,
                tversky_weight=weights.room_tversky,
                pixel_weights=coarse_pixels,
            )
            hierarchy_loss = (
                hierarchy_loss
                + float(weights.room_refine) * refined_parts["loss"]
            )
            hierarchy_components["room_refine"] = (
                float(weights.room_refine) * refined_parts["loss"]
            ).detach()
            hierarchy_components["room_refine_ce"] = refined_parts["partition_ce"]
            hierarchy_components["room_refine_dice"] = refined_parts["partition_dice"]
        if weights.room_transport > 0:
            _target_classes, target_tokens, target_valid = room_instance_token_targets(
                target,
                taxonomy,
            )
            predicted_tokens = hierarchy_predictions["room_region_descriptors"]
            transport_loss, transport_components = room_token_transport_loss(
                predicted_tokens,
                target_tokens,
                target_valid=target_valid,
            )
            hierarchy_components["room_transport_loss"] = transport_loss.detach()
            hierarchy_components.update(transport_components)
            hierarchy_loss = hierarchy_loss + float(weights.room_transport) * transport_loss
    polygon_components = {}
    polygon_loss = partition["loss"] * 0.0
    polygon_predictions = aux.get("polygon_loops")
    if (
        isinstance(polygon_predictions, dict)
        and weights.room_polygon > 0
    ):
        target_classes, target_polygons, target_valid = polygon_instance_targets(
            target,
            taxonomy,
            max_instances=polygon_predictions["polygons"].shape[1],
            points=polygon_predictions["polygons"].shape[2],
        )
        polygon_loss, polygon_components = polygon_loop_loss(
            polygon_predictions,
            target_classes,
            target_polygons,
            target_valid,
        )
        polygon_components["room_polygon"] = polygon_loss.detach()
    if weights.room_group > 0:
        partition_group, partition_group_components = room_group_loss(
            partition_logits,
            partition_target,
            taxonomy,
            pixel_weights=partition_pixel_weights,
        )
    else:
        partition_group = partition["loss"] * 0.0
        partition_group_components = {}
    room_group_logits = aux.get("room_group_logits")
    group_head_components = {}
    group_head_loss = partition["loss"] * 0.0
    if room_group_logits is not None and weights.room_group_head > 0:
        group_index = room_class_group_index(taxonomy, target.device)
        group_target = group_index[partition_target.clamp_min(0)]
        group_target = torch.where(partition_target >= 0, group_target, -100)
        group_head_parts = room_partition_loss(
            room_group_logits,
            group_target,
            pixel_weights=partition_pixel_weights,
            dice_weight=1.0,
        )
        group_head_loss = group_head_parts["loss"]
        group_head_components = {
            "room_group_head_ce": group_head_parts["partition_ce"],
            "room_group_head_dice": group_head_parts["partition_dice"],
        }
    boundary = partition_boundary_regularizer(partition_logits, partition_target)
    boundary_distance = boundary_distance_loss(partition_logits, partition_target)
    observable_components = {}
    observable_loss = partition["loss"] * 0.0
    if room_support is not None and weights.observable_room > 0:
        observable_loss, observable_components = observable_room_loss(
            partition_logits,
            target,
            room_support,
            footprint,
            taxonomy,
            support_weight=1.0,
        )
    geometry_components = {}
    geometry_loss = partition["loss"] * 0.0
    geometry_predictions = aux.get("geometry")
    if isinstance(geometry_predictions, dict) and weights.geometry > 0:
        geometry_targets = build_geometry_targets(target, footprint, taxonomy)
        candidates = structural_candidate_mask(target, footprint, taxonomy)
        geometry_loss, geometry_components = geometry_heatmap_loss(
            geometry_predictions,
            geometry_targets,
            structural_mask=candidates,
        )

    opening_ids = torch.tensor(taxonomy.opening_ids, device=target.device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=target.device)
    opening_weights = None
    furniture_weights = None
    if positive_class_weights is not None:
        class_weights = positive_class_weights.to(target.device)
        opening_weights = class_weights[opening_ids - 1]
        furniture_weights = class_weights[furniture_ids - 1]
        opening_weights = opening_weights / opening_weights.mean().clamp_min(1.0e-6)
        furniture_weights = furniture_weights / furniture_weights.mean().clamp_min(1.0e-6)
    opening, opening_ce, opening_dice = group_ce_dice_loss(
        objects["opening_logits"],
        target,
        opening_ids,
        footprint,
        opening_weights,
        objects.get("opening_background_logits"),
    )
    opening_candidate, opening_candidate_component = opening_candidate_loss(
        objects,
        target,
        footprint,
        taxonomy,
    )
    furniture, furniture_ce, furniture_dice = group_ce_dice_loss(
        objects["furniture_logits"],
        target,
        furniture_ids,
        footprint,
        furniture_weights,
        objects.get("furniture_background_logits"),
    )
    wall, wall_bce, wall_tversky = interior_wall_loss(
        objects["wall_logits"],
        target,
        footprint,
        taxonomy,
        target_dilation=0,
    )
    structural, structural_components = structural_consistency_loss(
        partition_logits,
        objects,
        target,
        footprint,
        taxonomy,
    )
    terms = [
        ObjectiveTerm("room_partition", partition["loss"], weights.room_partition),
        ObjectiveTerm("room_boundary", boundary, weights.room_boundary),
        ObjectiveTerm(
            "room_boundary_distance",
            boundary_distance,
            weights.room_boundary_distance,
        ),
        ObjectiveTerm("opening", opening, weights.opening),
        ObjectiveTerm(
            "opening_candidate",
            opening_candidate,
            0.5 * weights.opening,
        ),
        ObjectiveTerm("furniture", furniture, weights.furniture),
        ObjectiveTerm("wall", wall, weights.wall),
        ObjectiveTerm(
            "structural_consistency",
            structural,
            weights.structural_consistency,
        ),
    ]
    if geometry_components:
        terms.append(
            ObjectiveTerm(
                "geometry",
                geometry_loss,
                weights.geometry,
            )
        )
    if observable_components:
        terms.append(
            ObjectiveTerm(
                "observable_room",
                observable_loss,
                weights.observable_room,
            )
        )
    if hierarchy_components:
        terms.insert(
            1,
            ObjectiveTerm(
                "room_hierarchy",
                hierarchy_loss,
                1.0,
            ),
        )
    if polygon_components:
        terms.insert(
            2,
            ObjectiveTerm(
                "room_polygon",
                polygon_loss,
                weights.room_polygon,
            ),
        )
    room_mask_predictions = aux.get("room_mask_predictions")
    room_mask_components = {}
    room_mask_total = partition["loss"] * 0.0
    room_mask_partition_total = partition["loss"] * 0.0
    if room_mask_predictions is not None:
        target_classes, target_masks, target_valid = room_mask_targets(
            target,
            taxonomy,
            max_rooms=room_mask_predictions["class_logits"].shape[1],
        )
        room_mask_total, room_mask_components = room_mask_loss(
            room_mask_predictions,
            target_classes,
            target_masks,
            target_valid,
            class_weights=room_class_weights,
        )
        if weights.room_mask_partition > 0:
            room_mask_partition_total, partition_components = room_mask_partition_loss(
                room_mask_predictions,
                target,
                footprint,
                taxonomy,
            )
            room_mask_components.update(partition_components)
        room_mask_combined = (
            float(weights.room_mask) * room_mask_total
            + float(weights.room_mask_partition) * room_mask_partition_total
        )
        terms.append(
            ObjectiveTerm(
                "room_mask",
                room_mask_combined,
                1.0,
            )
        )
        room_mask_components["room_mask_objective"] = room_mask_combined.detach()
    if weights.room_group > 0:
        terms.insert(1, ObjectiveTerm("room_group", partition_group, weights.room_group))
    if room_group_logits is not None and weights.room_group_head > 0:
        terms.insert(1, ObjectiveTerm("room_group_head", group_head_loss, weights.room_group_head))
    program = aux.get("room_program")
    program_components = None
    if program is not None:
        program_loss, program_components = room_program_loss(
            program,
            target,
            taxonomy,
            class_weights=room_class_weights,
        )
        terms.insert(
            2,
            ObjectiveTerm("room_program", program_loss, weights.room_program),
        )
    total, components = compose_objective(terms)
    if room_mask_predictions is not None:
        components.update(room_mask_components)
    if program_components is not None:
        components.update(program_components)
    components.update(partition_group_components)
    components.update(group_head_components)
    components.update(hierarchy_components)
    components.update(polygon_components)
    components.update(geometry_components)
    components.update(structural_components)
    components.update(observable_components)
    components.update(
        {
            "room_partition_ce": partition["partition_ce"],
            "room_partition_dice": partition["partition_dice"],
            "opening_ce": opening_ce,
            "opening_dice": opening_dice,
            "opening_candidate": opening_candidate_component,
            "furniture_ce": furniture_ce,
            "furniture_dice": furniture_dice,
            "wall_bce": wall_bce,
            "wall_tversky": wall_tversky,
        }
    )
    dense_metrics = room_partition_metrics(partition_logits, partition_target, taxonomy)
    components["room_partition_dense_iou"] = torch.tensor(
        dense_metrics["room_partition_iou"],
        device=target.device,
    )
    return total, components, terms
