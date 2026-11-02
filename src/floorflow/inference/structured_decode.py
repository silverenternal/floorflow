"""Inference composition for jointly trained structured floorplan heads."""

from __future__ import annotations

import torch

from floorflow.geometry.core import compose_known_exterior
from floorflow.inference.instance_semantics import consolidate_room_instances
from floorflow.representation.program_conditioning import (
    apply_program_constraint,
    assign_instances_to_program,
)
from floorflow.representation.polygon_targets import decode_polygon_loops
from floorflow.heads.room_mask_decoder import render_room_masks
from floorflow.heads.structured_objects import (
    compose_structured_objects,
    constrain_structured_objects,
    object_group_predictions,
)
from floorflow.data.taxonomy import FloorplanTaxonomy


@torch.no_grad()
def decode_structured_floorplan(
    model,
    reference: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    floor_ids: torch.Tensor | None = None,
    opening_threshold: float | torch.Tensor = 0.0,
    furniture_threshold: float | torch.Tensor = 0.0,
    wall_threshold: float = 0.0,
    object_score_scale: str = "logit",
    structural_gate: bool = False,
    use_predicted_walls: bool = True,
    require_compatible_furniture: bool = True,
    partition_wall_radius: int = 0,
    return_room_support: bool = False,
    return_debug: bool = False,
    room_prior_logits: torch.Tensor | None = None,
    room_prior_weight: float = 0.0,
    room_prior_weights: torch.Tensor | None = None,
    retrieval_labels: torch.Tensor | None = None,
    opening_prior_weight: float = 0.0,
    furniture_prior_weight: float = 0.0,
    wall_prior_weight: float = 0.0,
    object_source: str = "auto",
    opening_source: str = "inherit",
    furniture_source: str = "inherit",
    room_source: str = "mask",
    state: torch.Tensor | None = None,
    wall_mode: str = "heads",
    use_geometry_walls: bool = False,
    geometry_wall_threshold: float = 0.7,
    geometry_wall_dilate: int = 1,
    instance_semantic_vote: bool = False,
    room_program: torch.Tensor | None = None,
    room_program_control: torch.Tensor | None = None,
    room_program_counts: torch.Tensor | None = None,
    coarse_layout_logits: torch.Tensor | None = None,
    coarse_layout_weight: float = 0.0,
    program_absent_penalty: float = 8.0,
) -> torch.Tensor:
    sampled_state = state is not None
    if state is None:
        state = torch.zeros(
            reference.shape[0],
            model.target_channels,
            *reference.shape[-2:],
            device=reference.device,
            dtype=footprint.dtype,
        )
    if state.shape != (
        reference.shape[0],
        model.target_channels,
        *reference.shape[-2:],
    ):
        raise ValueError("structured decoder state has an incompatible shape")
    _, aux = model.forward_with_aux(
        state,
        torch.full(
            (reference.shape[0],),
            1.0 if sampled_state else 0.0,
            device=reference.device,
            dtype=reference.dtype,
        ),
        footprint,
        floor_ids=floor_ids,
        layout_prior_logits=room_prior_logits,
        room_program=room_program_control,
    )
    object_logits = aux.get("structured_objects")
    if retrieval_labels is not None and object_logits is not None:
        retrieval_labels = retrieval_labels.to(device=footprint.device)
        object_logits = dict(object_logits)
        for name, class_ids, prior_weight in (
            (
                "opening",
                taxonomy.opening_ids,
                opening_prior_weight,
            ),
            (
                "furniture",
                taxonomy.furniture_ids,
                furniture_prior_weight,
            ),
        ):
            if prior_weight <= 0:
                continue
            ids = torch.tensor(class_ids, device=footprint.device)
            prior = torch.stack(
                [(retrieval_labels == class_id).float() for class_id in ids],
                dim=1,
            )
            object_logits[f"{name}_logits"] = object_logits[f"{name}_logits"] + prior_weight * prior
        if wall_prior_weight > 0:
            wall_prior = (
                2.0 * (retrieval_labels == taxonomy.class_id("wall")).float()
                - 1.0
            )
            object_logits["wall_logits"] = (
                object_logits["wall_logits"] + wall_prior_weight * wall_prior
            )
    partition_logits = aux.get("room_partition_logits")
    if partition_logits is not None and room_prior_logits is not None and room_prior_weight > 0:
        if room_prior_logits.shape != partition_logits.shape:
            raise ValueError(
                f"room prior shape {tuple(room_prior_logits.shape)} does not match "
                f"partition shape {tuple(partition_logits.shape)}"
            )
        prior = room_prior_logits
        if room_prior_weights is not None:
            class_weights = room_prior_weights.to(
                device=prior.device,
                dtype=prior.dtype,
            )
            if class_weights.numel() != prior.shape[1]:
                raise ValueError("room_prior_weights must match room classes")
            prior = prior * class_weights[None, :, None, None]
        partition_logits = partition_logits + float(room_prior_weight) * prior
    if partition_logits is not None and coarse_layout_logits is not None:
        partition_logits = partition_logits + float(coarse_layout_weight) * coarse_layout_logits
    if partition_logits is not None and room_program is not None:
        partition_logits = apply_program_constraint(
            partition_logits,
            room_program,
            absent_penalty=program_absent_penalty,
        )
    if object_logits is None:
        raise ValueError("checkpoint does not expose structured-object heads")
    hierarchy = aux.get("hierarchical_rooms")
    if room_source == "semantic" and aux.get("semantic_logits") is not None:
        semantic_logits = aux["semantic_logits"]
        semantic_labels = torch.cat(
            [torch.zeros_like(semantic_logits[:, :1]), semantic_logits],
            dim=1,
        ).argmax(dim=1)
        semantic_room_ids = torch.tensor(
            taxonomy.room_ids,
            device=footprint.device,
        )
        rooms = torch.where(
            torch.isin(semantic_labels, semantic_room_ids),
            semantic_labels,
            torch.zeros_like(semantic_labels),
        ) * footprint.long()
    elif (
        room_source == "hierarchical"
        and isinstance(hierarchy, dict)
        and hierarchy.get("refined_room_logits") is not None
    ):
        refined_room_logits = hierarchy["refined_room_logits"]
        refined_logits = torch.nn.functional.interpolate(
            refined_room_logits,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        room_ids = torch.tensor(taxonomy.room_ids, device=footprint.device)
        rooms = room_ids[refined_logits.argmax(dim=1)] * footprint.long()
    elif room_source == "polygon" and aux.get("polygon_loops") is not None:
        rooms = decode_polygon_loops(
            aux["polygon_loops"],
            taxonomy,
            height=reference.shape[-2],
            width=reference.shape[-1],
            validity_threshold=0.0,
        )
        rooms = rooms * footprint.long()
    elif room_source == "partition" and partition_logits is not None:
        room_ids = torch.tensor(taxonomy.room_ids, device=footprint.device)
        rooms = room_ids[partition_logits.argmax(dim=1)] * footprint.long()
        if instance_semantic_vote:
            rooms = consolidate_room_instances(
                rooms,
                partition_logits,
                footprint,
                taxonomy.room_ids,
            )
    else:
        room_predictions = aux.get("room_mask_predictions")
        if room_predictions is None:
            raise ValueError("checkpoint does not expose a room-mask or room-partition head")
        rooms = render_room_masks(room_predictions, taxonomy, footprint)
    raw_objects = object_group_predictions(
        object_logits,
        taxonomy,
        opening_threshold=opening_threshold,
        furniture_threshold=furniture_threshold,
        wall_threshold=wall_threshold,
        score_scale=object_score_scale,
    )
    objects = raw_objects
    geometry = aux.get("geometry")
    if use_geometry_walls and isinstance(geometry, dict):
        wall_logits = geometry.get("wall_centerline_logits")
        if wall_logits is not None:
            geometry_wall = (
                wall_logits.sigmoid() >= float(geometry_wall_threshold)
            ).long()
            radius = max(0, int(geometry_wall_dilate))
            if radius > 0:
                geometry_wall = torch.nn.functional.max_pool2d(
                    geometry_wall.float(),
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                ).long()
            objects["wall"] = (
                objects["wall"].bool()
                | geometry_wall[:, 0].bool()
            ).long()
    semantic_logits = aux.get("semantic_logits")
    if semantic_logits is not None:
        semantic = torch.cat(
            [torch.zeros_like(semantic_logits[:, :1]), semantic_logits],
            dim=1,
        ).argmax(dim=1)
        opening_mode = (
            object_source if opening_source == "inherit" else opening_source
        )
        furniture_mode = (
            object_source if furniture_source == "inherit" else furniture_source
        )
        opening_semantic = opening_mode == "semantic" or (
            opening_mode == "auto" and objects["opening"].max() == 0
        )
        furniture_semantic = furniture_mode == "semantic" or (
            furniture_mode == "auto" and objects["furniture"].max() == 0
        )
        if opening_semantic:
            objects["opening"] = torch.where(
                torch.isin(semantic, torch.tensor(taxonomy.opening_ids, device=semantic.device)),
                semantic,
                torch.zeros_like(semantic),
            )
        if furniture_semantic:
            objects["furniture"] = torch.where(
                torch.isin(semantic, torch.tensor(taxonomy.furniture_ids, device=semantic.device)),
                semantic,
                torch.zeros_like(semantic),
            )
    if structural_gate:
        objects = constrain_structured_objects(
            objects,
            rooms,
            footprint,
            taxonomy,
            use_predicted_walls=wall_mode != "partition",
            require_compatible_furniture=require_compatible_furniture,
            wall_radius=partition_wall_radius,
        )
        if wall_mode == "union":
            from floorflow.heads.structured_objects import partition_wall_mask

            transition_wall = partition_wall_mask(
                rooms,
                radius=max(1, int(partition_wall_radius)),
            )
            objects["wall"] = (
                objects["wall"].bool() | transition_wall
            ).long()
    labels = compose_structured_objects(rooms, objects, footprint, taxonomy)
    labels = compose_known_exterior(labels, footprint, opening_ids=taxonomy.opening_ids)
    if room_program_counts is not None and partition_logits is not None:
        labels = assign_instances_to_program(
            labels,
            partition_logits,
            room_program_counts,
            taxonomy.room_ids,
        )
    if return_debug:
        return labels, rooms, raw_objects, objects
    if return_room_support:
        return labels, rooms
    return labels
