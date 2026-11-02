"""Reusable deterministic pixel/element/retrieval overlay decoder."""

from __future__ import annotations

import torch

from floorflow.inference.distribution_ensemble import (
    blend_distribution_outputs,
)
from floorflow.inference.distribution_overlay import (
    DistributionOverlaySettings,
    compose_distribution_overlay,
    element_family_scores,
    overlay_room_labels,
)
from floorflow.inference.instance_semantics import (
    consolidate_room_instances_with_prior,
)
from floorflow.inference.target_free import target_free_generation
from floorflow.representation.contour_memory import ContourPartitionMemory


@target_free_generation
@torch.inference_mode()
def decode_overlay(
    model,
    memory: ContourPartitionMemory | None,
    retrieval_weights: torch.Tensor | None,
    footprint: torch.Tensor,
    taxonomy,
    *,
    floor_ids: torch.Tensor | None = None,
    style_ids: torch.Tensor | None = None,
    retrieval_prior_weight: float = 0.0,
    room_program: torch.Tensor | None = None,
    room_program_weight: float = 0.0,
    room_prior_logits: torch.Tensor | None = None,
    room_prior_weight: float = 0.0,
    room_prior_confidence_threshold: float = 1.0,
    room_partition_weight: float = 0.0,
    ensemble_model=None,
    ensemble_weight: float = 0.0,
    settings: DistributionOverlaySettings | None = None,
) -> dict[str, object]:
    """Decode one batch without target information or best-of-N selection."""
    if not 0.0 <= float(room_prior_confidence_threshold) <= 1.0:
        raise ValueError("room prior confidence threshold must be in [0, 1]")
    settings = settings or DistributionOverlaySettings(
        opening_element_weight=0.0
    )
    distribution = model.contour_initial_state(
        footprint,
        floor_ids=floor_ids,
        style_ids=style_ids,
    )
    if ensemble_model is not None and ensemble_weight > 0.0:
        secondary = ensemble_model.contour_initial_state(
            footprint,
            floor_ids=floor_ids,
            style_ids=style_ids,
        )
        distribution = blend_distribution_outputs(
            distribution,
            secondary,
            ensemble_weight,
        )
    pixel_logits = distribution["pixel"]["logits"]
    conditioned_logits = pixel_logits
    room_ids = tuple(model.hub.hierarchy.room_ids)
    stale_room_confidence = pixel_logits[:, room_ids].softmax(dim=1).amax(dim=1)
    prior_gate = (
        stale_room_confidence <= float(room_prior_confidence_threshold)
    ).to(pixel_logits.dtype)[:, None]
    if room_program is not None and room_program_weight > 0.0:
        program = room_program.to(
            device=pixel_logits.device,
            dtype=pixel_logits.dtype,
        )
        if program.shape != (pixel_logits.shape[0], len(room_ids)):
            raise ValueError("room program shape does not match room hierarchy")
        conditioned_logits = pixel_logits.clone()
        conditioned_logits[:, room_ids] = (
            conditioned_logits[:, room_ids]
            + float(room_program_weight)
            * prior_gate
            * (2.0 * program - 1.0)[:, :, None, None]
        )
    if room_prior_logits is not None and room_prior_weight > 0.0:
        prior = torch.nn.functional.interpolate(
            room_prior_logits,
            size=pixel_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if prior.shape[1] != len(room_ids):
            raise ValueError("room prior channel count does not match hierarchy")
        conditioned_logits = conditioned_logits.clone()
        conditioned_logits[:, room_ids] = (
            conditioned_logits[:, room_ids]
            + float(room_prior_weight)
            * prior_gate
            * prior.log_softmax(dim=1).to(conditioned_logits.dtype)
        )
    partition_logits = distribution.get("room_partition_logits")
    if room_partition_weight > 0.0 and partition_logits is not None:
        partition_logits = torch.nn.functional.interpolate(
            partition_logits,
            size=pixel_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if partition_logits.shape[1] != len(room_ids):
            raise ValueError("room partition channel count does not match hierarchy")
        conditioned_logits = conditioned_logits.clone()
        conditioned_logits[:, room_ids] = (
            conditioned_logits[:, room_ids]
            + float(room_partition_weight)
            * partition_logits.to(conditioned_logits.dtype).log_softmax(dim=1)
        )
    scores = element_family_scores(
        distribution["elements"],
        model.hub.hierarchy,
        pixel_logits.shape[-2:],
    )
    raw_rooms = torch.tensor(room_ids, device=pixel_logits.device)[
        pixel_logits[:, room_ids].argmax(dim=1)
    ] * footprint[:, 0].long()
    element_rooms = overlay_room_labels(
        conditioned_logits,
        scores["room"],
        room_ids,
        footprint[:, 0],
        settings,
    )
    use_retrieval = float(retrieval_prior_weight) > 0.0
    if use_retrieval and (memory is None or retrieval_weights is None):
        raise ValueError(
            "retrieval_prior_weight requires memory and retrieval_weights"
        )
    if use_retrieval:
        raw_prior = memory.retrieve_logits(
            footprint[:, 0].cpu().numpy(),
            taxonomy,
            pixel_logits.device,
        )
        retrieval_logits = (
            conditioned_logits[:, room_ids]
            + float(retrieval_prior_weight)
            * raw_prior
            * retrieval_weights[None, :, None, None]
        )
    else:
        raw_prior = conditioned_logits.new_zeros(
            (
                conditioned_logits.shape[0],
                len(room_ids),
                *conditioned_logits.shape[-2:],
            )
        )
        retrieval_logits = conditioned_logits[:, room_ids]
    retrieval_rooms = torch.tensor(
        room_ids,
        device=pixel_logits.device,
    )[retrieval_logits.argmax(dim=1)]
    if use_retrieval:
        retrieval_rooms = (
            consolidate_room_instances_with_prior(
                retrieval_rooms,
                conditioned_logits[:, room_ids],
                raw_prior,
                footprint,
                room_ids,
                retrieval_weights,
            )
            * footprint[:, 0].long()
        )
    fused_rooms = torch.where(
        (retrieval_rooms == 0) & (element_rooms != 0),
        element_rooms,
        retrieval_rooms,
    )
    raw_pixel_labels = pixel_logits.argmax(dim=1) * footprint[:, 0].long()
    labels, layers = compose_distribution_overlay(
        fused_rooms,
        scores,
        footprint[:, 0],
        taxonomy,
        settings,
        pixel_labels=raw_pixel_labels,
    )
    return {
        "labels": labels,
        "layers": layers,
        "pixel_labels": raw_pixel_labels,
        "raw_rooms": raw_rooms,
        "element_rooms": element_rooms,
        "retrieval_rooms": retrieval_rooms,
        "retrieval_prior": raw_prior,
        "retrieval_used": use_retrieval,
        "conditioned_logits": conditioned_logits,
        "room_program": room_program,
        "room_prior_logits": room_prior_logits,
        "room_prior_gate": prior_gate,
        "distribution": distribution,
    }


__all__ = ["decode_overlay"]
