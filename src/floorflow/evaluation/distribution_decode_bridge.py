"""Compose the repeated deterministic distribution decode arguments."""

from __future__ import annotations

from floorflow.inference.overlay_decoder import decode_overlay


def decode_distribution_candidate(
    model,
    memory,
    retrieval_weights,
    footprint,
    taxonomy,
    *,
    args,
    floor_ids,
    style_ids,
    room_prior_logits,
    program,
    ensemble_model,
    settings,
):
    return decode_overlay(
        model,
        memory,
        retrieval_weights,
        footprint,
        taxonomy,
        floor_ids=floor_ids,
        style_ids=style_ids,
        retrieval_prior_weight=args.retrieval_prior_weight,
        room_program=program,
        room_program_weight=args.program_prior_weight,
        room_prior_logits=room_prior_logits,
        room_prior_weight=args.room_prior_weight,
        room_prior_confidence_threshold=args.room_prior_confidence_threshold,
        room_partition_weight=args.room_partition_weight,
        ensemble_model=ensemble_model,
        ensemble_weight=args.ensemble_weight,
        settings=settings,
    )


__all__ = ["decode_distribution_candidate"]
