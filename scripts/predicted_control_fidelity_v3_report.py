"""Serializable configuration for predicted-control fidelity v3 reports."""

from __future__ import annotations

from argparse import Namespace


def report_config(
    args: Namespace,
    gate_threshold: float | None,
    structure_config: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "seed": args.seed,
        "predicted_root": str(args.predicted_root),
        "data_root": str(args.data_root),
        "cache_root": str(args.cache_root),
        "structure_checkpoint": str(args.structure_checkpoint),
        "structure_contract": (
            (structure_config or {}).get("structure_contract", "legacy")
        ),
        "structure_global_route_attention": bool(
            (structure_config or {}).get(
                "structure_global_route_attention",
                False,
            )
        ),
        "room_checkpoint": str(args.room_checkpoint),
        "room_query_checkpoint": str(args.room_query_checkpoint),
        "room_identity_gate": (
            str(args.room_identity_gate)
            if args.room_identity_gate is not None
            else None
        ),
        "room_identity_gate_threshold": gate_threshold,
        "room_identity_safety_fallback": args.room_identity_safety_fallback,
        "room_safety_min_wall_continuity": (
            args.room_safety_min_wall_continuity
        ),
        "wall_centerline_checkpoint": (
            str(args.wall_centerline_checkpoint)
            if args.wall_centerline_checkpoint is not None
            else None
        ),
        "wall_centerline_threshold": args.wall_centerline_threshold,
        "wall_centerline_dilation": args.wall_centerline_dilation,
        "wall_centerline_opening_protect_radius": (
            args.wall_centerline_opening_protect_radius
        ),
        "wall_centerline_mode": args.wall_centerline_mode,
        "structured_object_checkpoint": (
            str(args.structured_object_checkpoint)
            if args.structured_object_checkpoint is not None
            else None
        ),
        "structured_opening_threshold": args.structured_opening_threshold,
        "interval_opening_readout": args.interval_opening_readout,
        "interval_opening_threshold": args.interval_opening_threshold,
        "opening_heatmap_checkpoint": (
            str(args.opening_heatmap_checkpoint)
            if args.opening_heatmap_checkpoint is not None
            else None
        ),
        "opening_heatmap_threshold": args.opening_heatmap_threshold,
        "opening_heatmap_independent_foreground": (
            args.opening_heatmap_independent_foreground
        ),
        "opening_family_window_bias": args.opening_family_window_bias,
        "opening_exterior_window_bias": args.opening_exterior_window_bias,
        "opening_interval_radius": args.opening_interval_radius,
        "opening_score_smoothing": args.opening_score_smoothing,
        "opening_max_gap_samples": args.opening_max_gap_samples,
        "opening_peak_relative_threshold": (
            args.opening_peak_relative_threshold
        ),
        "opening_heatmap_candidate_ranker": (
            str(args.opening_heatmap_candidate_ranker)
            if args.opening_heatmap_candidate_ranker is not None
            else None
        ),
        "opening_heatmap_candidate_ranker_threshold": (
            args.opening_heatmap_candidate_ranker_threshold
        ),
        "opening_heatmap_min_peak_distance": (
            args.opening_heatmap_min_peak_distance
        ),
        "opening_center_decode_mode": args.opening_center_decode_mode,
        "opening_ranker_batch_mode": args.opening_ranker_batch_mode,
        "limit": args.limit,
        "offset": args.offset,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "room_size": args.room_size,
        "room_instance_blend": args.room_instance_blend,
        "room_instance_route": args.room_instance_route,
        "room_mask_threshold": args.room_mask_threshold,
        "room_mask_strength": args.room_mask_strength,
        "structure_room_source": args.structure_room_source,
        "structure_size": args.structure_size,
        "image_size": args.image_size,
        "room_base_channels": args.room_base_channels,
        "structure_base_channels": args.structure_base_channels,
        "boundary_width": args.boundary_width,
        "reconcile_gap_fraction": args.reconcile_gap_fraction,
        "wall_skeleton_close_kernel": args.wall_skeleton_close_kernel,
        "wall_skeleton_simplification_fraction": (
            args.wall_skeleton_simplification_fraction
        ),
        "wall_skeleton_min_segment_length": (
            args.wall_skeleton_min_segment_length
        ),
        "room_merge_min_area": args.room_merge_min_area,
        "include_room_decomposition": args.include_room_decomposition,
        "fpeval_tolerance": 0.05,
        "ged_evaluator": "housemind-style-deterministic-v2",
        "ged_legacy_timeout_seconds": args.ged_timeout_seconds,
    }
