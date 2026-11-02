"""Shared reporting helpers for predicted-control evaluations."""

from __future__ import annotations

import numpy as np

from floorflow.inference.opening_heatmap_decoder import (
    predict_opening_detection_maps,
    predict_opening_family_probabilities,
)


def numeric_row_summary(rows: list[dict]) -> dict[str, float | int]:
    """Summarize numeric per-case fields with mean and sample standard deviation."""

    numeric = {
        key: np.asarray(
            [row[key] for row in rows],
            dtype=np.float64,
        )
        for key in rows[0]
        if key != "case"
        and all(
            isinstance(row[key], (int, float, np.integer, np.floating))
            for row in rows
        )
    }
    return {
        "cases": len(rows),
        **{
            f"{key}_mean": float(values.mean())
            for key, values in numeric.items()
        },
        **{
            f"{key}_std": float(values.std(ddof=1))
            if len(values) > 1
            else 0.0
            for key, values in numeric.items()
        },
    }


def opening_metric_kwargs(
    args,
    index: int,
    structure: dict,
    opening_probability,
    prediction_plans,
    opening_host_plans,
    opening_foreground=None,
    opening_family_logits=None,
    opening_candidate_ranker=None,
    opening_center=None,
    opening_interval_width=None,
    opening_precomputed_detections=None,
) -> dict:
    """Resolve mutually exclusive opening sources for one evaluated case."""

    return {
        "prediction_opening_probability": (
            structure["opening_logits"][index, 0].sigmoid().cpu().numpy()
            if args.interval_opening_readout
            else None
        ),
        "prediction_opening_logits": (
            structure["opening_type_logits"][index].cpu().numpy()
            if args.interval_opening_readout
            else None
        ),
        "prediction_opening_threshold": args.interval_opening_threshold,
        "prediction_opening_heatmap": (
            opening_probability[index].numpy()
            if opening_probability is not None
            else None
        ),
        "prediction_opening_foreground": (
            opening_foreground[index].numpy()
            if opening_foreground is not None
            else None
        ),
        "prediction_opening_family_logits": (
            opening_family_logits[index].numpy()
            if opening_family_logits is not None
            else None
        ),
        "prediction_opening_family_aggregation": (
            args.opening_family_aggregation
        ),
        "prediction_opening_exterior_window_bias": (
            args.opening_exterior_window_bias
        ),
        "prediction_opening_interval_radius": args.opening_interval_radius,
        "prediction_opening_score_smoothing": args.opening_score_smoothing,
        "prediction_opening_max_gap_samples": args.opening_max_gap_samples,
        "prediction_opening_peak_relative_threshold": (
            args.opening_peak_relative_threshold
        ),
        "prediction_opening_candidate_ranker": opening_candidate_ranker,
        "prediction_opening_center": (
            opening_center[index].numpy()
            if opening_center is not None
            else None
        ),
        "prediction_opening_interval_width": (
            opening_interval_width[index].numpy()
            if opening_interval_width is not None
            else None
        ),
        "prediction_opening_precomputed_detections": (
            opening_precomputed_detections[index]
            if opening_precomputed_detections is not None
            else None
        ),
        "prediction_opening_min_peak_distance": (
            args.opening_heatmap_min_peak_distance
        ),
        "prediction_opening_center_decode_mode": (
            args.opening_center_decode_mode
        ),
        "prediction_opening_remap_final_wall": (
            args.opening_remap_final_wall
        ),
        "prediction_opening_heatmap_threshold": (
            float(args.opening_heatmap_threshold)
            if args.opening_heatmap_threshold is not None
            else 0.30
        ),
        "prediction_opening_min_area": args.opening_heatmap_min_area,
        "prediction_opening_max_wall_distance": (
            args.opening_heatmap_max_wall_distance
        ),
        "prediction_opening_host_plan": (
            opening_host_plans[index]
            if opening_host_plans is not None
            else None
        ),
        "prediction_plan": (
            prediction_plans[index]
            if prediction_plans is not None
            else None
        ),
        "ip_wall_reconcile": bool(args.ip_wall_reconcile),
        "owner_cell_wall_plan": bool(args.owner_cell_wall_plan),
        "owner_cell_edge_support_threshold": args.owner_cell_edge_support,
        "owner_cell_support_radius": args.owner_cell_support_radius,
        "owner_cell_minimum_area": args.owner_cell_min_area,
        "near_duplicate_wall_prune": bool(args.near_duplicate_wall_prune),
        "near_duplicate_wall_distance_fraction": (
            args.near_duplicate_wall_distance_fraction
        ),
        "near_duplicate_wall_angle_degrees": (
            args.near_duplicate_wall_angle_degrees
        ),
        "near_duplicate_wall_overlap": args.near_duplicate_wall_overlap,
        "ip_wall_use_milp": bool(args.ip_wall_milp),
        "ip_wall_tolerance_fraction": args.ip_wall_tolerance_fraction,
        "ip_wall_merge_gap_fraction": args.ip_wall_merge_gap_fraction,
    }


def opening_runtime_tensors(args, runtime, footprint, image, taxonomy):
    """Resolve legacy and independent-foreground opening score maps."""

    if runtime is None:
        return None, None, None, None, None
    if args.opening_heatmap_independent_foreground:
        maps = predict_opening_detection_maps(
            runtime,
            footprint,
            image,
            taxonomy,
            window_logit_bias=float(args.opening_family_window_bias),
        )
        return (
            maps["family_probability"].cpu(),
            maps["foreground"].cpu(),
            maps["family_logits"].cpu(),
            maps["center"].cpu() if "center" in maps else None,
            (
                maps["interval_width"].cpu()
                if "interval_width" in maps
                else None
            ),
        )
    probability = predict_opening_family_probabilities(
        runtime,
        footprint,
        image,
        taxonomy,
        window_logit_bias=float(args.opening_family_window_bias),
    ).cpu()
    return probability, None, None, None, None


__all__ = [
    "numeric_row_summary",
    "opening_metric_kwargs",
    "opening_runtime_tensors",
]
