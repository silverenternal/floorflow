"""Canonical target-comparative metrics for predicted-control CAD outputs."""

from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fpeval_bridge import (
    attach_wall_interval_openings,
    evaluate,
    to_fpeval_plan,
)
from floorflow.evaluation.generation_plausibility import generation_plausibility
from floorflow.evaluation.housemind_structure import (
    housemind_style_metrics,
    housemind_style_metrics_from_plan,
)
from floorflow.evaluation.room_corner_metrics import room_corner_angle_metrics
from floorflow.evaluation.room_error_decomposition import decompose_rooms
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.geometry.ip_reconcile import ip_reconcile_plan
from floorflow.geometry.partition_polygons import explicit_partition_plan
from floorflow.geometry.reconciliation import reconcile_plan
from floorflow.geometry.wall_deduplication import prune_near_duplicate_walls
from floorflow.geometry.wall_graph_readout import WallSkeletonPolicy
from floorflow.inference.opening_center_width_decoder import (
    attach_center_width_opening_detections,
)
from floorflow.inference.opening_heatmap_decoder import (
    attach_learned_opening_detections,
    attach_learned_opening_detections_from_maps,
)
from floorflow.inference.structure_compositor import owner_cell_plan


def predicted_control_explicit_plan(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    reconcile_gap_fraction: float | None,
    wall_policy: WallSkeletonPolicy | None = None,
):
    plan = explicit_partition_plan(
        labels,
        footprint,
        taxonomy,
        wall_policy=wall_policy,
    )
    if reconcile_gap_fraction is None:
        return plan
    rooms = plan.rooms
    plan = reconcile_plan(
        plan,
        labels,
        taxonomy,
        gap_tolerance_fraction=float(reconcile_gap_fraction),
    )
    plan.rooms = rooms
    plan.room_source = f"{plan.room_source}_walls_reconciled"
    return plan


def predicted_control_fidelity_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    reconcile_gap_fraction: float | None = 0.04,
    fpeval_tolerance: float = 0.05,
    graph_ged_timeout_seconds: float = 5.0,
    graph_source: str = "raster",
    include_room_decomposition: bool = False,
    prediction_opening_probability: np.ndarray | None = None,
    prediction_opening_logits: np.ndarray | None = None,
    prediction_opening_threshold: float = 0.5,
    prediction_opening_heatmap: np.ndarray | None = None,
    prediction_opening_heatmap_threshold: float = 0.30,
    prediction_opening_min_area: int = 7,
    prediction_opening_max_wall_distance: float = 4.0,
    prediction_opening_foreground: np.ndarray | None = None,
    prediction_opening_family_logits: np.ndarray | None = None,
    prediction_opening_family_aggregation: str = "center",
    prediction_opening_exterior_window_bias: float = 0.0,
    prediction_opening_interval_radius: int = 1,
    prediction_opening_score_smoothing: int = 3,
    prediction_opening_max_gap_samples: int = 2,
    prediction_opening_peak_relative_threshold: float = 0.0,
    prediction_opening_candidate_ranker=None,
    prediction_opening_min_peak_distance: float = 4.0,
    prediction_opening_center_decode_mode: str = "wall",
    prediction_opening_precomputed_detections=None,
    prediction_opening_remap_final_wall: bool = False,
    prediction_opening_center: np.ndarray | None = None,
    prediction_opening_interval_width: np.ndarray | None = None,
    prediction_opening_host_plan=None,
    prediction_plan=None,
    ip_wall_reconcile: bool = False,
    ip_wall_use_milp: bool = True,
    ip_wall_tolerance_fraction: float = 0.006,
    ip_wall_merge_gap_fraction: float = 0.02,
    owner_cell_wall_plan: bool = False,
    owner_cell_edge_support_threshold: float = 0.25,
    owner_cell_support_radius: int = 1,
    owner_cell_minimum_area: int = 32,
    near_duplicate_wall_prune: bool = False,
    near_duplicate_wall_distance_fraction: float = 0.006,
    near_duplicate_wall_angle_degrees: float = 4.0,
    near_duplicate_wall_overlap: float = 0.65,
    wall_readout_policy: WallSkeletonPolicy | None = None,
) -> dict[str, float | int | bool | str | list[str]]:
    """Evaluate raster fidelity, CAD fidelity, and structural plausibility."""

    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    footprint = np.asarray(footprint, dtype=bool)
    if prediction_plan is not None:
        predicted_plan = prediction_plan
    elif owner_cell_wall_plan:
        predicted_plan = owner_cell_plan(
            prediction,
            footprint,
            taxonomy,
            minimum_area=owner_cell_minimum_area,
            wall_support_radius=owner_cell_support_radius,
            edge_support_threshold=owner_cell_edge_support_threshold,
        )
    else:
        predicted_plan = predicted_control_explicit_plan(
            prediction,
            footprint,
            taxonomy,
            reconcile_gap_fraction=reconcile_gap_fraction,
            wall_policy=wall_readout_policy,
        )
    if near_duplicate_wall_prune:
        predicted_plan = prune_near_duplicate_walls(
            predicted_plan,
            distance_tolerance_fraction=(
                near_duplicate_wall_distance_fraction
            ),
            angle_tolerance_degrees=near_duplicate_wall_angle_degrees,
            minimum_overlap_fraction=near_duplicate_wall_overlap,
        )
    target_plan = predicted_control_explicit_plan(
        target,
        footprint,
        taxonomy,
        reconcile_gap_fraction=None,
    )
    if (
        prediction_opening_center is not None
        and prediction_opening_interval_width is not None
        and prediction_opening_family_logits is not None
    ):
        attach_center_width_opening_detections(
            predicted_plan,
            prediction_opening_center,
            prediction_opening_interval_width,
            prediction_opening_family_logits,
            taxonomy,
            threshold=prediction_opening_heatmap_threshold,
            min_width=max(3, prediction_opening_min_area),
            min_peak_distance=prediction_opening_min_peak_distance,
            candidate_ranker=prediction_opening_candidate_ranker,
            host_plan=prediction_opening_host_plan,
            decode_mode=prediction_opening_center_decode_mode,
            max_wall_distance=prediction_opening_max_wall_distance,
            precomputed_detections=(
                prediction_opening_precomputed_detections
            ),
            remap_wall=prediction_opening_remap_final_wall,
        )
    elif (
        prediction_opening_foreground is not None
        and prediction_opening_family_logits is not None
    ):
        attach_learned_opening_detections_from_maps(
            predicted_plan,
            prediction_opening_foreground,
            prediction_opening_family_logits,
            taxonomy,
            threshold=prediction_opening_heatmap_threshold,
            min_area=prediction_opening_min_area,
            host_plan=prediction_opening_host_plan,
            family_aggregation=prediction_opening_family_aggregation,
            exterior_wall_window_bias=(
                prediction_opening_exterior_window_bias
            ),
            interval_radius=prediction_opening_interval_radius,
            score_smoothing=prediction_opening_score_smoothing,
            max_gap_samples=prediction_opening_max_gap_samples,
            peak_relative_threshold=(
                prediction_opening_peak_relative_threshold
            ),
            candidate_ranker=prediction_opening_candidate_ranker,
        )
    elif prediction_opening_heatmap is not None:
        attach_learned_opening_detections(
            predicted_plan,
            prediction_opening_heatmap,
            taxonomy,
            threshold=prediction_opening_heatmap_threshold,
            min_area=prediction_opening_min_area,
            max_wall_distance=prediction_opening_max_wall_distance,
            host_plan=prediction_opening_host_plan,
        )
    elif (
        prediction_opening_probability is not None
        and prediction_opening_logits is not None
    ):
        attach_wall_interval_openings(
            predicted_plan,
            prediction_opening_probability,
            prediction_opening_logits,
            taxonomy,
            threshold=prediction_opening_threshold,
        )
    if ip_wall_reconcile:
        predicted_plan = ip_reconcile_plan(
            predicted_plan,
            prediction,
            taxonomy,
            tolerance_fraction=float(ip_wall_tolerance_fraction),
            merge_gap_fraction=float(ip_wall_merge_gap_fraction),
            use_milp=bool(ip_wall_use_milp),
        )
    room = room_metric_suite(
        prediction,
        target,
        footprint,
        taxonomy,
        predicted_room_support=np.isin(prediction, taxonomy.room_ids),
        predicted_instance_support=np.isin(prediction, taxonomy.room_ids),
        target_room_support=np.isin(target, taxonomy.room_ids),
    )
    instance = room["matched_instance"]
    fpeval = evaluate(
        to_fpeval_plan(predicted_plan),
        to_fpeval_plan(target_plan),
        tol_frac=float(fpeval_tolerance),
    )
    corner = room_corner_angle_metrics(
        prediction,
        target,
        taxonomy,
        footprint=footprint,
        explicit_rooms=True,
    )
    if graph_source == "plan":
        graph = housemind_style_metrics_from_plan(
            predicted_plan,
            target,
            footprint,
            taxonomy,
            ged_timeout_seconds=float(graph_ged_timeout_seconds),
        )
    elif graph_source == "raster":
        graph = housemind_style_metrics(
            prediction,
            target,
            footprint,
            taxonomy,
            ged_timeout_seconds=float(graph_ged_timeout_seconds),
        )
    else:
        raise ValueError("graph_source must be plan or raster")
    plausibility = generation_plausibility(
        prediction,
        footprint,
        taxonomy,
        explicit_rooms=True,
        reconcile_gap_fraction=reconcile_gap_fraction,
    )
    metrics = {
        "room_macro_iou": float(room["visible_semantic"]["room_macro_iou"]),
        "prediction_room_source": predicted_plan.room_source,
        "graph_source": graph_source,
        "room_support_iou": float(
            room["occlusion_aware_support"]["room_support_iou"]
        ),
        **{
            f"instance_{key}": float(value)
            for key, value in instance.items()
        },
        "fpeval_wall_f1": float(fpeval.walls["f1"]),
        "fpeval_wall_precision": float(fpeval.walls["precision"]),
        "fpeval_wall_recall": float(fpeval.walls["recall"]),
        "fpeval_opening_f1": float(fpeval.openings["f1"]),
        "fpeval_room_f1": float(fpeval.rooms["f1"]),
        "fpeval_room_mean_iou": float(fpeval.rooms["mean_iou"] or 0.0),
        "fpeval_room_label_accuracy": float(
            fpeval.rooms["label_acc"] or 0.0
        ),
        "fpeval_edit_cost": float(fpeval.edit["total"]),
        "fpeval_edit_cost_per_gt_wall": float(fpeval.edit["per_gt_wall"]),
        **{key: float(value) if isinstance(value, (int, float)) else value
           for key, value in corner.items()},
        **{
            key: (
                float(value)
                if isinstance(value, (int, float, np.integer, np.floating))
                else value
            )
            for key, value in graph.items()
        },
        **plausibility,
    }
    if include_room_decomposition:
        import torch

        metrics["room_error_decomposition"] = decompose_rooms(
            torch.from_numpy(prediction)[None, None],
            torch.from_numpy(target)[None, None],
            torch.from_numpy(footprint)[None, None],
            taxonomy,
        )
    return metrics


__all__ = [
    "predicted_control_explicit_plan",
    "predicted_control_fidelity_metrics",
]
