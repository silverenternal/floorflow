"""Batch candidate verification and attachment for opening plans."""

from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.opening_readout import OpeningDetection
from floorflow.geometry.wall_graph_readout import RasterWallPlan
from floorflow.inference.opening_candidate_ranker import (
    FEATURE_NAMES_V3,
    OpeningCandidateRanker,
    opening_candidate_features,
    opening_candidate_features_v3,
)
from floorflow.inference.opening_center_width_decoder import (
    read_center_width_opening_detections,
    set_opening_detections,
)


def _wall_identity_matches(
    left: RasterWallPlan,
    right: RasterWallPlan,
    tolerance: float = 0.5,
) -> bool:
    if len(left.walls) != len(right.walls):
        return False
    for left_wall, right_wall in zip(left.walls, right.walls, strict=True):
        left_start = np.asarray(left_wall.start, dtype=np.float64)
        left_end = np.asarray(left_wall.end, dtype=np.float64)
        right_start = np.asarray(right_wall.start, dtype=np.float64)
        right_end = np.asarray(right_wall.end, dtype=np.float64)
        direct = max(
            float(np.linalg.norm(left_start - right_start)),
            float(np.linalg.norm(left_end - right_end)),
        )
        reversed_order = max(
            float(np.linalg.norm(left_start - right_end)),
            float(np.linalg.norm(left_end - right_start)),
        )
        if min(direct, reversed_order) > float(tolerance):
            return False
    return True


def rank_and_attach_center_width_openings(
    plans: list[RasterWallPlan],
    host_plans: list[RasterWallPlan],
    center_maps: list[np.ndarray],
    foreground_maps: list[np.ndarray],
    width_maps: list[np.ndarray],
    family_maps: list[np.ndarray],
    taxonomy: FloorplanTaxonomy,
    ranker: OpeningCandidateRanker,
    *,
    threshold: float,
    min_area: int,
    min_peak_distance: float,
    frame: float,
    allow_host_identity_change: bool = False,
) -> None:
    """Verify all case candidates in one classifier call, then attach."""

    if not allow_host_identity_change and any(
        not _wall_identity_matches(plan, host_plan)
        for plan, host_plan in zip(plans, host_plans, strict=True)
    ):
        raise ValueError(
            "batched opening attachment requires matching host wall identity"
        )
    ranked_groups = rank_center_width_openings(
        host_plans,
        center_maps,
        foreground_maps,
        width_maps,
        family_maps,
        taxonomy,
        ranker,
        threshold=threshold,
        min_area=min_area,
        min_peak_distance=min_peak_distance,
        frame=frame,
    )
    for plan, detections in zip(plans, ranked_groups, strict=True):
        set_opening_detections(plan, detections)


def rank_center_width_openings(
    host_plans: list[RasterWallPlan],
    center_maps: list[np.ndarray],
    foreground_maps: list[np.ndarray],
    width_maps: list[np.ndarray],
    family_maps: list[np.ndarray],
    taxonomy: FloorplanTaxonomy,
    ranker: OpeningCandidateRanker,
    *,
    threshold: float,
    min_area: int,
    min_peak_distance: float,
    frame: float,
) -> list[list[OpeningDetection]]:
    """Verify all case candidates in one classifier call."""

    detection_groups = [
        read_center_width_opening_detections(
            host_plan,
            center,
            width,
            family,
            taxonomy,
            threshold=threshold,
            min_width=float(max(3, min_area)),
            min_peak_distance=min_peak_distance,
        )
        for host_plan, center, width, family in zip(
            host_plans,
            center_maps,
            width_maps,
            family_maps,
            strict=True,
        )
    ]
    feature_function = (
        opening_candidate_features_v3
        if tuple(ranker.feature_names) == FEATURE_NAMES_V3
        else opening_candidate_features
    )
    feature_groups = [
        (
            np.stack(
                [
                    feature_function(
                        detection,
                        foreground,
                        family,
                        host_plan,
                        detections,
                        frame=frame,
                    )
                    for detection in detections
                ]
            )
            if detections
            else np.zeros((0, len(ranker.feature_names)), dtype=np.float32)
        )
        for host_plan, foreground, family, detections in zip(
            host_plans,
            foreground_maps,
            family_maps,
            detection_groups,
            strict=True,
        )
    ]
    return ranker.rank_feature_groups(
        detection_groups,
        feature_groups,
        taxonomy,
    )


__all__ = [
    "rank_and_attach_center_width_openings",
    "rank_center_width_openings",
]
