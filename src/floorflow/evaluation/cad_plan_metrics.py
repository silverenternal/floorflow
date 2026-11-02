"""Canonical metrics for composed CAD wall plans."""

from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fpeval_bridge import evaluate, to_fpeval_plan, verify
from floorflow.evaluation.structural_legality import wall_junction_summary
from floorflow.evaluation.wall_junction import wall_junction_metrics
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    raster_labels_to_wall_plan,
)


def cad_wall_plan_metrics(
    plan: RasterWallPlan,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    """Evaluate one vector plan against one canonical raster target."""

    target_wall_plan = raster_labels_to_wall_plan(target, taxonomy)
    target_plan = to_fpeval_plan(target_wall_plan)
    report = evaluate(to_fpeval_plan(plan), target_plan)
    plan_verify = verify(to_fpeval_plan(plan))
    target_verify = verify(target_plan)
    junction = wall_junction_metrics(
        plan.walls,
        target_wall_plan.walls,
        cluster_tolerance=2.0,
        match_distance=6.0,
    )
    continuity = wall_junction_summary(plan.walls)
    eligible = float(target_verify.watertight)
    return {
        "wall": float(report.walls["f1"]),
        "wall_precision": float(report.walls["precision"]),
        "wall_recall": float(report.walls["recall"]),
        "junction_f1": float(junction["junction_f1"]),
        "wall_continuity": float(continuity["wall_continuity_rate"]),
        "opening": float(report.openings["f1"]),
        "room": float(report.rooms["f1"]),
        "valid": float(plan_verify.valid),
        "watertight": float(plan_verify.watertight),
        "eligible": eligible,
        "eligible_wall": float(report.walls["f1"]) if eligible else 0.0,
        "eligible_opening": (float(report.openings["f1"]) if eligible else 0.0),
        "eligible_room": float(report.rooms["f1"]) if eligible else 0.0,
        "eligible_valid": float(plan_verify.valid) if eligible else 0.0,
        "eligible_watertight": (float(plan_verify.watertight) if eligible else 0.0),
    }


__all__ = ["cad_wall_plan_metrics"]
