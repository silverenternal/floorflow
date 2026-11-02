"""Composed-plan metrics for scored CAD wall candidates."""

from __future__ import annotations

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fpeval_bridge import evaluate, to_fpeval_plan, verify
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.inference.cad_wall_candidate_runtime import (
    compose_scored_candidate_plan,
)


def candidate_plan_metrics(
    endpoints: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    rooms: list[list[dict]],
    host_walls: list[list],
    opening_probability: torch.Tensor,
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    threshold: float,
    wall_budget: int | torch.Tensor | None = None,
    exterior: torch.Tensor | None = None,
    exterior_budget: int | torch.Tensor | None = None,
    interior_budget: int | torch.Tensor | None = None,
    wall_type_logits: torch.Tensor | None = None,
    routed_openings: list | None = None,
    repair_keep_logits: torch.Tensor | None = None,
    repair_endpoint_delta: torch.Tensor | None = None,
    repair_max_edits: int = 8,
    repair_keep_threshold: float = 0.5,
) -> dict[str, float]:
    rows = []
    for index in range(target.shape[0]):
        budget = (
            int(wall_budget[index].item())
            if torch.is_tensor(wall_budget)
            else wall_budget
        )
        exterior_count = (
            int(exterior_budget[index].item())
            if torch.is_tensor(exterior_budget)
            else exterior_budget
        )
        interior_count = (
            int(interior_budget[index].item())
            if torch.is_tensor(interior_budget)
            else interior_budget
        )
        plan = compose_scored_candidate_plan(
            endpoints[index],
            scores[index],
            valid[index],
            rooms[index],
            float(target.shape[-1]),
            host_walls[index],
            opening_probability[index].cpu().numpy(),
            taxonomy,
            score_threshold=threshold,
            wall_budget=budget,
            exterior=None if exterior is None else exterior[index],
            exterior_budget=exterior_count,
            interior_budget=interior_count,
            wall_type_logits=(
                None if wall_type_logits is None else wall_type_logits[index]
            ),
            routed_openings=(
                None if routed_openings is None else routed_openings[index]
            ),
            repair_keep_logits=(
                None
                if repair_keep_logits is None
                else repair_keep_logits[index]
            ),
            repair_endpoint_delta=(
                None
                if repair_endpoint_delta is None
                else repair_endpoint_delta[index]
            ),
            repair_max_edits=int(repair_max_edits),
            repair_keep_threshold=float(repair_keep_threshold),
        )
        target_plan = to_fpeval_plan(
            raster_labels_to_wall_plan(target[index].cpu().numpy(), taxonomy)
        )
        report = evaluate(to_fpeval_plan(plan), target_plan)
        plan_verify = verify(to_fpeval_plan(plan))
        target_verify = verify(target_plan)
        rows.append(
            {
                "wall": float(report.walls["f1"]),
                "opening": float(report.openings["f1"]),
                "room": float(report.rooms["f1"]),
                "valid": float(plan_verify.valid),
                "watertight": float(plan_verify.watertight),
                "eligible": float(target_verify.watertight),
            }
        )
    eligible = [row for row in rows if row["eligible"] > 0.5]
    eligible_rows = eligible or [
        {key: 0.0 for key in rows[0]}
    ]
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    } | {
        f"eligible_{key}": float(np.mean([row[key] for row in eligible_rows]))
        for key in rows[0]
        if key != "eligible"
    }


__all__ = ["candidate_plan_metrics"]
