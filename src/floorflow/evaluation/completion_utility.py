"""Completion-specific edit and topology metrics."""

from __future__ import annotations

import numpy as np

from floorflow.evaluation.control_utility import (
    adjacency_f1,
    intent_preservation,
    room_component_topology,
)
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics


def completion_utility_metrics(
    prediction: np.ndarray,
    control: np.ndarray,
    target: np.ndarray,
    taxonomy,
    *,
    output_edit_cost: float,
    output_edit_cost_per_gt_wall: float,
) -> dict[str, float | int]:
    control_fpeval = fpeval_raster_metrics(
        control,
        target,
        taxonomy,
        tolerance_fractions=(0.05,),
    )
    intent_fpeval = fpeval_raster_metrics(
        prediction,
        control,
        taxonomy,
        tolerance_fractions=(0.05,),
    )
    topology = room_component_topology(prediction, taxonomy)
    topology.pop("room_adjacency")
    control_edit_cost = control_fpeval["summary"]["0.05"]["edit_cost"]
    return {
        **{
            key: value
            for key, value in topology.items()
            if isinstance(value, (int, float))
        },
        **adjacency_f1(prediction, target, taxonomy),
        **intent_preservation(prediction, control, taxonomy),
        "fpeval_edit_cost": output_edit_cost,
        "fpeval_edit_cost_per_gt_wall": output_edit_cost_per_gt_wall,
        "control_edit_cost_to_target": control_edit_cost,
        "edit_cost_reduction": control_edit_cost - output_edit_cost,
        "intent_edit_cost": intent_fpeval["summary"]["0.05"]["edit_cost"],
    }


__all__ = ["completion_utility_metrics"]
