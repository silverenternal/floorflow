"""Target-free legalizer portfolio helpers for external ORFS placements."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from trinity.floorplan.geometry import overlapping_pairs
from trinity.floorplan.legalize.scale_pack import last_info
from trinity.floorplan.scoring.feasibility import check_feasibility
from trinity.floorplan.scoring.validate import validate
from trinity.floorplan.types import Placement

from floorflow.chip_flow.legalization import (
    scale_pack_legalize_reproducible,
    translate_into_core,
)
from floorflow.chip_flow.legalizer_policy import LegalizerPolicy
from floorflow.chip_flow.ppa_external import (
    deflate_boxes,
    inflate_boxes,
    restore_fixed_dimensions,
    shelf_pack_fixed,
)


@dataclass(frozen=True)
class ExternalLegalizerCandidate:
    name: str
    boxes: np.ndarray
    route: str
    v_rel: float
    boundary: float
    grouping: float
    displacement_mean_um: float
    rung: int
    solves: int
    geometry: dict[str, bool | float | int]
    analytical_wirelength: float | None = None
    analytical_congestion: float | None = None

    @property
    def eligible(self) -> bool:
        required_geometry = (
            "checker_feasible",
            "inside_core",
            "overlap_pair_count",
            "fixed_shape_max_relative_error",
        )
        if any(key not in self.geometry for key in required_geometry):
            return False
        numeric = (
            self.v_rel,
            self.boundary,
            self.grouping,
            self.displacement_mean_um,
        )
        return (
            all(math.isfinite(float(value)) and float(value) >= 0.0 for value in numeric)
            and int(self.rung) >= 0
            and int(self.solves) >= 0
            and bool(self.geometry["checker_feasible"])
            and bool(self.geometry["inside_core"])
            and int(self.geometry["overlap_pair_count"]) == 0
            and float(self.geometry["fixed_shape_max_relative_error"]) < 1.0e-6
        )


def refinement_input(z: torch.Tensor, graph, instance) -> torch.Tensor:
    core_width, core_height = graph.core_size
    physical_scale = math.sqrt(float(instance.area_targets.sum()))
    core_scale = math.sqrt(core_width * core_height)
    centers = z[0, :, :2] * core_scale
    centers[..., 0] += graph.core[0]
    centers[..., 1] += graph.core[1]
    widths = torch.as_tensor(
        [macro["width_dbu"] for macro in graph.macros],
        device=z.device,
        dtype=z.dtype,
    ) / graph.dbu_per_micron
    heights = torch.as_tensor(
        [macro["height_dbu"] for macro in graph.macros],
        device=z.device,
        dtype=z.dtype,
    ) / graph.dbu_per_micron
    rho = torch.log(widths / heights).clamp(-4.0, 4.0)
    return torch.stack(
        [
            (centers[:, 0] - graph.core[0]) / physical_scale,
            (centers[:, 1] - graph.core[1]) / physical_scale,
            rho,
        ],
        dim=1,
    )[None]


def decode_refined(z: torch.Tensor, graph, instance) -> np.ndarray:
    scale = math.sqrt(float(instance.area_targets.sum()))
    widths = torch.as_tensor(
        [macro["width_dbu"] for macro in graph.macros],
        device=z.device,
        dtype=z.dtype,
    ) / graph.dbu_per_micron
    heights = torch.as_tensor(
        [macro["height_dbu"] for macro in graph.macros],
        device=z.device,
        dtype=z.dtype,
    ) / graph.dbu_per_micron
    centers = z[0, :, :2] * scale + z.new_tensor([graph.core[0], graph.core[1]])
    return torch.stack(
        [
            centers[:, 0] - widths / 2,
            centers[:, 1] - heights / 2,
            widths,
            heights,
        ],
        dim=1,
    ).cpu().numpy()


def geometry_audit(boxes: np.ndarray, graph, instance) -> dict:
    x0, y0, x1, y1 = graph.core
    inside = bool(
        np.all(boxes[:, 0] >= x0 - 1.0e-4)
        and np.all(boxes[:, 1] >= y0 - 1.0e-4)
        and np.all(boxes[:, 0] + boxes[:, 2] <= x1 + 1.0e-4)
        and np.all(boxes[:, 1] + boxes[:, 3] <= y1 + 1.0e-4)
    )
    expected = np.asarray(
        [
            [
                macro["width_dbu"] / graph.dbu_per_micron,
                macro["height_dbu"] / graph.dbu_per_micron,
            ]
            for macro in graph.macros
        ]
    )
    shape_error = float(
        np.max(np.abs(boxes[:, 2:] / expected - 1.0))
        if boxes.size
        else 0.0
    )
    return {
        "checker_feasible": bool(check_feasibility(Placement(boxes, instance)).is_feasible()),
        "inside_core": inside,
        "overlap_pair_count": len(overlapping_pairs(boxes)),
        "fixed_shape_max_relative_error": shape_error,
    }


def displacement_summary(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    delta = (after[:, :2] + after[:, 2:] / 2) - (before[:, :2] + before[:, 2:] / 2)
    distance = np.linalg.norm(delta, axis=1)
    return {
        "mean_um": float(distance.mean()),
        "p50_um": float(np.quantile(distance, 0.50)),
        "p90_um": float(np.quantile(distance, 0.90)),
        "p95_um": float(np.quantile(distance, 0.95)),
        "max_um": float(distance.max()),
    }


def legalized_candidate(
    name: str,
    refined_boxes: np.ndarray,
    graph,
    instance,
    legalization_instance,
    *,
    relation: str,
    halo_um: float,
    force_shelf: bool,
    position_aware_shelf: bool,
    time_limit_ms: int,
    legalizer_policy: LegalizerPolicy | None = None,
) -> ExternalLegalizerCandidate:
    inflated = inflate_boxes(refined_boxes, halo_um)
    policy = legalizer_policy or LegalizerPolicy(
        relation=relation,
        time_limit_ms=time_limit_ms,
    )
    scaled = scale_pack_legalize_reproducible(
        Placement(inflated, legalization_instance),
        time_limit_ms=policy.time_limit_ms,
        **policy.kwargs(),
    )
    info = dict(last_info())
    scale_boxes = restore_fixed_dimensions(
        deflate_boxes(np.asarray(scaled.xywh, dtype=np.float64), halo_um),
        graph,
    )
    scale_audit = geometry_audit(scale_boxes, graph, instance)
    translated = None
    translated_audit = None
    if (
        scale_audit["checker_feasible"]
        and scale_audit["overlap_pair_count"] == 0
        and scale_audit["fixed_shape_max_relative_error"] < 1.0e-6
    ):
        translated = translate_into_core(scale_boxes, graph.core)
        if translated is not None:
            translated_audit = geometry_audit(translated, graph, instance)
    if force_shelf:
        packed = shelf_pack_fixed(
            inflated,
            graph.core_origin,
            graph.core_size,
            position_aware_ties=position_aware_shelf,
        )
        boxes = (
            scale_boxes
            if packed is None
            else restore_fixed_dimensions(deflate_boxes(packed, halo_um), graph)
        )
        route = "shelf_pack_forced" if packed is not None else "shelf_pack_failed"
    elif (
        translated is not None
        and translated_audit is not None
        and translated_audit["checker_feasible"]
        and translated_audit["inside_core"]
        and translated_audit["overlap_pair_count"] == 0
    ):
        boxes = translated
        route = "scale_pack_core_fit"
    else:
        packed = shelf_pack_fixed(
            inflated,
            graph.core_origin,
            graph.core_size,
            position_aware_ties=position_aware_shelf,
        )
        boxes = (
            scale_boxes
            if packed is None
            else restore_fixed_dimensions(deflate_boxes(packed, halo_um), graph)
        )
        route = "shelf_pack_fallback" if packed is not None else "shelf_pack_failed"
    score = validate(Placement(boxes, instance), "full_fast").score
    soft = score.soft
    displacement = displacement_summary(refined_boxes, boxes)
    return ExternalLegalizerCandidate(
        name=name,
        boxes=np.asarray(boxes, dtype=np.float64),
        route=route,
        v_rel=float(soft.v_rel if soft is not None else 0.0),
        boundary=float(soft.boundary if soft is not None else 0.0),
        grouping=float(soft.grouping if soft is not None else 0.0),
        displacement_mean_um=float(displacement["mean_um"]),
        rung=int(info["rung"]),
        solves=int(info["solves"]),
        geometry=geometry_audit(boxes, graph, instance),
    )


def select_candidate(
    baseline: ExternalLegalizerCandidate,
    candidates: tuple[ExternalLegalizerCandidate, ...],
) -> ExternalLegalizerCandidate:
    eligible = [
        candidate
        for candidate in (baseline, *candidates)
        if candidate.eligible
        and candidate.rung <= baseline.rung
        and candidate.solves <= baseline.solves
    ]
    if not eligible:
        raise RuntimeError("external portfolio has no legal non-escalating candidate")
    return min(
        eligible,
        key=lambda candidate: (
            candidate.v_rel,
            candidate.displacement_mean_um,
            candidate.boundary,
            candidate.grouping,
            candidate.name,
        ),
    )


def select_candidate_with_analytical_response(
    baseline: ExternalLegalizerCandidate,
    candidates: tuple[ExternalLegalizerCandidate, ...],
    *,
    analytical_scores: dict[str, float] | None = None,
) -> ExternalLegalizerCandidate:
    """Select by routed-WL response only after all legality guards pass."""

    names = [candidate.name for candidate in (baseline, *candidates)]
    if len(names) != len(set(names)):
        raise RuntimeError("external portfolio candidate names must be unique")
    eligible = [
        candidate
        for candidate in (baseline, *candidates)
        if candidate.eligible
        and candidate.rung <= baseline.rung
        and candidate.solves <= baseline.solves
    ]
    if not eligible:
        raise RuntimeError("external portfolio has no legal non-escalating candidate")
    scores = {}
    for candidate in eligible:
        value = (
            analytical_scores.get(candidate.name)
            if analytical_scores is not None
            else candidate.analytical_wirelength
        )
        if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
            raise RuntimeError(
                f"analytical response is missing for eligible candidate {candidate.name}"
            )
        scores[candidate.name] = float(value)
        congestion = candidate.analytical_congestion
        if congestion is not None and (
            not math.isfinite(float(congestion)) or float(congestion) < 0.0
        ):
            raise RuntimeError(
                f"analytical congestion is invalid for eligible candidate {candidate.name}"
            )
    return min(
        eligible,
        key=lambda candidate: (
            scores[candidate.name],
            candidate.analytical_congestion
            if candidate.analytical_congestion is not None
            else float("inf"),
            candidate.displacement_mean_um,
            candidate.v_rel,
            candidate.name,
        ),
    )


__all__ = [
    "ExternalLegalizerCandidate",
    "decode_refined",
    "displacement_summary",
    "geometry_audit",
    "legalized_candidate",
    "refinement_input",
    "select_candidate",
    "select_candidate_with_analytical_response",
]
