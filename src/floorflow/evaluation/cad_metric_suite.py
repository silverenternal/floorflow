"""Unified CAD reverse-design metrics without collapsing metric scopes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MetricSpec:
    name: str
    family: str
    direction: str
    provenance: str
    description: str

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("metric direction must be maximize or minimize")


METRIC_SPECS = (
    MetricSpec("room_macro_iou", "pixel_semantics", "maximize", "room_metric_suite", "case-mean present room IoU"),
    MetricSpec("room_dataset_macro_iou", "pixel_semantics", "maximize", "room_iou_diagnostics", "class-balanced dataset room IoU"),
    MetricSpec("room_dataset_micro_iou", "pixel_semantics", "maximize", "room_iou_diagnostics", "global room-pixel IoU"),
    MetricSpec("boundary_f1", "pixel_semantics", "maximize", "cad_metric diagnostics", "room boundary F1"),
    MetricSpec("room_support_iou", "pixel_semantics", "maximize", "room_metric_suite", "occlusion-aware support IoU"),
    MetricSpec("room_support_micro_iou", "pixel_semantics", "maximize", "overlay diagnostics", "global room support IoU"),
    MetricSpec("fpeval_wall_f1", "vector_geometry", "maximize", "fpvec-lab/fpeval", "wall F1"),
    MetricSpec("fpeval_opening_f1", "vector_geometry", "maximize", "fpvec-lab/fpeval", "opening F1"),
    MetricSpec("fpeval_room_polygon_f1", "vector_geometry", "maximize", "fpvec-lab/fpeval", "room polygon F1"),
    MetricSpec("fpeval_room_face_f1", "vector_geometry", "maximize", "fpvec-lab/fpeval", "room face F1"),
    MetricSpec("fpeval_edit_cost", "vector_geometry", "minimize", "fpvec-lab/fpeval", "correction edit cost"),
    MetricSpec("fpeval_validity", "topology", "maximize", "fpvec-lab/fpeval", "schema validity"),
    MetricSpec("fpeval_watertight", "topology", "maximize", "fpvec-lab/fpeval", "watertight room faces"),
    MetricSpec("conditional_watertight", "topology", "maximize", "fpvec-lab/fpeval", "watertight fraction on GT-watertight cases"),
    MetricSpec("room_component_gap", "topology", "minimize", "room_error_decomposition", "fragmentation component gap"),
    MetricSpec("outside_footprint_pixels", "spatial_constraints", "minimize", "overlay diagnostics", "pixels outside footprint"),
    MetricSpec("room_coverage", "spatial_constraints", "maximize", "overlay diagnostics", "in-footprint room coverage"),
    MetricSpec("room_target_recall", "spatial_constraints", "maximize", "overlay diagnostics", "predicted room recall over ground-truth room pixels"),
    MetricSpec("opening_wall_alignment", "spatial_constraints", "maximize", "overlay diagnostics", "opening-to-wall alignment"),
    MetricSpec("furniture_room_containment", "spatial_constraints", "maximize", "structural_metrics", "furniture inside rooms"),
    MetricSpec("instance_geometric_iou", "instances", "maximize", "room_metric_suite", "Hungarian matched geometry IoU"),
    MetricSpec("instance_room_f1", "instances", "maximize", "room_metric_suite", "instance F1 at IoU threshold"),
    MetricSpec("instance_type_accuracy", "instances", "maximize", "room_metric_suite", "matched class accuracy"),
    MetricSpec("room_group_macro_iou", "instances", "maximize", "room_groups", "functional room-group macro IoU"),
    MetricSpec("wall_iou", "objects", "maximize", "floorplan_metrics", "wall raster IoU"),
    MetricSpec("opening_iou", "objects", "maximize", "floorplan_metrics", "opening raster IoU"),
    MetricSpec("furniture_iou", "objects", "maximize", "floorplan_metrics", "furniture raster IoU"),
    MetricSpec("style_separation", "generation", "maximize", "style semantic diagnostics", "between-style separation"),
    MetricSpec("seed_iou_gap", "generation", "minimize", "seed diagnostics", "seed-induced IoU variation"),
)

SPEC_BY_NAME = {spec.name: spec for spec in METRIC_SPECS}
FAMILIES = tuple(dict.fromkeys(spec.family for spec in METRIC_SPECS))

DEFAULT_HARD_FLOORS = {
    "fpeval_validity": 1.0,
    "conditional_watertight": 0.95,
    "room_target_recall": 0.95,
    "outside_footprint_pixels": 0.0,
}

BALANCED_WEIGHTS = {
    "room_macro_iou": 0.30,
    "room_support_iou": 0.10,
    "fpeval_room_polygon_f1": 0.15,
    "fpeval_wall_f1": 0.10,
    "fpeval_opening_f1": 0.10,
    "instance_geometric_iou": 0.10,
    "instance_type_accuracy": 0.05,
    "furniture_iou": 0.05,
    "fpeval_validity": 0.05,
}


def summarize_metric_rows(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, float]:
    """Average only metrics with an explicit registry spec."""

    rows = list(rows)
    summary = {}
    for name in SPEC_BY_NAME:
        values = [
            float(row[name])
            for row in rows
            if isinstance(row.get(name), (int, float))
            and math.isfinite(float(row[name]))
        ]
        if values:
            summary[name] = float(np.mean(values))
    return summary


def evaluate_hard_floors(
    summary: Mapping[str, float],
    floors: Mapping[str, float] | None = None,
) -> dict[str, dict[str, object]]:
    """Evaluate legal and spatial requirements before quality ranking."""

    floors = dict(DEFAULT_HARD_FLOORS if floors is None else floors)
    result = {}
    for name, threshold in floors.items():
        spec = SPEC_BY_NAME.get(name)
        if spec is None:
            raise ValueError(f"hard floor metric has no registry spec: {name}")
        value = summary.get(name)
        passed = (
            value is not None
            and (
                value >= float(threshold)
                if spec.direction == "maximize"
                else value <= float(threshold)
            )
        )
        result[name] = {
            "value": value,
            "threshold": float(threshold),
            "direction": spec.direction,
            "passed": bool(passed),
        }
    return result


def balanced_cad_score(summary: Mapping[str, float]) -> dict[str, object]:
    """Compute a documented secondary score without hiding raw metrics."""

    available = {}
    for name, weight in BALANCED_WEIGHTS.items():
        if name not in summary:
            continue
        value = max(0.0, min(1.0, float(summary[name])))
        available[name] = weight * value
    total_weight = sum(BALANCED_WEIGHTS[name] for name in available)
    coverage_gap = max(
        0.0,
        0.95 - float(summary.get("room_target_recall", 0.0)),
    )
    score = (
        sum(available.values()) / total_weight if total_weight else 0.0
    ) - 0.25 * coverage_gap
    return {
        "name": "balanced_cad_secondary",
        "score": float(score),
        "coverage_gap_penalty": float(0.25 * coverage_gap),
        "available_weight": float(total_weight),
        "weights": dict(BALANCED_WEIGHTS),
    }


def metric_agreement(
    rows: Iterable[Mapping[str, object]],
    pairs: Iterable[tuple[str, str]] = (
        ("room_macro_iou", "room_support_iou"),
        ("room_macro_iou", "fpeval_room_polygon_f1"),
        ("room_macro_iou", "instance_geometric_iou"),
        ("room_macro_iou", "fpeval_wall_f1"),
        ("room_macro_iou", "fpeval_opening_f1"),
    ),
) -> dict[str, dict[str, float | int | None]]:
    """Measure correlation and centered sign disagreements between metrics."""

    rows = list(rows)
    result = {}
    for left_name, right_name in pairs:
        paired = [
            (float(row[left_name]), float(row[right_name]))
            for row in rows
            if isinstance(row.get(left_name), (int, float))
            and isinstance(row.get(right_name), (int, float))
        ]
        if len(paired) < 2:
            result[f"{left_name}__{right_name}"] = {
                "cases": len(paired),
                "pearson": None,
                "sign_agreement": None,
                "sign_inversions": None,
            }
            continue
        left, right = np.asarray(paired).T
        if float(left.std()) == 0.0 or float(right.std()) == 0.0:
            correlation = None
        else:
            correlation = float(np.corrcoef(left, right)[0, 1])
        centered = np.sign(
            np.stack([left - left.mean(), right - right.mean()], axis=1)
        ).prod(axis=1)
        result[f"{left_name}__{right_name}"] = {
            "cases": len(paired),
            "pearson": correlation,
            "sign_agreement": float((centered > 0).mean()),
            "sign_inversions": int((centered < 0).sum()),
        }
    return result


def pareto_front(
    candidates: Iterable[Mapping[str, object]],
    objectives: Iterable[str] = (
        "room_macro_iou",
        "fpeval_room_polygon_f1",
        "instance_geometric_iou",
    ),
) -> list[dict[str, object]]:
    """Return non-dominated candidates with all hard-gated failures removed."""

    candidates = list(candidates)
    objectives = tuple(objectives)
    for name in objectives:
        if name not in SPEC_BY_NAME:
            raise ValueError(f"unknown Pareto objective: {name}")
    qualified = [
        candidate
        for candidate in candidates
        if bool(candidate.get("hard_gate_passed", False))
    ]
    valid = [
        candidate
        for candidate in qualified
        if all(
            isinstance(candidate.get("summary", {}).get(name), (int, float))
            for name in objectives
        )
    ]
    front = []
    for candidate in valid:
        summary = candidate["summary"]
        dominated = False
        for opponent in valid:
            if opponent is candidate:
                continue
            other = opponent["summary"]
            weakly_better = True
            strictly_better = False
            for name in objectives:
                direction = SPEC_BY_NAME[name].direction
                left = float(summary[name])
                right = float(other[name])
                if direction == "maximize":
                    weakly_better &= right >= left
                    strictly_better |= right > left
                else:
                    weakly_better &= right <= left
                    strictly_better |= right < left
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def build_cad_metric_report(
    rows: Iterable[Mapping[str, object]],
    *,
    hard_floors: Mapping[str, float] | None = None,
) -> dict[str, object]:
    rows = list(rows)
    summary = summarize_metric_rows(rows)
    floors = evaluate_hard_floors(summary, hard_floors)
    missing = [
        spec.name for spec in METRIC_SPECS if spec.name not in summary
    ]
    families = {
        family: {
            spec.name: summary.get(spec.name)
            for spec in METRIC_SPECS
            if spec.family == family
        }
        for family in FAMILIES
    }
    return {
        "protocol": {
            "cases": len(rows),
            "micro_macro_instance_separated": True,
            "raw_metrics_preserved": True,
        },
        "metric_specs": {
            name: asdict(spec) for name, spec in SPEC_BY_NAME.items()
        },
        "summary": summary,
        "families": families,
        "hard_gate": {
            "passed": all(item["passed"] for item in floors.values()),
            "requirements": floors,
        },
        "balanced_secondary": balanced_cad_score(summary),
        "agreement": metric_agreement(rows),
        "missing_metrics": missing,
    }


__all__ = [
    "BALANCED_WEIGHTS",
    "DEFAULT_HARD_FLOORS",
    "FAMILIES",
    "METRIC_SPECS",
    "SPEC_BY_NAME",
    "balanced_cad_score",
    "build_cad_metric_report",
    "evaluate_hard_floors",
    "metric_agreement",
    "pareto_front",
    "summarize_metric_rows",
]
