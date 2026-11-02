"""Paired fidelity, tail, and distribution summaries for graph refinement."""

from __future__ import annotations

import math

import numpy as np

from floorflow.evaluation.sample_diversity import (
    coverage_density,
    mean_pairwise_distance,
    mmd_rbf,
)

METRICS = ("node_f1", "edge_overlap", "ged")


def _mean(rows: list[dict], method: str, metric: str) -> float:
    return float(
        np.mean([row["metrics"][method][metric] for row in rows])
    )


def compare_rows(
    rows: list[dict],
    *,
    baseline: str = "wall_wall_primary",
    candidate: str = "direct_tiled",
) -> dict:
    summary = {
        "cases": len(rows),
        "paired": {},
    }
    for metric in METRICS:
        left = _mean(rows, baseline, metric)
        right = _mean(rows, candidate, metric)
        summary["paired"][metric] = {
            "baseline": left,
            "candidate": right,
            "delta": right - left,
            "relative_delta": (right - left) / max(abs(left), 1.0e-6),
        }
    return summary


def stratified_rows(rows: list[dict]) -> dict:
    operations = {
        operation: compare_rows(
            [row for row in rows if row["operation"] == operation]
        )
        for operation in sorted({row["operation"] for row in rows})
    }
    room_bins = {}
    for name, lower, upper in (
        ("rooms_0_4", 0, 4),
        ("rooms_5_7", 5, 7),
        ("rooms_8_plus", 8, math.inf),
    ):
        selected = [
            row
            for row in rows
            if lower
            <= row["metrics"]["wall_wall_primary"]["target_nodes"]
            <= upper
        ]
        room_bins[name] = compare_rows(selected)
    return {"operations": operations, "room_count": room_bins}


def tail_rows(rows: list[dict]) -> dict:
    count = max(1, math.ceil(len(rows) * 0.1))
    orders = {
        "baseline_node_f1_lowest": sorted(
            rows,
            key=lambda row: row["metrics"]["wall_wall_primary"]["node_f1"],
        )[:count],
        "candidate_node_f1_lowest": sorted(
            rows,
            key=lambda row: row["metrics"]["direct_tiled"]["node_f1"],
        )[:count],
        "candidate_ged_highest": sorted(
            rows,
            key=lambda row: row["metrics"]["direct_tiled"]["ged"],
            reverse=True,
        )[:count],
    }
    return {
        name: compare_rows(selected)
        for name, selected in orders.items()
    }


def _distribution_features(rows: list[dict], method: str) -> tuple[np.ndarray, np.ndarray]:
    generated = np.asarray(
        [
            [
                row["metrics"][method]["predicted_nodes"],
                row["metrics"][method]["predicted_edges"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    target = np.asarray(
        [
            [
                row["metrics"][method]["target_nodes"],
                row["metrics"][method]["target_edges"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    joined = np.concatenate([generated, target], axis=0)
    mean = joined.mean(axis=0)
    scale = joined.std(axis=0).clip(min=1.0e-6)
    return (generated - mean) / scale, (target - mean) / scale


def distribution_summary(rows: list[dict], method: str) -> dict:
    generated, target = _distribution_features(rows, method)
    distribution = coverage_density(generated, target)
    bias = np.abs(
        np.asarray(
            [
                [
                    row["metrics"][method]["predicted_nodes"]
                    - row["metrics"][method]["target_nodes"],
                    row["metrics"][method]["predicted_edges"]
                    - row["metrics"][method]["target_edges"],
                ]
                for row in rows
            ]
        )
    ).mean(axis=0)
    return {
        "coverage": distribution["coverage"],
        "density": distribution["density"],
        "mmd": mmd_rbf(generated, target),
        "diversity": mean_pairwise_distance(generated),
        "node_count_mae": float(bias[0]),
        "edge_count_mae": float(bias[1]),
    }


def build_frontier_summary(rows: list[dict]) -> dict:
    overall = compare_rows(rows)
    strata = stratified_rows(rows)
    tails = tail_rows(rows)
    distribution = {
        "baseline": distribution_summary(rows, "wall_wall_primary"),
        "candidate": distribution_summary(rows, "direct_tiled"),
    }
    operation_points = {}
    for operation, values in strata["operations"].items():
        selected = [
            row for row in rows if row["operation"] == operation
        ]
        operation_points[operation] = {
            "baseline": {
                "node_f1": values["paired"]["node_f1"]["baseline"],
                "distribution": distribution_summary(
                    selected,
                    "wall_wall_primary",
                ),
            },
            "candidate": {
                "node_f1": values["paired"]["node_f1"]["candidate"],
                "distribution": distribution_summary(
                    selected,
                    "direct_tiled",
                ),
            },
        }
    return {
        "cases": len(rows),
        "overall": overall,
        "strata": strata,
        "tail": tails,
        "distribution": distribution,
        "operation_points": operation_points,
        "efficiency": {
            "extra_neural_nfe": 0,
            "candidate_budget": "deterministic graph readout over one frozen prediction",
        },
    }


__all__ = [
    "build_frontier_summary",
    "compare_rows",
    "distribution_summary",
    "stratified_rows",
    "tail_rows",
]
