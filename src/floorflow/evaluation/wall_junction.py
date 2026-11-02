"""Target-matched wall-junction metrics for detected wall segments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class WallJunction:
    center: tuple[float, float]
    degree: int


def cluster_wall_endpoints(
    walls,
    *,
    tolerance: float = 2.0,
) -> list[WallJunction]:
    """Cluster nearby segment endpoints and return their incident degree."""

    endpoints = [point for wall in walls for point in (wall.start, wall.end)]
    clusters: list[list[tuple[float, float]]] = []
    for point in endpoints:
        for cluster in clusters:
            center = (
                sum(item[0] for item in cluster) / len(cluster),
                sum(item[1] for item in cluster) / len(cluster),
            )
            if math.dist(point, center) <= float(tolerance):
                cluster.append(point)
                break
        else:
            clusters.append([point])
    return [
        WallJunction(
            center=(
                sum(item[0] for item in cluster) / len(cluster),
                sum(item[1] for item in cluster) / len(cluster),
            ),
            degree=len(cluster),
        )
        for cluster in clusters
    ]


def _degree_strata(
    junctions: list[WallJunction],
    indices: list[int],
) -> dict[str, float]:
    if not indices:
        return {"L": 0.0, "T": 0.0, "cross": 0.0, "open": 0.0}
    degrees = np.asarray([junctions[index].degree for index in indices])
    return {
        "L": float(np.mean(degrees == 2)),
        "T": float(np.mean(degrees == 3)),
        "cross": float(np.mean(degrees >= 4)),
        "open": float(np.mean(degrees < 2)),
    }


def wall_junction_metrics(
    predicted_walls,
    target_walls,
    *,
    cluster_tolerance: float = 2.0,
    match_distance: float = 6.0,
) -> dict[str, float | int]:
    """Match predicted junctions to target junctions by spatial distance."""

    predicted = cluster_wall_endpoints(
        predicted_walls,
        tolerance=cluster_tolerance,
    )
    target = cluster_wall_endpoints(
        target_walls,
        tolerance=cluster_tolerance,
    )
    if not predicted and not target:
        empty = {"L": 0.0, "T": 0.0, "cross": 0.0, "open": 0.0}
        return {
            "junction_precision": 1.0,
            "junction_recall": 1.0,
            "junction_f1": 1.0,
            "degree_exact_rate": 1.0,
            "degree_mae": 0.0,
            "mean_match_distance": 0.0,
            "predicted_junctions": 0,
            **{f"predicted_{key}": value for key, value in empty.items()},
            **{f"target_{key}": value for key, value in empty.items()},
        }
    if not predicted or not target:
        predicted_strata = _degree_strata(
            predicted,
            list(range(len(predicted))),
        )
        target_strata = _degree_strata(target, list(range(len(target))))
        return {
            "junction_precision": 0.0,
            "junction_recall": 0.0,
            "junction_f1": 0.0,
            "degree_exact_rate": 0.0,
            "degree_mae": float("nan"),
            "mean_match_distance": float("nan"),
            "predicted_junctions": len(predicted),
            "target_junctions": len(target),
            **{
                f"predicted_{key}": value
                for key, value in predicted_strata.items()
            },
            **{
                f"target_{key}": value
                for key, value in target_strata.items()
            },
        }
    distance = np.zeros((len(predicted), len(target)), dtype=np.float64)
    for left, predicted_junction in enumerate(predicted):
        for right, target_junction in enumerate(target):
            distance[left, right] = math.dist(
                predicted_junction.center,
                target_junction.center,
            )
    rows, columns = linear_sum_assignment(distance)
    accepted = [
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
        if distance[row, column] <= float(match_distance)
    ]
    precision = len(accepted) / len(predicted)
    recall = len(accepted) / len(target)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    matched_degrees = [
        (predicted[row].degree, target[column].degree)
        for row, column in accepted
    ]
    degree_exact = (
        float(np.mean([left == right for left, right in matched_degrees]))
        if matched_degrees
        else 0.0
    )
    degree_mae = (
        float(np.mean([abs(left - right) for left, right in matched_degrees]))
        if matched_degrees
        else float("nan")
    )
    return {
        "junction_precision": float(precision),
        "junction_recall": float(recall),
        "junction_f1": float(f1),
        "degree_exact_rate": degree_exact,
        "degree_mae": degree_mae,
        "mean_match_distance": (
            float(np.mean([distance[row, column] for row, column in accepted]))
            if accepted
            else float("nan")
        ),
        "predicted_junctions": len(predicted),
        "target_junctions": len(target),
        **{
            f"predicted_{key}": value
            for key, value in _degree_strata(
                predicted,
                list(range(len(predicted))),
            ).items()
        },
        **{
            f"target_{key}": value
            for key, value in _degree_strata(
                target,
                list(range(len(target))),
            ).items()
        },
    }


__all__ = ["WallJunction", "cluster_wall_endpoints", "wall_junction_metrics"]
