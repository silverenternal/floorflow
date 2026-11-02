"""Edit-region and ambiguity strata for room-graph diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from floorflow.evaluation.room_graph_matching import (
    RoomGraph,
    RoomGraphNode,
)

LOCATION_ORDER = (
    "inside",
    "boundary_1",
    "boundary_2",
    "boundary_5",
    "far_outside",
)


@dataclass(frozen=True)
class EditRegion:
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def from_normalized(
        cls,
        values: Iterable[float],
        *,
        width: int,
        height: int,
    ) -> EditRegion:
        x, y, box_width, box_height = (float(value) for value in values)
        return cls(
            left=x * width,
            top=y * height,
            right=(x + box_width) * width,
            bottom=(y + box_height) * height,
        )

    def classify(self, point: tuple[float, float]) -> str:
        x, y = point
        if self.left <= x <= self.right and self.top <= y <= self.bottom:
            return "inside"
        dx = max(self.left - x, 0.0, x - self.right)
        dy = max(self.top - y, 0.0, y - self.bottom)
        distance = max(dx, dy)
        if distance <= 1.0:
            return "boundary_1"
        if distance <= 2.0:
            return "boundary_2"
        if distance <= 5.0:
            return "boundary_5"
        return "far_outside"


def _midpoint(
    left: RoomGraphNode,
    right: RoomGraphNode,
) -> tuple[float, float]:
    return (
        0.5 * (left.centroid[0] + right.centroid[0]),
        0.5 * (left.centroid[1] + right.centroid[1]),
    )


def semantic_error_rows(
    predicted: RoomGraph,
    target: RoomGraph,
    match: dict,
    region: EditRegion,
) -> list[dict]:
    rows = []
    for index in match["node_false_positive_indices"]:
        rows.append(
            {
                "kind": "node_false_positive",
                "location": region.classify(predicted.nodes[index].centroid),
                "predicted_index": int(index),
            }
        )
    for index in match["node_false_negative_indices"]:
        rows.append(
            {
                "kind": "node_false_negative",
                "location": region.classify(target.nodes[index].centroid),
                "target_index": int(index),
            }
        )
    for left, right in match["edge_false_positive"]:
        rows.append(
            {
                "kind": "edge_false_positive",
                "location": region.classify(
                    _midpoint(predicted.nodes[left], predicted.nodes[right])
                ),
                "predicted_edge": [int(left), int(right)],
            }
        )
    for left, right in match["edge_false_negative"]:
        rows.append(
            {
                "kind": "edge_false_negative",
                "location": region.classify(
                    _midpoint(target.nodes[left], target.nodes[right])
                ),
                "target_edge": [int(left), int(right)],
            }
        )
    return rows


def ged_error_rows(
    predicted: RoomGraph,
    target: RoomGraph,
    mapping: dict[int, int],
    region: EditRegion,
    *,
    node_deletions: Iterable[int] | None = None,
    node_insertions: Iterable[int] | None = None,
    edge_deletions: Iterable[tuple[int, int]] | None = None,
    edge_insertions: Iterable[tuple[int, int]] | None = None,
) -> list[dict]:
    rows = []
    mapped_targets = set(mapping.values())
    deletion_indices = (
        list(node_deletions)
        if node_deletions is not None
        else [
            index for index in range(len(predicted.nodes)) if index not in mapping
        ]
    )
    insertion_indices = (
        list(node_insertions)
        if node_insertions is not None
        else [
            index for index in range(len(target.nodes)) if index not in mapped_targets
        ]
    )
    for index in deletion_indices:
        rows.append(
            {
                "kind": "node_deletion",
                "location": region.classify(predicted.nodes[index].centroid),
                "predicted_index": int(index),
            }
        )
    for index in insertion_indices:
        rows.append(
            {
                "kind": "node_insertion",
                "location": region.classify(target.nodes[index].centroid),
                "target_index": int(index),
            }
        )
    for left, right in mapping.items():
        if predicted.nodes[left].label != target.nodes[right].label:
            rows.append(
                {
                    "kind": "node_substitution",
                    "location": region.classify(
                        _midpoint(predicted.nodes[left], target.nodes[right])
                    ),
                    "predicted_index": int(left),
                    "target_index": int(right),
                }
            )
    if edge_deletions is None:
        mapped_edges = {
            tuple(sorted((mapping[left], mapping[right])))
            for left, right in predicted.edges
            if left in mapping and right in mapping
        }
        deletions = mapped_edges - set(target.edges)
    else:
        deletions = {tuple(sorted(edge)) for edge in edge_deletions}
    if edge_insertions is None:
        reverse = {right: left for left, right in mapping.items()}
        predicted_edges = {
            tuple(sorted((mapping[left], mapping[right])))
            for left, right in predicted.edges
            if left in mapping and right in mapping
        }
        target_edges = {
            tuple(sorted((left, right)))
            for left, right in target.edges
            if left in reverse and right in reverse
        }
        insertions = target_edges - predicted_edges
    else:
        insertions = {tuple(sorted(edge)) for edge in edge_insertions}
    for left, right in sorted(deletions):
        rows.append(
            {
                "kind": "edge_deletion",
                "location": region.classify(
                    _midpoint(predicted.nodes[left], predicted.nodes[right])
                ),
                "predicted_edge": [int(left), int(right)],
            }
        )
    for left, right in sorted(insertions):
        rows.append(
            {
                "kind": "edge_insertion",
                "location": region.classify(
                    _midpoint(target.nodes[left], target.nodes[right])
                ),
                "target_edge": [int(left), int(right)],
            }
        )
    return rows


def _many_to_many_incidence(
    predicted: RoomGraph,
    target: RoomGraph,
    *,
    fraction_threshold: float = 0.5,
) -> tuple[list[int], list[int]]:
    predicted_hits = [[] for _ in predicted.nodes]
    target_hits = [[] for _ in target.nodes]
    for predicted_index, predicted_node in enumerate(predicted.nodes):
        for target_index, target_node in enumerate(target.nodes):
            intersection = predicted_node.polygon.intersection(
                target_node.polygon
            ).area
            denominator = min(
                predicted_node.polygon.area,
                target_node.polygon.area,
            )
            if denominator > 0.0 and intersection / denominator >= fraction_threshold:
                predicted_hits[predicted_index].append(target_index)
                target_hits[target_index].append(predicted_index)
    merged = [
        predicted_index
        for predicted_index, hits in enumerate(predicted_hits)
        if len(hits) > 1
    ]
    split = [
        target_index
        for target_index, hits in enumerate(target_hits)
        if len(hits) > 1
    ]
    return merged, split


def ambiguity_summary(predicted: RoomGraph, target: RoomGraph) -> dict:
    same_type_edges = [
        (left, right)
        for left, right in target.edges
        if target.nodes[left].label == target.nodes[right].label
    ]
    low_purity_nodes = [
        node.index for node in target.nodes if node.label_purity < 0.9
    ]
    ambiguous_nodes = [
        node.index
        for node in target.nodes
        if node.label_purity < 0.9 or node.second_label_fraction >= 0.1
    ]
    merged, split = _many_to_many_incidence(predicted, target)
    return {
        "target_same_type_adjacent_edges": len(same_type_edges),
        "target_low_purity_nodes": len(low_purity_nodes),
        "target_ambiguous_nodes": len(ambiguous_nodes),
        "target_undefined_nodes": sum(
            node.label == "undefined" for node in target.nodes
        ),
        "predicted_undefined_nodes": sum(
            node.label == "undefined" for node in predicted.nodes
        ),
        "merged_predicted_nodes": len(merged),
        "split_target_nodes": len(split),
        "merged_predicted_indices": merged,
        "split_target_indices": split,
        "mean_target_label_purity": (
            float(np.mean([node.label_purity for node in target.nodes]))
            if target.nodes
            else 1.0
        ),
    }


def error_strata(rows: Iterable[dict]) -> dict[str, dict[str, int]]:
    strata = {
        kind: {location: 0 for location in LOCATION_ORDER}
        for kind in (
            "node_false_positive",
            "node_false_negative",
            "edge_false_positive",
            "edge_false_negative",
            "node_deletion",
            "node_insertion",
            "node_substitution",
            "edge_deletion",
            "edge_insertion",
        )
    }
    for row in rows:
        kind = row["kind"]
        location = row["location"]
        if kind in strata and location in strata[kind]:
            strata[kind][location] += 1
    return strata


def summarize_values(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def quantile_summary(
    covariate: Iterable[float],
    metric: Iterable[float],
) -> dict:
    x = np.asarray(list(covariate), dtype=np.float64)
    y = np.asarray(list(metric), dtype=np.float64)
    if x.size < 3 or np.allclose(x, x[0]):
        return {"bins": [], "spearman": None, "p_value": None}
    edges = np.unique(np.quantile(x, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]))
    bins = []
    for index in range(len(edges) - 1):
        left, right = edges[index], edges[index + 1]
        if index == len(edges) - 2:
            selected = (x >= left) & (x <= right)
        else:
            selected = (x >= left) & (x < right)
        if not selected.any():
            continue
        bins.append(
            {
                "range": [float(left), float(right)],
                "cases": int(selected.sum()),
                "mean_metric": float(y[selected].mean()),
                "median_metric": float(np.median(y[selected])),
            }
        )
    correlation = spearmanr(x, y)
    return {
        "bins": bins,
        "spearman": float(correlation.statistic),
        "p_value": float(correlation.pvalue),
    }


__all__ = [
    "LOCATION_ORDER",
    "EditRegion",
    "ambiguity_summary",
    "error_strata",
    "ged_error_rows",
    "quantile_summary",
    "semantic_error_rows",
    "summarize_values",
]
