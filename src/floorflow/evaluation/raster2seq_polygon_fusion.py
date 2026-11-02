"""Orientation-aware polygon fusion for deterministic Raster2Seq TTA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon


@dataclass(frozen=True)
class FusionConfig:
    """Configuration for one validation-selectable fusion arm."""

    minimum_support: int = 2
    original_weight: float = 1.0
    selected_weight: float = 1.0
    minimum_agreement: float = 0.0
    samples: int = 128
    simplify_tolerance: float = 0.25
    preserve_reference_vertices: bool = True


def signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def canonical_points(item: dict) -> np.ndarray:
    points = np.asarray(item["segmentation"], dtype=float).reshape(-1, 2)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) >= 3 and signed_area(points) < 0.0:
        points = points[::-1]
    return points


def resample(points: np.ndarray, count: int) -> np.ndarray:
    closed = np.concatenate([points, points[:1]], axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    if cumulative[-1] <= 0.0:
        return np.repeat(points[:1], count, axis=0)
    positions = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    output = np.empty((count, 2), dtype=float)
    output[:, 0] = np.interp(positions, cumulative, closed[:, 0])
    output[:, 1] = np.interp(positions, cumulative, closed[:, 1])
    return output


def align_to_reference(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove cyclic vertex shift before averaging two closed contours."""

    if signed_area(candidate) * signed_area(reference) < 0.0:
        candidate = candidate[::-1]
    distances = [
        float(np.mean(np.sum((np.roll(candidate, shift, axis=0) - reference) ** 2, axis=1)))
        for shift in range(len(candidate))
    ]
    return np.roll(candidate, int(np.argmin(distances)), axis=0)


def simplify_average(points: np.ndarray, tolerance: float) -> list[float]:
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return points.reshape(-1).tolist()
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda value: value.area)
    if polygon.geom_type != "Polygon":
        return points.reshape(-1).tolist()
    if tolerance > 0.0:
        polygon = polygon.simplify(tolerance, preserve_topology=True)
    return np.asarray(polygon.exterior.coords[:-1], dtype=float).reshape(-1).tolist()


def polygon_iou(left: dict, right: dict) -> float:
    first = Polygon(canonical_points(left))
    second = Polygon(canonical_points(right))
    if not first.is_valid:
        first = first.buffer(0)
    if not second.is_valid:
        second = second.buffer(0)
    union = first.union(second).area
    return (
        float(first.intersection(second).area / union)
        if union > 0.0
        else 0.0
    )


def fuse_cluster(
    cluster: dict,
    fallback: dict,
    config: FusionConfig,
    selected_source: str | None = None,
) -> dict:
    """Average a supported cluster and otherwise retain a stable fallback."""

    entries = [
        (source, item)
        for source, item in cluster["items"]
        if source == selected_source
        or polygon_iou(fallback, item) >= config.minimum_agreement
    ]
    if len({source for source, _ in entries}) < config.minimum_support:
        return fallback
    original = next(
        (item for source, item in entries if source == "original"),
        None,
    )
    reference_item = next(
        (
            item
            for source, item in entries
            if source == selected_source
        ),
        original or entries[0][1],
    )
    reference_points = canonical_points(reference_item)
    reference = resample(reference_points, config.samples)
    aligned = []
    weights = []
    for source, item in entries:
        candidate = resample(canonical_points(item), config.samples)
        aligned.append(align_to_reference(candidate, reference))
        weights.append(
            config.selected_weight
            if source == selected_source
            else config.original_weight
            if source == "original"
            else 1.0
        )
    weights_array = np.asarray(weights, dtype=float)
    average = np.average(
        np.stack(aligned, axis=0),
        axis=0,
        weights=weights_array,
    )
    if config.preserve_reference_vertices:
        indices = [
            int(
                np.argmin(
                    np.sum((reference - point) ** 2, axis=1)
                )
            )
            for point in reference_points
        ]
        average = average[indices]
    polygon = Polygon(average)
    if polygon.is_empty or polygon.area <= 0.0:
        return fallback
    return {
        **reference_item,
        "segmentation": simplify_average(
            average,
            0.0 if config.preserve_reference_vertices else config.simplify_tolerance,
        ),
    }


def fuse_items(
    items: dict[str, list[dict]],
    clusters: list[dict],
    opening_ids: frozenset[int],
    config: FusionConfig,
) -> list[dict]:
    """Fuse all supported same-type room clusters."""

    output = []
    original = items["original"]
    for cluster in clusters:
        entries = cluster["items"]
        original_item = next(
            (item for source, item in entries if source == "original"),
            None,
        )
        fallback = original_item or entries[0][1]
        output.append(fuse_cluster(cluster, fallback, config))
    output.extend(
        item for item in original if int(item["category_id"]) in opening_ids
    )
    return [
        {**item, "id": index}
        for index, item in enumerate(output)
    ]


def fuse_selected_view(
    items: dict[str, list[dict]],
    selected: str,
    clusters: list[dict],
    opening_ids: frozenset[int],
    config: FusionConfig,
) -> list[dict]:
    """Fuse geometry around one count-consensus-selected prediction."""

    cluster_for = {}
    for cluster in clusters:
        for source, item in cluster["items"]:
            cluster_for[(source, id(item))] = cluster
    output = []
    for item in items[selected]:
        if int(item["category_id"]) in opening_ids:
            output.append(item)
            continue
        cluster = cluster_for.get((selected, id(item)))
        if cluster is None or len(cluster["sources"]) < config.minimum_support:
            output.append(item)
            continue
        output.append(
            fuse_cluster(
                cluster,
                item,
                FusionConfig(
                    minimum_support=config.minimum_support,
                    original_weight=config.original_weight,
                    selected_weight=config.selected_weight,
                    minimum_agreement=config.minimum_agreement,
                    samples=config.samples,
                    simplify_tolerance=config.simplify_tolerance,
                    preserve_reference_vertices=config.preserve_reference_vertices,
                ),
                selected_source=selected,
            )
        )
    return [
        {**item, "id": index}
        for index, item in enumerate(output)
    ]
