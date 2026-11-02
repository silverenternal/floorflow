"""Decode opening centers and widths directly on wall centerlines."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.opening_readout import OpeningDetection
from floorflow.geometry.wall_graph_readout import RasterWallPlan, ReadoutOpening


def _sample(
    values: np.ndarray,
    point: np.ndarray,
) -> np.ndarray | float:
    height, width = values.shape[-2:]
    column = min(max(round(float(point[0])), 0), width - 1)
    row = min(max(round(float(point[1])), 0), height - 1)
    return values[..., row, column]


def _nearest_wall_index(
    center: tuple[float, float],
    plan: RasterWallPlan,
) -> int:
    if not plan.walls:
        return 0
    point = np.asarray(center, dtype=np.float64)
    best_index = 0
    best_distance = float("inf")
    for index, wall in enumerate(plan.walls):
        start = np.asarray(wall.start, dtype=np.float64)
        end = np.asarray(wall.end, dtype=np.float64)
        delta = end - start
        denominator = float(delta @ delta)
        fraction = (
            0.0
            if denominator <= 1.0e-9
            else float(np.clip((point - start) @ delta / denominator, 0.0, 1.0))
        )
        distance = float(np.linalg.norm(point - (start + fraction * delta)))
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def _bounded_opening_width(
    predicted_width: float,
    minimum_width: float,
    host_length: float,
) -> float:
    minimum = min(float(minimum_width), float(host_length))
    return min(max(float(predicted_width), minimum), float(host_length))


def read_center_width_opening_detections(
    plan: RasterWallPlan,
    center_probability: np.ndarray,
    interval_width_fraction: np.ndarray,
    family_logits: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float = 0.30,
    min_width: float = 3.0,
    min_peak_distance: float = 4.0,
    max_per_wall: int = 8,
    family_window_bias: float = 0.0,
) -> list[OpeningDetection]:
    """Read local center peaks and their predicted 1D opening widths."""

    center_probability = np.asarray(center_probability, dtype=np.float32)
    width_fraction = np.asarray(
        interval_width_fraction,
        dtype=np.float32,
    )
    family_logits = np.asarray(family_logits, dtype=np.float32)
    if center_probability.shape != width_fraction.shape:
        raise ValueError("center and width maps must share a shape")
    if family_logits.ndim != 3 or family_logits.shape[0] != 2:
        raise ValueError("family logits must be [2,H,W]")
    door_id = next(
        class_id
        for class_id in taxonomy.opening_ids
        if taxonomy.class_names[class_id].startswith("door:")
    )
    window_id = next(
        class_id
        for class_id in taxonomy.opening_ids
        if taxonomy.class_names[class_id].startswith("window:")
    )
    detections = []
    for wall_index, wall in enumerate(plan.walls):
        start = np.asarray(wall.start, dtype=np.float64)
        end = np.asarray(wall.end, dtype=np.float64)
        length = float(np.linalg.norm(end - start))
        if length <= 1.0e-6:
            continue
        sample_count = max(3, int(np.ceil(length)) + 1)
        fractions = np.linspace(0.0, 1.0, sample_count)
        points = start[None] + fractions[:, None] * (end - start)[None]
        scores = np.asarray(
            [_sample(center_probability, point) for point in points],
            dtype=np.float32,
        )
        candidates = []
        for index in range(sample_count):
            left = scores[index - 1] if index else -1.0
            right = scores[index + 1] if index + 1 < sample_count else -1.0
            if scores[index] < threshold:
                continue
            if scores[index] < left or scores[index] < right:
                continue
            candidates.append(index)
        selected = []
        for index in sorted(candidates, key=lambda item: (-scores[item], item)):
            point = points[index]
            if any(
                np.linalg.norm(point - points[other])
                < float(min_peak_distance)
                for other in selected
            ):
                continue
            selected.append(index)
            if len(selected) >= max_per_wall:
                break
        for index in selected:
            point = points[index]
            width = _bounded_opening_width(
                float(_sample(width_fraction, point)) * float(plan.frame),
                min_width,
                length,
            )
            half_span = 0.5 * width / max(1.0, length)
            family_fractions = np.linspace(
                max(0.0, fractions[index] - half_span),
                min(1.0, fractions[index] + half_span),
                7,
            )
            family_scores = np.stack(
                [
                    _sample(
                        family_logits,
                        start + float(fraction) * (end - start),
                    )
                    for fraction in family_fractions
                ]
            ).mean(axis=0)
            family_scores = family_scores.copy()
            family_scores[1] += float(family_window_bias)
            family_index = int(family_scores.argmax())
            class_id = door_id if family_index == 0 else window_id
            detections.append(
                OpeningDetection(
                    center=(float(point[0]), float(point[1])),
                    width=width,
                    family="window" if family_index == 1 else "door",
                    class_id=int(class_id),
                    score=float(scores[index]),
                    wall_index=wall_index,
                    wall_distance=0.0,
                )
            )
    return detections


def read_center_width_image_detections(
    plan: RasterWallPlan,
    center_probability: np.ndarray,
    interval_width_fraction: np.ndarray,
    family_logits: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float = 0.30,
    min_width: float = 3.0,
    min_peak_distance: float = 8.0,
    max_wall_distance: float = 12.0,
    max_detections: int = 96,
) -> list[OpeningDetection]:
    """Decode 2D center peaks and associate them with the nearest plan wall."""

    center_probability = np.asarray(center_probability, dtype=np.float32)
    width_fraction = np.asarray(interval_width_fraction, dtype=np.float32)
    family_logits = np.asarray(family_logits, dtype=np.float32)
    radius = max(1, round(float(min_peak_distance) / 2.0))
    local_max = maximum_filter(center_probability, size=2 * radius + 1)
    ys, xs = np.nonzero(
        (center_probability >= float(threshold))
        & (center_probability >= local_max)
    )
    order = sorted(
        range(len(xs)),
        key=lambda index: (-float(center_probability[ys[index], xs[index]]), int(xs[index]), int(ys[index])),
    )
    selected: list[int] = []
    for index in order:
        point = np.asarray([xs[index], ys[index]], dtype=np.float64)
        if any(
            np.linalg.norm(
                point
                - np.asarray(
                    [xs[other], ys[other]],
                    dtype=np.float64,
                )
            )
            < float(min_peak_distance)
            for other in selected
        ):
            continue
        selected.append(index)
        if len(selected) >= max_detections:
            break
    door_id = next(
        class_id
        for class_id in taxonomy.opening_ids
        if taxonomy.class_names[class_id].startswith("door:")
    )
    window_id = next(
        class_id
        for class_id in taxonomy.opening_ids
        if taxonomy.class_names[class_id].startswith("window:")
    )
    detections = []
    for index in selected:
        point = np.asarray([xs[index], ys[index]], dtype=np.float64)
        best_wall = None
        best_distance = float("inf")
        for wall_index, wall in enumerate(plan.walls):
            start = np.asarray(wall.start, dtype=np.float64)
            end = np.asarray(wall.end, dtype=np.float64)
            delta = end - start
            denominator = float(delta @ delta)
            fraction = (
                0.0
                if denominator <= 1.0e-9
                else float(np.clip((point - start) @ delta / denominator, 0.0, 1.0))
            )
            distance = float(np.linalg.norm(point - (start + fraction * delta)))
            if distance < best_distance:
                best_wall = wall_index
                best_distance = distance
        if best_wall is None or best_distance > max_wall_distance:
            continue
        host = plan.walls[best_wall]
        host_length = float(
            np.linalg.norm(np.asarray(host.end) - np.asarray(host.start))
        )
        start = np.asarray(host.start, dtype=np.float64)
        end = np.asarray(host.end, dtype=np.float64)
        point = np.asarray([xs[index], ys[index]], dtype=np.float64)
        width = _bounded_opening_width(
            float(width_fraction[ys[index], xs[index]])
            * float(plan.frame),
            min_width,
            host_length,
        )
        half_span = 0.5 * width / max(1.0, host_length)
        center_fraction = float(
            np.clip(
                (point - start) @ (end - start)
                / max(float((end - start) @ (end - start)), 1.0e-9),
                0.0,
                1.0,
            )
        )
        family_fractions = np.linspace(
            max(0.0, center_fraction - half_span),
            min(1.0, center_fraction + half_span),
            7,
        )
        family_scores = np.stack(
            [
                _sample(
                    family_logits,
                    start + float(fraction) * (end - start),
                )
                for fraction in family_fractions
            ]
        ).mean(axis=0)
        family_index = int(family_scores.argmax())
        class_id = door_id if family_index == 0 else window_id
        detections.append(
            OpeningDetection(
                center=(float(xs[index]), float(ys[index])),
                width=width,
                family="window" if family_index == 1 else "door",
                class_id=int(class_id),
                score=float(center_probability[ys[index], xs[index]]),
                wall_index=best_wall,
                wall_distance=best_distance,
            )
        )
    return detections


def attach_center_width_opening_detections(
    plan: RasterWallPlan,
    center_probability: np.ndarray,
    interval_width_fraction: np.ndarray,
    family_logits: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float = 0.30,
    min_width: float = 3.0,
    min_peak_distance: float = 4.0,
    candidate_ranker=None,
    host_plan: RasterWallPlan | None = None,
    decode_mode: str = "wall",
    max_wall_distance: float = 12.0,
    precomputed_detections: list[OpeningDetection] | None = None,
    remap_wall: bool = False,
) -> int:
    """Replace only the opening layer with center-width detections."""

    host = host_plan or plan
    if decode_mode not in {"wall", "image"}:
        raise ValueError("decode_mode must be wall or image")
    detections = precomputed_detections
    if detections is None:
        detections = (
            read_center_width_image_detections(
                host,
                center_probability,
                interval_width_fraction,
                family_logits,
                taxonomy,
                threshold=threshold,
                min_width=min_width,
                min_peak_distance=min_peak_distance,
                max_wall_distance=max_wall_distance,
            )
            if decode_mode == "image"
            else read_center_width_opening_detections(
                host,
                center_probability,
                interval_width_fraction,
                family_logits,
                taxonomy,
                threshold=threshold,
                min_width=min_width,
                min_peak_distance=min_peak_distance,
            )
        )
    if candidate_ranker is not None and precomputed_detections is None:
        detections = candidate_ranker.rank(
            detections,
            center_probability,
            family_logits,
            host,
            taxonomy,
            frame=float(plan.frame),
        )
    set_opening_detections(plan, detections, remap_wall=remap_wall)
    return len(plan.openings)


def set_opening_detections(
    plan: RasterWallPlan,
    detections: list[OpeningDetection],
    *,
    remap_wall: bool = False,
) -> None:
    """Attach openings while preserving their decoded host-wall identity."""

    plan.openings = [
        ReadoutOpening(
            center=detection.center,
            width=detection.width,
            kind=detection.family,
            wall_index=(
                _nearest_wall_index(detection.center, plan)
                if remap_wall
                else detection.wall_index
            ),
        )
        for detection in detections
    ]


__all__ = [
    "attach_center_width_opening_detections",
    "read_center_width_image_detections",
    "read_center_width_opening_detections",
    "set_opening_detections",
]
