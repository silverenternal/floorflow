"""Detection-style door/window readout and family-aware matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing
from scipy.ndimage import label as component_label
from scipy.optimize import linear_sum_assignment

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import RasterWallPlan


@dataclass(frozen=True)
class OpeningDetection:
    center: tuple[float, float]
    width: float
    family: str
    class_id: int
    score: float
    wall_index: int | None
    wall_distance: float | None


def _nearest_wall(
    center: tuple[float, float],
    plan: RasterWallPlan | None,
) -> tuple[int | None, float | None]:
    if plan is None or not plan.walls:
        return None, None
    best = None
    distance = float("inf")
    for index, wall in enumerate(plan.walls):
        start = np.asarray(wall.start)
        end = np.asarray(wall.end)
        delta = end - start
        scale = float(delta @ delta)
        point = np.asarray(center)
        if scale <= 1.0e-9:
            candidate = float(np.linalg.norm(point - start))
        else:
            fraction = np.clip((point - start) @ delta / scale, 0.0, 1.0)
            candidate = float(np.linalg.norm(point - (start + fraction * delta)))
        if candidate < distance:
            best = index
            distance = candidate
    return best, distance


def read_opening_detections(
    opening_logits: np.ndarray,
    opening_ids: tuple[int, ...],
    class_names: tuple[str, ...],
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float,
    wall_plan: RasterWallPlan | None = None,
    min_area: int = 2,
    max_wall_distance: float = 4.0,
) -> list[OpeningDetection]:
    detections = []
    for local_index, class_id in enumerate(opening_ids):
        probability = opening_logits[local_index]
        components, count = component_label(probability >= threshold)
        for component_id in range(1, int(count) + 1):
            mask = components == component_id
            if mask.sum() < min_area:
                continue
            ys, xs = np.nonzero(mask)
            center = (float(xs.mean()), float(ys.mean()))
            family = (
                "window"
                if class_names[class_id].startswith("window:")
                else "door"
            )
            wall_index, wall_distance = _nearest_wall(center, wall_plan)
            if (
                wall_plan is not None
                and wall_distance is not None
                and wall_distance > max_wall_distance
            ):
                continue
            detections.append(
                OpeningDetection(
                    center=center,
                    width=float(
                        max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
                    ),
                    family=family,
                    class_id=int(class_id),
                    score=float(probability[mask].mean()),
                    wall_index=wall_index,
                    wall_distance=wall_distance,
                )
            )
    return detections


def _opening_component_class(
    labels: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    family_probability: np.ndarray | None,
) -> tuple[int, str]:
    counts = np.bincount(
        labels[ys, xs],
        minlength=len(taxonomy.class_names),
    )
    family_name = None
    if family_probability is not None:
        family_index = int(
            family_probability[:, ys, xs].mean(axis=1).argmax()
        )
        family_name = "window" if family_index == 1 else "door"
        candidates = [
            class_id
            for class_id in taxonomy.opening_ids
            if taxonomy.class_names[class_id].startswith(
                f"{family_name}:"
            )
        ]
        if not candidates:
            candidates = list(taxonomy.opening_ids)
            family_name = None
    else:
        candidates = list(taxonomy.opening_ids)
    class_id = max(candidates, key=lambda value: counts[value])
    return class_id, (
        family_name
        if family_name is not None
        else (
            "window"
            if taxonomy.class_names[class_id].startswith("window:")
            else "door"
        )
    )


def consolidate_opening_components(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    family_probability: np.ndarray,
) -> np.ndarray:
    """Assign one learned door/window family to each connected opening."""

    family = np.asarray(family_probability, dtype=np.float32)
    if family.ndim != 3 or family.shape[0] != 2:
        raise ValueError("family probability must be [2,H,W]")
    output = np.asarray(labels).copy()
    opening_mask = np.isin(output, np.asarray(taxonomy.opening_ids))
    components, count = component_label(opening_mask)
    for component_id in range(1, int(count) + 1):
        mask = components == component_id
        ys, xs = np.nonzero(mask)
        class_id, _family = _opening_component_class(
            output,
            ys,
            xs,
            taxonomy,
            family,
        )
        output[mask] = class_id
    return output


def detections_from_labels(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    wall_plan: RasterWallPlan | None = None,
    family_probability: np.ndarray | None = None,
) -> list[OpeningDetection]:
    if family_probability is not None:
        family_probability = np.asarray(family_probability, dtype=np.float32)
        if family_probability.ndim != 3 or family_probability.shape[0] != 2:
            raise ValueError("family probability must be [2,H,W]")
    detections = []
    opening_mask = np.isin(labels, np.asarray(taxonomy.opening_ids))
    components, count = component_label(opening_mask)
    for component_id in range(1, int(count) + 1):
        mask = components == component_id
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        class_id, family_name = _opening_component_class(
            labels,
            ys,
            xs,
            taxonomy,
            family_probability,
        )
        center = (float(xs.mean()), float(ys.mean()))
        wall_index, wall_distance = _nearest_wall(center, wall_plan)
        detections.append(
            OpeningDetection(
                center=center,
                width=float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)),
                family=family_name,
                class_id=class_id,
                score=1.0,
                wall_index=wall_index,
                wall_distance=wall_distance,
            )
        )
    return detections


def read_group_opening_detections(
    foreground_probability: np.ndarray,
    opening_logits: np.ndarray,
    opening_ids: tuple[int, ...],
    class_names: tuple[str, ...],
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float,
    wall_plan: RasterWallPlan | None = None,
    min_area: int = 3,
    max_wall_distance: float = 4.0,
) -> list[OpeningDetection]:
    components, count = component_label(foreground_probability >= threshold)
    detections = []
    for component_id in range(1, int(count) + 1):
        mask = components == component_id
        if mask.sum() < min_area:
            continue
        ys, xs = np.nonzero(mask)
        local_class = int(
            opening_logits[:, mask].mean(axis=1).argmax()
        )
        center = (float(xs.mean()), float(ys.mean()))
        wall_index, wall_distance = _nearest_wall(center, wall_plan)
        if (
            wall_plan is not None
            and wall_distance is not None
            and wall_distance > max_wall_distance
        ):
            continue
        class_id = int(opening_ids[local_class])
        detections.append(
            OpeningDetection(
                center=center,
                width=float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)),
                family=(
                    "window"
                    if class_names[class_id].startswith("window:")
                    else "door"
                ),
                class_id=class_id,
                score=float(foreground_probability[mask].mean()),
                wall_index=wall_index,
                wall_distance=wall_distance,
            )
        )
    return detections


def read_wall_interval_detections(
    foreground_probability: np.ndarray,
    opening_logits: np.ndarray,
    opening_ids: tuple[int, ...],
    class_names: tuple[str, ...],
    wall_plan: RasterWallPlan,
    *,
    threshold: float,
    radius: int = 2,
    min_width: float = 3.0,
    max_per_wall: int = 2,
    score_smoothing: int = 1,
    max_gap_samples: int = 0,
    family_aggregation: str = "center",
    exterior_wall_window_bias: float = 0.0,
    peak_relative_threshold: float = 0.0,
) -> list[OpeningDetection]:
    """Read 1D opening intervals from probability along each wall centerline."""

    if family_aggregation not in {"center", "interval"}:
        raise ValueError("family_aggregation must be center or interval")
    height, width = foreground_probability.shape
    detections = []
    for wall_index, wall in enumerate(wall_plan.walls):
        start = np.asarray(wall.start, dtype=np.float64)
        end = np.asarray(wall.end, dtype=np.float64)
        length = float(np.linalg.norm(end - start))
        if length <= 1.0e-6:
            continue
        samples = max(2, int(np.ceil(length)) + 1)
        fractions = np.linspace(0.0, 1.0, samples)
        points = start[None] + fractions[:, None] * (end - start)[None]
        scores = []
        for x, y in points:
            x = min(max(float(x), 0.0), float(width - 1))
            y = min(max(float(y), 0.0), float(height - 1))
            left = max(0, round(x) - radius)
            right = min(width, round(x) + radius + 1)
            top = max(0, round(y) - radius)
            bottom = min(height, round(y) + radius + 1)
            scores.append(float(foreground_probability[top:bottom, left:right].max()))
        scores_array = np.asarray(scores)
        if score_smoothing > 1:
            scores_array = np.convolve(
                scores_array,
                np.ones(int(score_smoothing)) / float(score_smoothing),
                mode="same",
            )
        active = scores_array >= threshold
        if max_gap_samples > 0:
            active = binary_closing(
                active,
                structure=np.ones(2 * int(max_gap_samples) + 1, dtype=bool),
            )
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False])
        wall_detections = []
        spans = []
        for first, last in zip(starts, ends):
            if peak_relative_threshold <= 0.0:
                spans.append((int(first), int(last)))
                continue
            local_scores = scores_array[first : last + 1]
            local_threshold = max(
                float(threshold),
                float(local_scores.max()) * float(peak_relative_threshold),
            )
            local_active = local_scores >= local_threshold
            local_starts = np.flatnonzero(
                local_active & ~np.r_[False, local_active[:-1]]
            )
            local_ends = np.flatnonzero(
                local_active & ~np.r_[local_active[1:], False]
            )
            spans.extend(
                (int(first + start), int(first + end))
                for start, end in zip(local_starts, local_ends)
            )
        for first, last in spans:
            interval_width = float((last - first + 1) / max(1, samples - 1) * length)
            if interval_width < min_width:
                continue
            fraction = float((first + last) / (2.0 * max(1, samples - 1)))
            center = start + fraction * (end - start)
            x = min(max(round(float(center[0])), 0), width - 1)
            y = min(max(round(float(center[1])), 0), height - 1)
            if family_aggregation == "interval":
                family_samples = []
                for sample_x, sample_y in points[first : last + 1]:
                    sample_x = min(max(round(float(sample_x)), 0), width - 1)
                    sample_y = min(max(round(float(sample_y)), 0), height - 1)
                    family_samples.append(opening_logits[:, sample_y, sample_x])
                family_scores = np.mean(np.stack(family_samples), axis=0)
            else:
                family_scores = opening_logits[
                    :,
                    max(0, y - radius) : min(height, y + radius + 1),
                    max(0, x - radius) : min(width, x + radius + 1),
                ].mean(axis=(1, 2))
            if wall.wall_type == "exterior":
                family_scores = np.asarray(family_scores, dtype=np.float64).copy()
                family_scores[1] += float(exterior_wall_window_bias)
            local_class = int(family_scores.argmax())
            class_id = int(opening_ids[local_class])
            wall_detections.append(
                OpeningDetection(
                    center=(float(center[0]), float(center[1])),
                    width=interval_width,
                    family=(
                        "window"
                        if class_names[class_id].startswith("window:")
                        else "door"
                    ),
                    class_id=class_id,
                    score=float(np.mean(scores_array[first : last + 1])),
                    wall_index=wall_index,
                    wall_distance=0.0,
                )
            )
        wall_detections.sort(key=lambda item: item.score, reverse=True)
        detections.extend(wall_detections[: max(1, int(max_per_wall))])
    return detections


def opening_detection_metrics(
    predicted: list[OpeningDetection],
    target: list[OpeningDetection],
    *,
    tolerance: float,
) -> dict[str, float | int]:
    if not predicted and not target:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "matched": 0,
            "geometry_matched": 0,
            "family_consistent": 0,
            "door_to_window": 0,
            "window_to_door": 0,
            "unmatched_predicted": 0,
            "unmatched_target": 0,
            "predicted_doors": 0,
            "predicted_windows": 0,
            "target_doors": 0,
            "target_windows": 0,
            "matched_doors": 0,
            "matched_windows": 0,
            "predicted_door_width_mean": 0.0,
            "predicted_window_width_mean": 0.0,
            "target_door_width_mean": 0.0,
            "target_window_width_mean": 0.0,
            "geometry_center_error_mean": 0.0,
            "geometry_width_ratio_mean": 0.0,
        }
    cost = np.full((len(predicted), len(target)), np.inf)
    for left_index, left in enumerate(predicted):
        for right_index, right in enumerate(target):
            distance = float(np.linalg.norm(np.asarray(left.center) - np.asarray(right.center)))
            width_ratio = min(left.width, right.width) / max(1.0, left.width, right.width)
            if left.family == right.family and distance <= tolerance and width_ratio >= 0.3:
                cost[left_index, right_index] = distance
    finite = np.isfinite(cost)
    if finite.any():
        rows, columns = linear_sum_assignment(np.where(finite, cost, 1.0e9))
        matched = int(sum(finite[row, column] for row, column in zip(rows, columns)))
    else:
        matched = 0
    precision = matched / max(1, len(predicted))
    recall = matched / max(1, len(target))
    geometry_cost = np.full((len(predicted), len(target)), np.inf)
    for left_index, left in enumerate(predicted):
        for right_index, right in enumerate(target):
            distance = float(
                np.linalg.norm(
                    np.asarray(left.center) - np.asarray(right.center)
                )
            )
            width_ratio = min(left.width, right.width) / max(
                1.0,
                left.width,
                right.width,
            )
            if distance <= tolerance and width_ratio >= 0.3:
                geometry_cost[left_index, right_index] = distance
    geometry_rows, geometry_columns = linear_sum_assignment(
        np.where(np.isfinite(geometry_cost), geometry_cost, 1.0e9)
    )
    geometry_pairs = [
        (int(row), int(column))
        for row, column in zip(geometry_rows, geometry_columns, strict=True)
        if np.isfinite(geometry_cost[row, column])
    ]
    family_consistent = sum(
        predicted[row].family == target[column].family
        for row, column in geometry_pairs
    )
    door_to_window = sum(
        predicted[row].family == "door" and target[column].family == "window"
        for row, column in geometry_pairs
    )
    window_to_door = sum(
        predicted[row].family == "window" and target[column].family == "door"
        for row, column in geometry_pairs
    )
    center_errors = [
        float(
            np.linalg.norm(
                np.asarray(predicted[row].center)
                - np.asarray(target[column].center)
            )
        )
        for row, column in geometry_pairs
    ]
    width_ratios = [
        min(predicted[row].width, target[column].width)
        / max(1.0, predicted[row].width, target[column].width)
        for row, column in geometry_pairs
    ]
    matched_doors = sum(
        predicted[row].family == "door" and target[column].family == "door"
        for row, column in geometry_pairs
    )
    matched_windows = sum(
        predicted[row].family == "window" and target[column].family == "window"
        for row, column in geometry_pairs
    )
    predicted_door_widths = [
        item.width for item in predicted if item.family == "door"
    ]
    predicted_window_widths = [
        item.width for item in predicted if item.family == "window"
    ]
    target_door_widths = [
        item.width for item in target if item.family == "door"
    ]
    target_window_widths = [
        item.width for item in target if item.family == "window"
    ]
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(1.0e-8, precision + recall)),
        "matched": matched,
        "geometry_matched": len(geometry_pairs),
        "family_consistent": int(family_consistent),
        "door_to_window": int(door_to_window),
        "window_to_door": int(window_to_door),
        "unmatched_predicted": len(predicted) - len(geometry_pairs),
        "unmatched_target": len(target) - len(geometry_pairs),
        "predicted_doors": sum(item.family == "door" for item in predicted),
        "predicted_windows": sum(item.family == "window" for item in predicted),
        "target_doors": sum(item.family == "door" for item in target),
        "target_windows": sum(item.family == "window" for item in target),
        "matched_doors": int(matched_doors),
        "matched_windows": int(matched_windows),
        "predicted_door_width_mean": (
            float(np.mean(predicted_door_widths))
            if predicted_door_widths
            else 0.0
        ),
        "predicted_window_width_mean": (
            float(np.mean(predicted_window_widths))
            if predicted_window_widths
            else 0.0
        ),
        "target_door_width_mean": (
            float(np.mean(target_door_widths)) if target_door_widths else 0.0
        ),
        "target_window_width_mean": (
            float(np.mean(target_window_widths))
            if target_window_widths
            else 0.0
        ),
        "geometry_center_error_mean": (
            float(np.mean(center_errors)) if center_errors else 0.0
        ),
        "geometry_width_ratio_mean": (
            float(np.mean(width_ratios)) if width_ratios else 0.0
        ),
    }
