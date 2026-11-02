"""Separate semantic, support, and matched-instance room metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as component_label
from scipy.optimize import linear_sum_assignment

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.conditional_ceiling import room_macro_iou_numpy
from floorflow.representation.core import (
    RoomInstance,
    extract_room_instances,
    instance_iou,
)


@dataclass(frozen=True)
class InstanceMatch:
    predicted_index: int
    target_index: int
    iou: float
    type_correct: bool


def _binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(prediction, dtype=bool)
    expected = np.asarray(target, dtype=bool)
    union = np.logical_or(predicted, expected).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(predicted, expected).sum() / union)


def _binary_components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    components, count = component_label(np.asarray(mask, dtype=bool))
    return [
        components == index
        for index in range(1, int(count) + 1)
        if int((components == index).sum()) >= int(min_area)
    ]


def matched_support_instances(
    predicted_support: np.ndarray,
    target_support: np.ndarray,
    *,
    iou_threshold: float = 0.5,
    min_area: int = 8,
) -> dict[str, float | int]:
    """Match geometry-only room components, independent of semantic labels."""

    predicted = _binary_components(predicted_support, min_area)
    expected = _binary_components(target_support, min_area)
    if not predicted and not expected:
        return {
            "predicted_instances": 0,
            "target_instances": 0,
            "matched_geometric_iou": 1.0,
            "instance_f1_at_threshold": 1.0,
        }
    if not predicted or not expected:
        return {
            "predicted_instances": len(predicted),
            "target_instances": len(expected),
            "matched_geometric_iou": 0.0,
            "instance_f1_at_threshold": 0.0,
        }
    scores = np.zeros((len(predicted), len(expected)), dtype=np.float64)
    for left_index, left in enumerate(predicted):
        for right_index, right in enumerate(expected):
            scores[left_index, right_index] = _binary_iou(left, right)
    rows, columns = linear_sum_assignment(-scores)
    accepted = scores[rows, columns] >= float(iou_threshold)
    precision = float(accepted.sum()) / len(predicted)
    recall = float(accepted.sum()) / len(expected)
    f1 = 2.0 * precision * recall / max(1.0e-8, precision + recall)
    return {
        "predicted_instances": len(predicted),
        "target_instances": len(expected),
        "matched_geometric_iou": float(scores[rows, columns].mean()),
        "instance_f1_at_threshold": float(f1),
    }


def matched_room_instances(
    predicted_instances: list[RoomInstance],
    target_instances: list[RoomInstance],
    *,
    class_constrained: bool = False,
) -> list[InstanceMatch]:
    """Class-agnostic geometry matching; type correctness is reported after."""

    if not predicted_instances and not target_instances:
        return []
    if not predicted_instances or not target_instances:
        return []
    if class_constrained:
        matches = []
        classes = sorted(
            {instance.class_id for instance in predicted_instances}
            | {instance.class_id for instance in target_instances}
        )
        for class_id in classes:
            predicted_subset = [
                (index, instance)
                for index, instance in enumerate(predicted_instances)
                if instance.class_id == class_id
            ]
            target_subset = [
                (index, instance)
                for index, instance in enumerate(target_instances)
                if instance.class_id == class_id
            ]
            if not predicted_subset or not target_subset:
                continue
            local = matched_room_instances(
                [item[1] for item in predicted_subset],
                [item[1] for item in target_subset],
            )
            matches.extend(
                InstanceMatch(
                    predicted_index=predicted_subset[match.predicted_index][0],
                    target_index=target_subset[match.target_index][0],
                    iou=match.iou,
                    type_correct=True,
                )
                for match in local
            )
        return matches
    scores = np.zeros(
        (len(predicted_instances), len(target_instances)),
        dtype=np.float64,
    )
    for predicted_index, predicted in enumerate(predicted_instances):
        for target_index, expected in enumerate(target_instances):
            scores[predicted_index, target_index] = instance_iou(
                predicted,
                expected,
            )
    predicted_indices, target_indices = linear_sum_assignment(-scores)
    return [
        InstanceMatch(
            predicted_index=int(predicted_index),
            target_index=int(target_index),
            iou=float(scores[predicted_index, target_index]),
            type_correct=(
                predicted_instances[predicted_index].class_id
                == target_instances[target_index].class_id
            ),
        )
        for predicted_index, target_index in zip(
            predicted_indices,
            target_indices,
        )
    ]


def matched_instance_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    iou_threshold: float = 0.5,
    min_area: int = 8,
) -> dict[str, float | int]:
    predicted = extract_room_instances(
        np.asarray(prediction),
        taxonomy,
        min_area=min_area,
    )
    expected = extract_room_instances(
        np.asarray(target),
        taxonomy,
        min_area=min_area,
    )
    if not predicted and not expected:
        return {
            "predicted_instances": 0,
            "target_instances": 0,
            "count_abs_error": 0,
            "matched_instances": 0,
            "matched_geometric_iou": 1.0,
            "instance_precision_at_threshold": 1.0,
            "instance_recall_at_threshold": 1.0,
            "instance_f1_at_threshold": 1.0,
            "matched_type_accuracy": 1.0,
            "matched_class_constrained_iou": 1.0,
            "instance_class_constrained_f1_at_threshold": 1.0,
        }
    matches = matched_room_instances(predicted, expected)
    constrained = matched_room_instances(
        predicted,
        expected,
        class_constrained=True,
    )
    accepted = [match for match in matches if match.iou >= iou_threshold]
    precision = len(accepted) / max(1, len(predicted))
    recall = len(accepted) / max(1, len(expected))
    f1 = 2.0 * precision * recall / max(1.0e-8, precision + recall)
    type_accuracy = (
        sum(match.type_correct for match in accepted) / len(accepted)
        if accepted
        else 0.0
    )
    constrained_accepted = [
        match for match in constrained if match.iou >= iou_threshold
    ]
    constrained_precision = len(constrained_accepted) / max(1, len(predicted))
    constrained_recall = len(constrained_accepted) / max(1, len(expected))
    constrained_f1 = (
        2.0
        * constrained_precision
        * constrained_recall
        / max(1.0e-8, constrained_precision + constrained_recall)
    )
    return {
        "predicted_instances": len(predicted),
        "target_instances": len(expected),
        "count_abs_error": abs(len(predicted) - len(expected)),
        "matched_instances": len(accepted),
        "matched_geometric_iou": (
            float(np.mean([match.iou for match in accepted]))
            if accepted
            else 0.0
        ),
        "instance_precision_at_threshold": float(precision),
        "instance_recall_at_threshold": float(recall),
        "instance_f1_at_threshold": float(f1),
        "matched_type_accuracy": float(type_accuracy),
        "matched_class_constrained_iou": (
            float(np.mean([match.iou for match in constrained_accepted]))
            if constrained_accepted
            else 0.0
        ),
        "instance_class_constrained_f1_at_threshold": float(constrained_f1),
    }


def room_metric_suite(
    prediction: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    predicted_room_support: np.ndarray | None = None,
    target_room_support: np.ndarray | None = None,
    occlusion_mask: np.ndarray | None = None,
    predicted_instance_support: np.ndarray | None = None,
) -> dict[str, dict | float | None]:
    """Return three non-interchangeable room scores and their availability."""

    inside = np.asarray(footprint, dtype=bool)
    visible_semantic = room_macro_iou_numpy(
        np.asarray(prediction),
        np.asarray(target),
        taxonomy,
    )
    if (
        predicted_room_support is not None
        and target_room_support is not None
    ):
        support_region = inside
        if occlusion_mask is not None:
            support_region = support_region & np.asarray(
                occlusion_mask,
                dtype=bool,
            )
        support_iou = _binary_iou(
            np.asarray(predicted_room_support, dtype=bool) & support_region,
            np.asarray(target_room_support, dtype=bool) & support_region,
        )
    else:
        support_iou = None
    return {
        "visible_semantic": {
            "room_macro_iou": float(visible_semantic),
            "scope": "exclusive visible raster labels",
        },
        "occlusion_aware_support": {
            "room_support_iou": support_iou,
            "scope": "room support over the requested in-footprint region",
            "available": support_iou is not None,
        },
        "matched_instance": matched_instance_metrics(
            prediction,
            target,
            taxonomy,
        ),
        "support_instance": (
            matched_support_instances(
                np.asarray(
                    predicted_instance_support
                    if predicted_instance_support is not None
                    else predicted_room_support,
                    dtype=bool,
                )
                & inside,
                np.asarray(target_room_support, dtype=bool) & inside,
            )
            if predicted_room_support is not None and target_room_support is not None
            else None
        ),
    }
