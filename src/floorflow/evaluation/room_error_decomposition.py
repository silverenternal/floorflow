"""Decompose room IoU into occupancy, semantics, instances, and boundaries."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, label

from floorflow.data.taxonomy import FloorplanTaxonomy


def _room_ids(taxonomy: FloorplanTaxonomy, device: torch.device) -> torch.Tensor:
    return torch.tensor(taxonomy.room_ids, device=device, dtype=torch.long)


def _component_sets(mask: np.ndarray) -> list[np.ndarray]:
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    components, count = label(np.asarray(mask, dtype=bool), structure=structure)
    return [(components == index) for index in range(1, int(count) + 1)]


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask)
    structure = np.ones((3, 3), dtype=bool)
    return mask & ~binary_erosion(mask, structure=structure, border_value=0)


def _boundary_scores(
    predicted: np.ndarray,
    target: np.ndarray,
    tolerance: int,
) -> tuple[float, float, float]:
    predicted_boundary = _boundary(predicted)
    target_boundary = _boundary(target)
    if not predicted_boundary.any() and not target_boundary.any():
        return 1.0, 1.0, 1.0
    structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    predicted_hit = predicted_boundary & binary_dilation(
        target_boundary,
        structure=structure,
    )
    target_hit = target_boundary & binary_dilation(
        predicted_boundary,
        structure=structure,
    )
    precision = float(predicted_hit.sum() / max(1, predicted_boundary.sum()))
    recall = float(target_hit.sum() / max(1, target_boundary.sum()))
    f1 = 2.0 * precision * recall / max(1.0e-8, precision + recall)
    return precision, recall, f1


def room_confusion(
    prediction: torch.Tensor,
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    """Return a target-by-prediction room confusion matrix including non-room."""

    room_ids = _room_ids(taxonomy, prediction.device)
    room_lookup = torch.full(
        (taxonomy.num_classes,),
        0,
        device=prediction.device,
        dtype=torch.long,
    )
    room_lookup[room_ids] = torch.arange(
        1,
        len(room_ids) + 1,
        device=prediction.device,
    )
    predicted = room_lookup[prediction.long()]
    expected = room_lookup[target.long()]
    size = len(room_ids) + 1
    encoded = expected * size + predicted
    return torch.bincount(
        encoded.flatten(),
        minlength=size * size,
    ).reshape(size, size)


def decompose_rooms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    boundary_tolerance: int = 2,
) -> dict:
    """Compute one-case decomposition for mutually exclusive raster labels."""

    if prediction.ndim == 2:
        prediction = prediction[None, None]
    elif prediction.ndim == 3:
        prediction = prediction[None]
    if target.ndim == 2:
        target = target[None, None]
    elif target.ndim == 3:
        target = target[None]
    if footprint.ndim == 2:
        footprint = footprint[None, None]
    elif footprint.ndim == 3:
        footprint = footprint[None]
    if prediction.shape[0] != 1 or target.shape[0] != 1 or footprint.shape[0] != 1:
        raise ValueError("decompose_rooms currently expects one case")
    if prediction.shape[1] != 1 or target.shape[1] != 1 or footprint.shape[1] != 1:
        raise ValueError("decompose_rooms expects one raster channel per input")
    room_ids = _room_ids(taxonomy, prediction.device)
    rooms = room_ids.cpu().numpy()
    predicted_labels = prediction[0, 0].detach().cpu().numpy()
    target_labels = target[0, 0].detach().cpu().numpy()
    inside = footprint[0, 0].detach().cpu().numpy().astype(bool)
    prediction_room = np.isin(predicted_labels, rooms) & inside
    target_room = np.isin(target_labels, rooms) & inside

    intersection = prediction_room & target_room
    union = prediction_room | target_room
    occupancy = {
        "intersection": int(intersection.sum()),
        "false_negative": int((target_room & ~prediction_room).sum()),
        "false_positive": int((prediction_room & ~target_room).sum()),
        "union": int(union.sum()),
        "iou": float(intersection.sum() / max(1, union.sum())),
    }
    correct_identity = intersection & (predicted_labels == target_labels)
    semantic_mismatch = intersection & (predicted_labels != target_labels)
    total_union = max(1, occupancy["union"])
    allocation = {
        "correct_identity_fraction": float(correct_identity.sum() / total_union),
        "semantic_mismatch_fraction": float(semantic_mismatch.sum() / total_union),
        "missing_room_fraction": float(occupancy["false_negative"] / total_union),
        "spurious_room_fraction": float(occupancy["false_positive"] / total_union),
    }

    confusion = room_confusion(prediction, target, taxonomy).cpu().numpy()
    target_to_prediction: list[dict] = []
    class_rows: list[dict] = []
    instance_rows: list[dict] = []
    boundary_rows: list[dict] = []
    for class_index, class_id in enumerate(taxonomy.room_ids):
        name = taxonomy.class_names[class_id]
        target_mask = (target_labels == class_id) & inside
        predicted_mask = (predicted_labels == class_id) & inside
        tp = int((target_mask & predicted_mask).sum())
        target_mass = int(target_mask.sum())
        predicted_mass = int(predicted_mask.sum())
        union_mass = target_mass + predicted_mass - tp
        iou = float(tp / union_mass) if union_mass > 0 else 0.0
        row = confusion[class_index + 1]
        predicted_index = int(row.argmax()) if row.sum() > 0 else -1
        dominant_prediction = (
            "non-room"
            if predicted_index == 0
            else taxonomy.class_names[taxonomy.room_ids[predicted_index - 1]]
        )
        class_rows.append(
            {
                "class": name,
                "iou": iou,
                "target_pixels": target_mass,
                "predicted_pixels": predicted_mass,
                "true_positive": tp,
                "false_negative": target_mass - tp,
                "false_positive": predicted_mass - tp,
                "present_in_target": target_mass > 0,
                "predicted": predicted_mass > 0,
                "missing_prediction": target_mass > 0 and predicted_mass == 0,
                "spurious_prediction": target_mass == 0 and predicted_mass > 0,
                "dominant_prediction": dominant_prediction,
                "dominant_prediction_fraction": (
                    float(row.max() / row.sum()) if row.sum() > 0 else 0.0
                ),
                "mean_iou_deficit_share": 0.0,
            }
        )
        for predicted_index, count in enumerate(row):
            if count == 0:
                continue
            predicted_name = (
                "non-room"
                if predicted_index == 0
                else taxonomy.class_names[taxonomy.room_ids[predicted_index - 1]]
            )
            target_to_prediction.append(
                {
                    "target_class": name,
                    "predicted_class": predicted_name,
                    "pixels": int(count),
                    "target_fraction": float(count / max(1, row.sum())),
                }
            )

        target_components = _component_sets(target_mask)
        predicted_components = _component_sets(predicted_mask)
        target_hit = 0
        target_component_recalls = []
        for component in target_components:
            component_mass = max(1, int(component.sum()))
            overlap = int((component & predicted_mask).sum())
            recall = overlap / component_mass
            target_component_recalls.append(recall)
            target_hit += int(recall >= 0.5)
        instance_rows.append(
            {
                "class": name,
                "target_components": len(target_components),
                "predicted_components": len(predicted_components),
                "component_gap": abs(len(predicted_components) - len(target_components)),
                "target_components_hit_at_0_5": target_hit,
                "target_component_recall": (
                    float(np.mean(target_component_recalls))
                    if target_component_recalls
                    else 1.0
                ),
            }
        )
        precision, recall, f1 = _boundary_scores(
            predicted_mask,
            target_mask,
            tolerance=max(1, int(boundary_tolerance)),
        )
        boundary_rows.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    present_iou = np.asarray(
        [row["iou"] for row in class_rows if row["present_in_target"]],
        dtype=np.float64,
    )
    strict_macro_iou = float(present_iou.mean()) if len(present_iou) else 0.0
    deficits = np.maximum(0.0, 1.0 - present_iou)
    if deficits.sum() > 0.0:
        for row in class_rows:
            if row["present_in_target"]:
                row["mean_iou_deficit_share"] = float(
                    (1.0 - row["iou"]) / deficits.sum()
                )
    return {
        "strict_macro_iou": strict_macro_iou,
        "occupancy": occupancy,
        "allocation": allocation,
        "missing_class_count": sum(row["missing_prediction"] for row in class_rows),
        "spurious_class_count": sum(row["spurious_prediction"] for row in class_rows),
        "fragmentation_gap": sum(row["component_gap"] for row in instance_rows),
        "class_rows": class_rows,
        "confusion_rows": target_to_prediction,
        "instance_rows": instance_rows,
        "boundary_rows": boundary_rows,
    }


def aggregate_decomposition(cases: list[dict]) -> dict:
    """Aggregate case decompositions without selecting favorable cases."""

    if not cases:
        raise ValueError("at least one room decomposition is required")
    case_keys = (
        "strict_macro_iou",
        "fragmentation_gap",
        "missing_class_count",
        "spurious_class_count",
    )
    summary = {
        key: float(np.mean([float(case[key]) for case in cases]))
        for key in case_keys
    }
    occupancy_keys = ("intersection", "false_negative", "false_positive", "union")
    summary["occupancy"] = {
        key: int(sum(int(case["occupancy"][key]) for case in cases))
        for key in occupancy_keys
    }
    summary["occupancy"]["iou"] = float(
        summary["occupancy"]["intersection"]
        / max(1, summary["occupancy"]["union"])
    )
    allocation_totals = Counter()
    for case in cases:
        for key, value in case["allocation"].items():
            allocation_totals[key] += float(value) * int(case["occupancy"]["union"])
    total_union = max(1, summary["occupancy"]["union"])
    summary["allocation"] = {
        key: float(value / total_union)
        for key, value in sorted(allocation_totals.items())
    }

    class_names = sorted(
        {row["class"] for case in cases for row in case["class_rows"]}
    )
    class_rows = []
    for name in class_names:
        selected = [
            row
            for case in cases
            for row in case["class_rows"]
            if row["class"] == name and row["present_in_target"]
        ]
        if not selected:
            continue
        target_pixels = sum(row["target_pixels"] for row in selected)
        predicted_pixels = sum(row["predicted_pixels"] for row in selected)
        true_positive = sum(row["true_positive"] for row in selected)
        union = target_pixels + predicted_pixels - true_positive
        class_rows.append(
            {
                "class": name,
                "mean_case_iou": float(np.mean([row["iou"] for row in selected])),
                "dataset_iou": float(true_positive / max(1, union)),
                "target_pixels": int(target_pixels),
                "predicted_pixels": int(predicted_pixels),
                "true_positive": int(true_positive),
                "false_negative": int(target_pixels - true_positive),
                "false_positive": int(predicted_pixels - true_positive),
                "present_cases": len(selected),
                "zero_iou_cases": int(sum(row["iou"] == 0.0 for row in selected)),
            }
        )
    class_rows.sort(key=lambda row: row["mean_case_iou"])
    summary["class_rows"] = class_rows
    summary["dataset_class_macro_iou"] = float(
        np.mean([row["dataset_iou"] for row in class_rows])
    )

    confusion = Counter()
    for case in cases:
        for row in case["confusion_rows"]:
            confusion[(row["target_class"], row["predicted_class"])] += int(
                row["pixels"]
            )
    summary["confusion_rows"] = [
        {
            "target_class": target,
            "predicted_class": predicted,
            "pixels": pixels,
        }
        for (target, predicted), pixels in confusion.most_common()
    ]

    instance_names = sorted(
        {row["class"] for case in cases for row in case["instance_rows"]}
    )
    summary["instance_rows"] = []
    for name in instance_names:
        rows = [
            row
            for case in cases
            for row in case["instance_rows"]
            if row["class"] == name
        ]
        summary["instance_rows"].append(
            {
                "class": name,
                "mean_component_gap": float(
                    np.mean([row["component_gap"] for row in rows])
                ),
                "mean_target_component_recall": float(
                    np.mean([row["target_component_recall"] for row in rows])
                ),
            }
        )
    summary["instance_rows"].sort(
        key=lambda row: (row["mean_target_component_recall"], -row["mean_component_gap"])
    )

    boundary_names = sorted(
        {row["class"] for case in cases for row in case["boundary_rows"]}
    )
    summary["boundary_rows"] = []
    for name in boundary_names:
        rows = [
            row
            for case in cases
            for row in case["boundary_rows"]
            if row["class"] == name
        ]
        summary["boundary_rows"].append(
            {
                "class": name,
                "mean_precision": float(np.mean([row["precision"] for row in rows])),
                "mean_recall": float(np.mean([row["recall"] for row in rows])),
                "mean_f1": float(np.mean([row["f1"] for row in rows])),
            }
        )
    summary["boundary_rows"].sort(key=lambda row: row["mean_f1"])
    return summary
