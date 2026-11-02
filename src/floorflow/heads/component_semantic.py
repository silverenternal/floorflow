"""Component-level room semantic classification utilities."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label
from torch import nn

SHAPE_FEATURES = 9


def component_feature_vector(
    component: np.ndarray,
    model_probability: np.ndarray,
    retrieval_probability: np.ndarray,
) -> np.ndarray:
    pixels = np.flatnonzero(component)
    if pixels.size == 0:
        raise ValueError("component must contain pixels")
    height, width = component.shape
    ys, xs = np.unravel_index(pixels, component.shape)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    area = float(pixels.size)
    box_area = float(max(1, (y1 - y0) * (x1 - x0)))
    model_values = model_probability[:, component]
    retrieval_values = retrieval_probability[:, component]
    shape = np.asarray(
        [
            area / float(height * width),
            float(xs.mean()) / width,
            float(ys.mean()) / height,
            (x1 - x0) / width,
            (y1 - y0) / height,
            area / box_area,
            max((x1 - x0) / max(1, y1 - y0), (y1 - y0) / max(1, x1 - x0)),
            np.log1p(area) / np.log1p(height * width),
            np.sqrt(area) / max(1.0, np.sqrt(height * width)),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            model_values.mean(axis=1),
            model_values.max(axis=1),
            retrieval_values.mean(axis=1),
            shape,
        ]
    ).astype(np.float32)


def component_records(
    room_labels: np.ndarray,
    target_labels: np.ndarray,
    model_probability: np.ndarray,
    retrieval_probability: np.ndarray,
    footprint: np.ndarray,
    room_ids: tuple[int, ...],
    *,
    min_area: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    room_id_array = np.asarray(room_ids, dtype=np.int64)
    support = np.isin(room_labels, room_id_array) & (footprint > 0)
    components, count = label(support, structure=np.ones((3, 3), dtype=bool))
    features = []
    targets = []
    component_ids = []
    lookup = {int(class_id): index for index, class_id in enumerate(room_ids)}
    for component_id in range(1, int(count) + 1):
        component = components == component_id
        area = int(component.sum())
        if area < int(min_area):
            continue
        target_values, target_counts = np.unique(
            target_labels[component],
            return_counts=True,
        )
        votes = [
            (int(target_counts[i]), lookup[int(class_id)])
            for i, class_id in enumerate(target_values)
            if int(class_id) in lookup
        ]
        if not votes:
            continue
        target_index = max(votes)[1]
        features.append(
            component_feature_vector(
                component,
                model_probability,
                retrieval_probability,
            )
        )
        targets.append(target_index)
        component_ids.append(component_id)
    if not features:
        feature_size = model_probability.shape[0] * 3 + SHAPE_FEATURES
        return (
            np.zeros((0, feature_size), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    return (
        np.stack(features),
        np.asarray(targets, dtype=np.int64),
        np.asarray(component_ids, dtype=np.int64),
    )


def assign_component_classes(
    room_labels: np.ndarray,
    class_probability: np.ndarray,
    component_ids: np.ndarray,
    room_ids: tuple[int, ...],
    *,
    min_area: int = 8,
    structure_mask: np.ndarray | None = None,
) -> np.ndarray:
    room_id_array = np.asarray(room_ids, dtype=np.int64)
    support = np.isin(room_labels, room_id_array)
    if structure_mask is not None:
        support &= ~np.asarray(structure_mask, dtype=bool)
    components, _count = label(
        support,
        structure=np.ones((3, 3), dtype=bool),
    )
    output = room_labels.copy()
    if structure_mask is not None:
        output[np.asarray(structure_mask, dtype=bool)] = 0
    for row, component_id in enumerate(component_ids.tolist()):
        mask = components == int(component_id)
        if int(mask.sum()) < int(min_area):
            continue
        output[mask] = int(room_id_array[int(class_probability[row].argmax())])
    return output


class ComponentSemanticClassifier(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


__all__ = [
    "ComponentSemanticClassifier",
    "assign_component_classes",
    "component_feature_vector",
    "component_records",
]
