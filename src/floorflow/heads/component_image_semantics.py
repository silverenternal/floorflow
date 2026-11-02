"""Image-aware semantic features for frozen predicted room components."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label


def _shape_features(component: np.ndarray) -> np.ndarray:
    height, width = component.shape
    ys, xs = np.nonzero(component)
    area = float(ys.size)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    box_area = float(max(1, (y1 - y0) * (x1 - x0)))
    return np.asarray(
        [
            area / float(height * width),
            float(xs.mean()) / width,
            float(ys.mean()) / height,
            (x1 - x0) / width,
            (y1 - y0) / height,
            area / box_area,
            max(
                (x1 - x0) / max(1, y1 - y0),
                (y1 - y0) / max(1, x1 - x0),
            ),
            np.log1p(area) / np.log1p(height * width),
            np.sqrt(area) / max(1.0, np.sqrt(height * width)),
        ],
        dtype=np.float32,
    )


def _image_features(component: np.ndarray, image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)[:, component]
    means = values.mean(axis=1)
    standard_deviation = values.std(axis=1)
    quantiles = np.quantile(values, (0.1, 0.5, 0.9), axis=1).reshape(-1)
    return np.concatenate(
        [means, standard_deviation, quantiles]
    ).astype(np.float32)


def native_component_records(
    room_labels: np.ndarray,
    target_labels: np.ndarray,
    model_probability: np.ndarray,
    image: np.ndarray,
    footprint: np.ndarray,
    room_ids: tuple[int, ...],
    *,
    min_area: int = 8,
    structure_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect per-component semantic features and target majority classes."""

    room_ids_array = np.asarray(room_ids, dtype=np.int64)
    lookup = {
        int(class_id): index
        for index, class_id in enumerate(room_ids)
    }
    support = np.isin(room_labels, room_ids_array) & np.asarray(
        footprint,
        dtype=bool,
    )
    if structure_mask is not None:
        support &= ~np.asarray(structure_mask, dtype=bool)
    components, count = label(support, structure=np.ones((3, 3), dtype=bool))
    features = []
    targets = []
    component_ids = []
    for component_id in range(1, int(count) + 1):
        component = components == component_id
        if int(component.sum()) < int(min_area):
            continue
        target_values, target_counts = np.unique(
            target_labels[component],
            return_counts=True,
        )
        votes = [
            (int(target_counts[index]), lookup[int(class_id)])
            for index, class_id in enumerate(target_values)
            if int(class_id) in lookup
        ]
        if not votes:
            continue
        target_index = max(votes)[1]
        neighborhood = binary_dilation(
            component,
            structure=np.ones((3, 3), dtype=bool),
        ) & ~component
        model_local = model_probability[:, component]
        neighbor_local = model_probability[:, neighborhood]
        neighbor_mean = (
            neighbor_local.mean(axis=1)
            if neighbor_local.size
            else np.zeros(len(room_ids), dtype=np.float32)
        )
        vector = np.concatenate(
            [
                model_local.mean(axis=1),
                model_local.max(axis=1),
                neighbor_mean,
                _image_features(component, image),
                _shape_features(component),
            ]
        ).astype(np.float32)
        features.append(vector)
        targets.append(target_index)
        component_ids.append(component_id)
    feature_size = len(room_ids) * 3 + 15 + 9
    if not features:
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


__all__ = ["native_component_records"]
