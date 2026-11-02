"""Structured room/wall representations and set-level matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment

from floorflow.data.taxonomy import FloorplanTaxonomy


@dataclass(frozen=True)
class RoomInstance:
    class_id: int
    area: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray


def extract_room_instances(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    min_area: int = 8,
) -> list[RoomInstance]:
    instances: list[RoomInstance] = []
    for class_id in taxonomy.room_ids:
        components, count = label(labels == class_id)
        for component_id in range(1, count + 1):
            mask = components == component_id
            area = int(mask.sum())
            if area < min_area:
                continue
            ys, xs = np.nonzero(mask)
            bbox = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            instances.append(RoomInstance(class_id, area, bbox, mask))
    return instances


def instance_iou(left: RoomInstance, right: RoomInstance) -> float:
    intersection = np.logical_and(left.mask, right.mask).sum()
    union = np.logical_or(left.mask, right.mask).sum()
    return float(intersection) / max(1, int(union))


def hungarian_room_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    predicted = extract_room_instances(prediction, taxonomy)
    expected = extract_room_instances(target, taxonomy)
    if not predicted and not expected:
        return {"matched_iou": 1.0, "instance_precision": 1.0, "instance_recall": 1.0}
    if not predicted or not expected:
        return {"matched_iou": 0.0, "instance_precision": 0.0, "instance_recall": 0.0}
    scores = np.zeros((len(predicted), len(expected)), dtype=np.float64)
    for left_index, left in enumerate(predicted):
        for right_index, right in enumerate(expected):
            if left.class_id == right.class_id:
                scores[left_index, right_index] = instance_iou(left, right)
    pred_indices, target_indices = linear_sum_assignment(-scores)
    matched = scores[pred_indices, target_indices]
    return {
        "matched_iou": float(matched.mean()),
        "instance_precision": float((matched > 0.1).sum()) / len(predicted),
        "instance_recall": float((matched > 0.1).sum()) / len(expected),
    }


def room_area_distribution(labels: np.ndarray, taxonomy: FloorplanTaxonomy) -> np.ndarray:
    counts = np.asarray([(labels == class_id).sum() for class_id in taxonomy.room_ids], dtype=np.float64)
    return counts / max(1.0, counts.sum())


def room_class_adjacency(labels: np.ndarray, taxonomy: FloorplanTaxonomy) -> np.ndarray:
    """Normalized upper-triangle adjacency counts between room classes."""

    room_ids = np.asarray(taxonomy.room_ids)
    lookup = np.full(taxonomy.num_classes, -1, dtype=np.int64)
    lookup[room_ids] = np.arange(len(room_ids))
    encoded = lookup[np.asarray(labels)]
    counts = np.zeros((len(room_ids), len(room_ids)), dtype=np.float64)
    for left, right in (
        (encoded[:, :-1], encoded[:, 1:]),
        (encoded[:-1, :], encoded[1:, :]),
    ):
        valid = (left >= 0) & (right >= 0) & (left != right)
        np.add.at(counts, (left[valid], right[valid]), 1.0)
        np.add.at(counts, (right[valid], left[valid]), 1.0)
    counts /= max(1.0, counts.sum())
    rows, columns = np.triu_indices(len(room_ids), k=1)
    return counts[rows, columns]


def reference_distribution_distance(
    prediction: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> float:
    predicted = room_area_distribution(prediction, taxonomy)
    expected = room_area_distribution(target, taxonomy)
    return float(np.abs(predicted - expected).mean())


def contour_descriptor(footprint: np.ndarray, size: int = 16) -> np.ndarray:
    values = footprint.astype(np.float32)
    stride = max(1, values.shape[0] // size)
    coarse = values[::stride, ::stride][:size, :size]
    if coarse.shape != (size, size):
        coarse = np.pad(coarse, ((0, size - coarse.shape[0]), (0, size - coarse.shape[1])))
    return np.concatenate([coarse.ravel(), [values.mean(), values.sum() / values.size]])


def floorplan_feature_vector(labels: np.ndarray, taxonomy: FloorplanTaxonomy) -> np.ndarray:
    area = room_area_distribution(labels, taxonomy)
    counts = np.asarray([float((labels == class_id).sum() > 0) for class_id in taxonomy.room_ids])
    structure = np.asarray(
        [
            float((labels == taxonomy.class_id("wall")).mean()),
            float(np.isin(labels, taxonomy.opening_ids).mean()),
            float(np.isin(labels, taxonomy.furniture_ids).mean()),
            float(np.isin(labels, taxonomy.room_ids).mean()),
        ],
        dtype=np.float64,
    )
    instances = extract_room_instances(labels, taxonomy)
    instance_features = np.asarray([len(instances), np.mean([item.area for item in instances]) if instances else 0.0], dtype=np.float64)
    return np.concatenate(
        [
            area,
            counts,
            structure,
            instance_features,
            room_class_adjacency(labels, taxonomy),
        ]
    )


def fit_feature_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a robust diagonal scaler without allowing test statistics to leak."""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must have shape [samples, dimensions]")
    center = np.median(features, axis=0)
    scale = np.quantile(features, 0.75, axis=0) - np.quantile(features, 0.25, axis=0)
    scale = np.where(scale > 1.0e-8, scale, np.std(features, axis=0))
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return center, scale


def transform_features(
    features: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - center) / scale


def pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return np.sqrt(((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=-1))


def median_heuristic_bandwidth(real: np.ndarray, generated: np.ndarray) -> float:
    """Choose an RBF bandwidth from pooled pairwise distances.

    A fixed bandwidth of one on high-dimensional, unnormalised floorplan
    features makes every RBF kernel entry close to zero. The resulting MMD is
    almost constant and cannot distinguish model variants.
    """

    pooled = np.concatenate([np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)])
    distances = pairwise_distances(pooled, pooled)
    positive = distances[distances > 1.0e-12]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def mmd_rbf(
    real: np.ndarray,
    generated: np.ndarray,
    bandwidth: float | None = None,
) -> float:
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if len(real) == 0 or len(generated) == 0:
        return 0.0
    bandwidth = median_heuristic_bandwidth(real, generated) if bandwidth is None else float(bandwidth)
    bandwidth = max(bandwidth, 1.0e-8)

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distances = pairwise_distances(left, right) ** 2
        return np.exp(-distances / (2.0 * bandwidth * bandwidth))

    return float(kernel(real, real).mean() + kernel(generated, generated).mean() - 2.0 * kernel(real, generated).mean())


def energy_distance(real: np.ndarray, generated: np.ndarray) -> float:
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if len(real) == 0 or len(generated) == 0:
        return 0.0
    return float(
        2.0 * pairwise_distances(real, generated).mean()
        - pairwise_distances(real, real).mean()
        - pairwise_distances(generated, generated).mean()
    )


def coverage_density(
    real: np.ndarray,
    generated: np.ndarray,
    neighbors: int = 5,
) -> tuple[float, float]:
    """Return bounded precision/recall-style coverage and density.

    Coverage is the fraction of real modes whose nearest generated sample is
    inside the local real-manifold radius. Density is the symmetric fraction
    of generated samples inside the local generated-manifold radius. Both are
    probabilities in [0, 1], unlike raw nearest-neighbour distance sums.
    """

    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if len(real) == 0 or len(generated) == 0:
        return 0.0, 0.0
    neighbors = max(1, int(neighbors))
    real_neighbor = min(neighbors, len(real) - 1)
    generated_neighbor = min(neighbors, len(generated) - 1)
    real_to_generated = pairwise_distances(real, generated)
    generated_to_real = real_to_generated.T
    real_radii = np.sort(pairwise_distances(real, real), axis=1)[:, real_neighbor]
    generated_radii = np.sort(pairwise_distances(generated, generated), axis=1)[:, generated_neighbor]
    coverage = np.any(real_to_generated <= real_radii[:, None], axis=1).mean()
    density = np.any(generated_to_real <= generated_radii[:, None], axis=1).mean()
    return float(coverage), float(density)
