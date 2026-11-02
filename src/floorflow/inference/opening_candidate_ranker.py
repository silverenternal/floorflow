"""Deterministic candidate features and ranking for wall-interval openings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.opening_readout import OpeningDetection
from floorflow.geometry.wall_graph_readout import RasterWallPlan, ReadoutWall

FEATURE_NAMES = (
    "detector_score",
    "foreground_mean",
    "foreground_peak",
    "foreground_std",
    "door_score_mean",
    "window_score_mean",
    "family_margin_mean",
    "width_px",
    "width_frame_fraction",
    "width_wall_fraction",
    "wall_length_fraction",
    "center_fraction",
    "center_x_fraction",
    "center_y_fraction",
    "orientation_cos",
    "orientation_sin",
    "exterior_wall",
    "same_wall_candidate_count",
    "same_wall_score_rank",
    "previous_center_gap_fraction",
    "next_center_gap_fraction",
)

HOST_FEATURE_NAMES = (
    "host_wall_continuity",
    "host_wall_visibility_gap",
    "host_thickness_fraction",
    "host_endpoint_margin_fraction",
    "same_wall_candidate_coverage",
    "host_wall_endpoint_distance_fraction",
    "collinear_neighbor_fraction",
    "collinear_length_fraction",
    "parallel_wall_fraction",
)

FEATURE_NAMES_V3 = FEATURE_NAMES + HOST_FEATURE_NAMES


def _sample_point(
    x: float,
    y: float,
    values: np.ndarray,
) -> np.ndarray:
    height, width = values.shape[-2:]
    column = min(max(round(float(x)), 0), width - 1)
    row = min(max(round(float(y)), 0), height - 1)
    return np.asarray(values[..., row, column], dtype=np.float32)


def _wall_fraction(
    center: tuple[float, float],
    wall: ReadoutWall,
) -> float:
    start = np.asarray(wall.start, dtype=np.float64)
    end = np.asarray(wall.end, dtype=np.float64)
    delta = end - start
    denominator = float(delta @ delta)
    if denominator <= 1.0e-9:
        return 0.0
    return float(
        np.clip((np.asarray(center) - start) @ delta / denominator, 0.0, 1.0)
    )


def opening_candidate_features(
    detection: OpeningDetection,
    foreground: np.ndarray,
    family_logits: np.ndarray,
    plan: RasterWallPlan,
    wall_detections: list[OpeningDetection],
    *,
    frame: float,
) -> np.ndarray:
    """Build target-free features for one detected wall interval."""

    if detection.wall_index is None:
        raise ValueError("opening candidate has no wall host")
    wall_index = int(detection.wall_index)
    wall = plan.walls[wall_index]
    start = np.asarray(wall.start, dtype=np.float64)
    end = np.asarray(wall.end, dtype=np.float64)
    delta = end - start
    wall_length = float(np.linalg.norm(delta))
    center_fraction = _wall_fraction(detection.center, wall)
    half_span = 0.5 * float(detection.width) / max(1.0, wall_length)
    fractions = np.linspace(
        max(0.0, center_fraction - half_span),
        min(1.0, center_fraction + half_span),
        9,
    )
    foreground_values = []
    family_values = []
    for fraction in fractions:
        point = start + float(fraction) * delta
        foreground_values.append(
            float(_sample_point(point[0], point[1], foreground))
        )
        family_values.append(
            _sample_point(point[0], point[1], family_logits)
        )
    foreground_array = np.asarray(foreground_values, dtype=np.float32)
    family_array = np.asarray(family_values, dtype=np.float32)
    family_mean = family_array.mean(axis=0)
    same_wall = sorted(
        [
            item
            for item in wall_detections
            if item.wall_index == wall_index
        ],
        key=lambda item: (-item.score, item.center),
    )
    rank = next(
        index
        for index, item in enumerate(same_wall)
        if item is detection
    )
    ordered_fractions = sorted(
        (
            _wall_fraction(item.center, wall),
            index,
        )
        for index, item in enumerate(same_wall)
    )
    position = next(
        index
        for index, (_fraction, candidate_index) in enumerate(
            ordered_fractions
        )
        if candidate_index == rank
    )
    previous_gap = (
        center_fraction - ordered_fractions[position - 1][0]
        if position > 0
        else 1.0
    )
    next_gap = (
        ordered_fractions[position + 1][0] - center_fraction
        if position + 1 < len(ordered_fractions)
        else 1.0
    )
    norm = max(1.0, wall_length)
    orientation = delta / norm
    return np.asarray(
        [
            detection.score,
            float(foreground_array.mean()),
            float(foreground_array.max()),
            float(foreground_array.std()),
            float(family_mean[0]),
            float(family_mean[1]),
            float(abs(family_mean[0] - family_mean[1])),
            float(detection.width),
            float(detection.width) / max(1.0, float(frame)),
            float(detection.width) / norm,
            wall_length / max(1.0, float(frame)),
            center_fraction,
            float(detection.center[0]) / max(1.0, float(frame)),
            float(detection.center[1]) / max(1.0, float(frame)),
            float(orientation[0]),
            float(orientation[1]),
            float(wall.wall_type == "exterior"),
            float(len(same_wall)),
            float(rank) / max(1, len(same_wall)),
            float(previous_gap),
            float(next_gap),
        ],
        dtype=np.float32,
    )


def _host_features(
    detection: OpeningDetection,
    plan: RasterWallPlan,
    wall_detections: list[OpeningDetection],
    *,
    frame: float,
) -> np.ndarray:
    wall_index = int(detection.wall_index)
    wall = plan.walls[wall_index]
    start = np.asarray(wall.start, dtype=np.float64)
    end = np.asarray(wall.end, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1.0e-6:
        return np.zeros(len(HOST_FEATURE_NAMES), dtype=np.float32)
    direction = delta / length
    center_fraction = _wall_fraction(detection.center, wall)
    repeated = [
        item for item in wall_detections if item.wall_index == wall_index
    ]
    coverage = sum(item.width for item in repeated) / max(1.0, length)
    endpoint_distance = min(
        center_fraction,
        1.0 - center_fraction,
    ) * length / max(1.0, float(frame))
    line = np.asarray([wall.start, wall.end], dtype=np.float32)
    neighbor_count = 0
    neighbor_length = 0.0
    parallel_count = 0
    tolerance = max(2.0, float(frame) * 0.01)
    for index, other in enumerate(plan.walls):
        if index == wall_index or other.wall_type != wall.wall_type:
            continue
        other_delta = np.asarray(other.end) - np.asarray(other.start)
        other_length = float(np.linalg.norm(other_delta))
        if other_length <= 1.0e-6:
            continue
        parallelism = abs(float(direction @ (other_delta / other_length)))
        if parallelism < 0.95:
            continue
        parallel_count += 1
        other_line = np.asarray(
            [other.start, other.end],
            dtype=np.float32,
        )
        distance = _line_distance(line, other_line)
        if distance <= tolerance:
            neighbor_count += 1
            neighbor_length += other_length
    return np.asarray(
        [
            float(wall.continuity),
            float(1.0 - wall.continuity),
            float(wall.thickness) / max(1.0, float(frame)),
            float(min(center_fraction, 1.0 - center_fraction)),
            float(coverage),
            float(endpoint_distance),
            float(neighbor_count) / max(1, len(plan.walls)),
            float(neighbor_length) / max(1.0, float(frame)),
            float(parallel_count) / max(1, len(plan.walls)),
        ],
        dtype=np.float32,
    )


def _line_distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = left[1] - left[0]
    normal = np.asarray([-delta[1], delta[0]], dtype=np.float32)
    normal = normal / max(float(np.linalg.norm(normal)), 1.0e-6)
    return float(
        min(
            abs(float((right[0] - left[0]) @ normal)),
            abs(float((right[1] - left[0]) @ normal)),
        )
    )


def opening_candidate_features_v3(
    detection: OpeningDetection,
    foreground: np.ndarray,
    family_logits: np.ndarray,
    plan: RasterWallPlan,
    wall_detections: list[OpeningDetection],
    *,
    frame: float,
) -> np.ndarray:
    """Append predicted-wall host coverage and fragmentation features."""

    base = opening_candidate_features(
        detection,
        foreground,
        family_logits,
        plan,
        wall_detections,
        frame=frame,
    )
    return np.concatenate(
        [
            base,
            _host_features(
                detection,
                plan,
                wall_detections,
                frame=frame,
            ),
        ]
    )


def _class_id_for_family(
    family: str,
    taxonomy: FloorplanTaxonomy,
) -> int:
    prefix = "window:" if family == "window" else "door:"
    return next(
        class_id
        for class_id in taxonomy.opening_ids
        if taxonomy.class_names[class_id].startswith(prefix)
    )


@dataclass(frozen=True)
class OpeningCandidateRanker:
    """Loaded presence and family models for interval candidates."""

    presence_model: Any
    family_model: Any
    presence_threshold: float
    feature_names: tuple[str, ...] = FEATURE_NAMES
    family_mode: str = "model"
    torch_ranker: Any | None = None

    def _predict(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.torch_ranker is not None:
            return self.torch_ranker.probabilities(matrix)
        return (
            self.presence_model.predict_proba(matrix)[:, 1],
            self.family_model.predict_proba(matrix),
        )

    def _materialize(
        self,
        candidates: list[OpeningDetection],
        presence: np.ndarray,
        family: np.ndarray,
        taxonomy: FloorplanTaxonomy,
    ) -> list[OpeningDetection]:
        rows = []
        for detection, probability, family_score in zip(
            candidates,
            presence,
            family,
            strict=True,
        ):
            if float(probability) < float(self.presence_threshold):
                continue
            kind = (
                "window" if int(np.argmax(family_score)) == 1 else "door"
            ) if self.family_mode == "model" else detection.family
            rows.append(
                replace(
                    detection,
                    family=kind,
                    class_id=_class_id_for_family(kind, taxonomy),
                )
            )
        return rows

    def rank_features(
        self,
        detections: list[OpeningDetection],
        features: np.ndarray,
        taxonomy: FloorplanTaxonomy,
    ) -> list[OpeningDetection]:
        if self.family_mode not in {"model", "detector"}:
            raise ValueError("family_mode must be model or detector")
        candidates = [
            detection
            for detection in detections
            if detection.wall_index is not None
        ]
        if not candidates:
            return []
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.shape != (len(candidates), len(self.feature_names)):
            raise ValueError("opening candidate feature matrix is invalid")
        presence, family = self._predict(matrix)
        return self._materialize(candidates, presence, family, taxonomy)

    def rank_feature_groups(
        self,
        detections: list[list[OpeningDetection]],
        feature_groups: list[np.ndarray],
        taxonomy: FloorplanTaxonomy,
    ) -> list[list[OpeningDetection]]:
        """Batch all case-level candidates into one classifier invocation."""

        candidates = [
            [
                detection
                for detection in group
                if detection.wall_index is not None
            ]
            for group in detections
        ]
        sizes = np.asarray([len(group) for group in candidates], dtype=np.int64)
        if int(sizes.sum()) == 0:
            return [[] for _ in candidates]
        matrices = [
            np.asarray(features, dtype=np.float32)
            for features in feature_groups
        ]
        for group, matrix in zip(candidates, matrices, strict=True):
            if matrix.shape != (len(group), len(self.feature_names)):
                raise ValueError("opening candidate feature group is invalid")
        combined = np.concatenate(matrices)
        presence, family = self._predict(combined)
        presence_parts = np.split(presence, np.cumsum(sizes)[:-1])
        family_parts = np.split(family, np.cumsum(sizes)[:-1])
        return [
            self._materialize(group, case_presence, case_family, taxonomy)
            for group, case_presence, case_family in zip(
                candidates,
                presence_parts,
                family_parts,
                strict=True,
            )
        ]

    def rank(
        self,
        detections: list[OpeningDetection],
        foreground: np.ndarray,
        family_logits: np.ndarray,
        plan: RasterWallPlan,
        taxonomy: FloorplanTaxonomy,
        *,
        frame: float,
    ) -> list[OpeningDetection]:
        candidates = [
            detection
            for detection in detections
            if detection.wall_index is not None
        ]
        features = [
            opening_candidate_features(
                detection,
                foreground,
                family_logits,
                plan,
                detections,
                frame=frame,
            )
            for detection in candidates
        ]
        return self.rank_features(
            candidates,
            (
                np.stack(features)
                if features
                else np.zeros((0, len(self.feature_names)), dtype=np.float32)
            ),
            taxonomy,
        )


def load_opening_candidate_ranker(
    path: str | Path,
    device: Any = "cpu",
) -> OpeningCandidateRanker:
    source = Path(path)
    if source.suffix == ".pt":
        from floorflow.inference.opening_candidate_torch import (
            load_opening_candidate_torch_ranker,
        )

        torch_ranker = load_opening_candidate_torch_ranker(source, device)
        feature_count = next(torch_ranker.model.parameters()).shape[-1]
        feature_names = (
            FEATURE_NAMES_V3
            if feature_count == len(FEATURE_NAMES_V3)
            else FEATURE_NAMES
        )
        return OpeningCandidateRanker(
            presence_model=None,
            family_model=None,
            presence_threshold=torch_ranker.presence_threshold,
            family_mode=torch_ranker.family_mode,
            torch_ranker=torch_ranker,
            feature_names=feature_names,
        )
    payload = joblib.load(path)
    feature_names = tuple(payload["feature_names"])
    if feature_names not in {FEATURE_NAMES, FEATURE_NAMES_V3}:
        raise ValueError("opening ranker feature contract mismatch")
    return OpeningCandidateRanker(
        presence_model=payload["presence_model"],
        family_model=payload["family_model"],
        presence_threshold=float(payload["presence_threshold"]),
        family_mode=str(payload.get("family_mode", "model")),
        feature_names=feature_names,
    )


def save_opening_candidate_ranker(
    path: str | Path,
    ranker: OpeningCandidateRanker,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".pt":
        if ranker.torch_ranker is None:
            raise ValueError("PT ranker output requires a torch backend")
        from floorflow.inference.opening_candidate_torch import (
            save_opening_candidate_torch_ranker,
        )

        save_opening_candidate_torch_ranker(
            destination,
            replace(
                ranker.torch_ranker,
                presence_threshold=float(ranker.presence_threshold),
            ),
        )
        return
    joblib.dump(
        {
            "feature_names": ranker.feature_names,
            "presence_model": ranker.presence_model,
            "family_model": ranker.family_model,
            "presence_threshold": float(ranker.presence_threshold),
            "family_mode": ranker.family_mode,
        },
        destination,
    )


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_NAMES_V3",
    "HOST_FEATURE_NAMES",
    "OpeningCandidateRanker",
    "load_opening_candidate_ranker",
    "opening_candidate_features",
    "opening_candidate_features_v3",
    "save_opening_candidate_ranker",
]
