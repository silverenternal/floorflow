"""Target-free candidate selection for legalized structure quality."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from floorflow.chip_flow.candidate_manifest import TARGET_FREE_FEATURE_NAMES

SELECTOR_METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_rho_mae",
    "legalizer_rung",
    "legalizer_solves",
)

DEFAULT_UTILITY_WEIGHTS = np.asarray(
    (1.0, 1.0, 0.0, 0.0, 0.1, 0.5, 0.1, 4.0, 0.0, 0.0),
    dtype=np.float64,
)


def candidate_structural_features(features: np.ndarray) -> np.ndarray:
    """Append within-case ranks, reference deltas and view identity."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("candidate features must have shape (candidate, feature)")
    if values.shape[1] != len(TARGET_FREE_FEATURE_NAMES):
        raise ValueError("candidate feature width does not match the target-free contract")
    order = np.argsort(values, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    positions = np.arange(len(values), dtype=np.float64)[:, None]
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(positions, order.shape),
        axis=0,
    )
    ranks /= max(len(values) - 1, 1)
    reference_delta = values - values[0]
    identity = np.eye(len(values), dtype=np.float64)
    return np.concatenate((values, ranks, reference_delta, identity), axis=1)


@dataclass
class StructuralSelectorCheckpoint:
    """Serializable regressor plus the frozen target-free scoring contract."""

    regressor: Any
    metric_mean: np.ndarray
    metric_scale: np.ndarray
    utility_weights: np.ndarray = field(
        default_factory=lambda: DEFAULT_UTILITY_WEIGHTS.copy()
    )
    cost_tiebreak: float = 0.25
    metadata: dict | None = None

    def validate(self) -> None:
        width = len(SELECTOR_METRICS)
        self.metric_mean = np.asarray(self.metric_mean, dtype=np.float64)
        self.metric_scale = np.asarray(self.metric_scale, dtype=np.float64)
        self.utility_weights = np.asarray(self.utility_weights, dtype=np.float64)
        if self.metric_mean.shape != (width,) or self.metric_scale.shape != (width,):
            raise ValueError("selector metric normalization shape mismatch")
        if self.utility_weights.shape != (width,):
            raise ValueError("selector utility weight shape mismatch")
        if not np.all(np.isfinite(self.metric_scale)) or np.any(self.metric_scale <= 0):
            raise ValueError("selector metric scales must be positive and finite")
        if not np.isfinite(self.cost_tiebreak) or self.cost_tiebreak < 0:
            raise ValueError("selector cost tiebreak must be finite and non-negative")

    def predict_metrics(self, features: np.ndarray) -> np.ndarray:
        self.validate()
        normalized = np.asarray(self.regressor.predict(candidate_structural_features(features)))
        expected = (len(features), len(SELECTOR_METRICS))
        if normalized.shape != expected:
            raise ValueError(f"selector prediction shape {normalized.shape} != {expected}")
        return normalized * self.metric_scale + self.metric_mean

    def select(self, features: np.ndarray) -> int:
        predictions = self.predict_metrics(features)
        utility = predictions @ self.utility_weights
        utility += self.cost_tiebreak * predictions[:, 0]
        return int(np.argmin(utility))


def save_structural_selector(
    path: Path,
    selector: StructuralSelectorCheckpoint,
) -> None:
    selector.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(pickle.dumps(selector, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)


def load_structural_selector(path: Path) -> StructuralSelectorCheckpoint:
    selector = pickle.loads(path.read_bytes())
    if not isinstance(selector, StructuralSelectorCheckpoint):
        raise TypeError("checkpoint does not contain a structural selector")
    selector.validate()
    return selector


__all__ = [
    "DEFAULT_UTILITY_WEIGHTS",
    "SELECTOR_METRICS",
    "StructuralSelectorCheckpoint",
    "candidate_structural_features",
    "load_structural_selector",
    "save_structural_selector",
]
