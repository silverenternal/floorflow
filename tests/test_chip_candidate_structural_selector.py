from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from floorflow.chip_flow.candidate_manifest import TARGET_FREE_FEATURE_NAMES
from floorflow.chip_flow.candidate_structural_selector import (
    SELECTOR_METRICS,
    StructuralSelectorCheckpoint,
    candidate_structural_features,
    load_structural_selector,
    save_structural_selector,
)


def _selector() -> StructuralSelectorCheckpoint:
    features = np.arange(4 * len(TARGET_FREE_FEATURE_NAMES), dtype=np.float64)
    features = features.reshape(4, len(TARGET_FREE_FEATURE_NAMES))
    design = candidate_structural_features(features)
    targets = np.arange(4 * len(SELECTOR_METRICS), dtype=np.float64).reshape(
        4,
        len(SELECTOR_METRICS),
    )
    regressor = LinearRegression().fit(design, targets)
    return StructuralSelectorCheckpoint(
        regressor=regressor,
        metric_mean=np.zeros(len(SELECTOR_METRICS)),
        metric_scale=np.ones(len(SELECTOR_METRICS)),
        metadata={"seed": 19},
    )


def test_structural_features_are_permutation_sensitive_and_target_free() -> None:
    features = np.stack(
        [
            np.arange(len(TARGET_FREE_FEATURE_NAMES)),
            np.arange(len(TARGET_FREE_FEATURE_NAMES))[::-1],
            np.ones(len(TARGET_FREE_FEATURE_NAMES)),
            np.full(len(TARGET_FREE_FEATURE_NAMES), 3.0),
        ]
    ).astype(np.float64)
    design = candidate_structural_features(features)
    assert design.shape == (4, 3 * len(TARGET_FREE_FEATURE_NAMES) + 4)
    delta_start = 2 * len(TARGET_FREE_FEATURE_NAMES)
    delta_end = 3 * len(TARGET_FREE_FEATURE_NAMES)
    assert np.all(design[0, delta_start:delta_end] == 0.0)
    assert np.all(
        design[1, delta_start:delta_end] != 0.0
    )


def test_structural_selector_checkpoint_round_trips(tmp_path) -> None:
    selector = _selector()
    path = tmp_path / "selector.pkl"
    save_structural_selector(path, selector)
    restored = load_structural_selector(path)
    features = np.zeros((4, len(TARGET_FREE_FEATURE_NAMES)), dtype=np.float64)
    prediction = restored.predict_metrics(features)
    assert prediction.shape == (4, len(SELECTOR_METRICS))
    assert 0 <= restored.select(features) < 4
    assert restored.metadata == {"seed": 19}


def test_structural_selector_rejects_wrong_feature_width() -> None:
    with pytest.raises(ValueError, match="feature width"):
        candidate_structural_features(np.zeros((4, 3), dtype=np.float64))
