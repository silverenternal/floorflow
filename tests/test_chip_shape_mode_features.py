from __future__ import annotations

import torch

from floorflow.chip_flow.shape_mode_features import (
    SHAPE_MODE_FEATURE_NAMES,
    select_pareto_safe_shape_mode,
    shape_mode_geometry_features,
)


def test_shape_mode_geometry_features_cover_candidates() -> None:
    batch = {
        "areas": torch.ones(1, 3),
        "constraints": torch.zeros(1, 3, 5),
        "b2b": torch.tensor([[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]]),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0], [1.0, 2.0, 0.5]]]),
        "pins": torch.tensor([[[0.2, 0.3], [1.8, 0.4]]]),
        "n_b2b": torch.tensor([2]),
        "n_p2b": torch.tensor([2]),
        "n_pins": torch.tensor([2]),
    }
    baseline = torch.tensor(
        [[[0.2, 0.2, 0.0], [0.6, 0.2, 0.0], [1.0, 0.2, 0.0]]]
    )
    modes = torch.stack([baseline, baseline], dim=1)
    modes[0, 1, :, 2] = 0.4

    features = shape_mode_geometry_features(modes, baseline, batch)

    assert features.shape == (
        1,
        2,
        len(SHAPE_MODE_FEATURE_NAMES),
    )
    assert torch.isfinite(features).all()
    assert not torch.allclose(features[0, 0], features[0, 1])


def test_pareto_selector_keeps_baseline_or_safe_improvement() -> None:
    baseline = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    regression = torch.tensor([1.02, 1.0, 0.8, 1.0, 1.0, 0.0, 0.0, 0.1])
    improvement = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.1])
    features = torch.stack([baseline, regression, improvement])[None]

    selected = select_pareto_safe_shape_mode(features)
    baseline_only = select_pareto_safe_shape_mode(features[:, :2])

    assert selected.item() == 2
    assert baseline_only.item() == 0
