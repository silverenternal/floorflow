import numpy as np
import pytest
import torch

from floorflow.representation.core import coverage_density, fit_feature_standardizer, mmd_rbf, transform_features
from floorflow.inference.sampling_policy import SamplingPolicy, best_of_n_curve, expected_sample_statistics
from floorflow.objectives.semantic_losses import room_group_iou, room_group_loss
from floorflow.data.taxonomy import FloorplanTaxonomy
from collections import Counter
from floorflow.data.taxonomy import build_taxonomy


def test_single_sample_policy_rejects_implicit_selection() -> None:
    with pytest.raises(ValueError):
        SamplingPolicy(purpose="single_sample", candidates_per_condition=4)
    with pytest.raises(ValueError):
        SamplingPolicy(purpose="distribution", candidates_per_condition=1)


def test_expected_statistics_are_not_best_of_n() -> None:
    values = np.asarray([[0.1, 0.2, 0.9], [0.2, 0.3, 0.4]])
    statistics = expected_sample_statistics(values)
    assert statistics["mean"] == pytest.approx(0.35)
    assert statistics["median"] == pytest.approx(0.25)
    curve = best_of_n_curve(values, (1, 2, 4))
    assert curve == pytest.approx({1: 0.15, 2: 0.25, 3: 0.65})


def test_distribution_metrics_are_bounded_and_sensitive() -> None:
    rng = np.random.default_rng(3)
    real = rng.normal(size=(24, 6))
    generated = rng.normal(size=(16, 6))
    coverage, density = coverage_density(real, generated)
    assert 0.0 <= coverage <= 1.0
    assert 0.0 <= density <= 1.0
    assert mmd_rbf(real, generated) >= 0.0
    repeated = np.repeat(real[:1], 16, axis=0)
    assert coverage_density(real, repeated)[0] <= coverage


def test_feature_scaling_uses_fit_statistics_only() -> None:
    train = np.asarray([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    center, scale = fit_feature_standardizer(train)
    transformed = transform_features(np.asarray([[1.0, 15.0]]), center, scale)
    assert transformed.shape == (1, 2)
    assert np.isfinite(transformed).all()


def test_room_group_loss_only_uses_room_labels() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Bath": 10}),
            "furniture": Counter({"Toilet": 10}),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    bedroom = taxonomy.class_id("room:Bedroom")
    target[:, 2:6, 2:6] = bedroom
    logits = torch.randn(1, taxonomy.num_target_channels, 8, 8)
    loss, ce, dice = room_group_loss(logits, target, taxonomy)
    assert torch.isfinite(loss)
    assert ce >= 0 and dice >= 0
    assert 0.0 <= room_group_iou(logits, target, taxonomy) <= 1.0
