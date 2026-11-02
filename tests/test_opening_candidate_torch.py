from __future__ import annotations

import numpy as np
import torch

from floorflow.inference.opening_candidate_torch import (
    load_opening_candidate_torch_ranker,
    save_opening_candidate_torch_ranker,
    train_opening_candidate_mlp,
)


def test_torch_candidate_ranker_trains_predicts_and_round_trips(tmp_path) -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(128, 21)).astype(np.float32)
    presence = (features[:, 0] > -0.2).astype(np.int64)
    family = np.where(features[:, 1] > 0.0, 1, 0).astype(np.int64)
    ranker = train_opening_candidate_mlp(
        features,
        presence,
        family,
        device="cpu",
        epochs=20,
    )
    probability, family_probability = ranker.probabilities(features[:16])
    assert probability.shape == (16,)
    assert family_probability.shape == (16, 2)
    assert np.isfinite(probability).all()
    assert np.allclose(family_probability.sum(axis=1), 1.0, atol=1.0e-5)

    path = tmp_path / "ranker.pt"
    save_opening_candidate_torch_ranker(path, ranker)
    loaded = load_opening_candidate_torch_ranker(path, "cpu")
    expected, expected_family = ranker.probabilities(features[:16])
    actual, actual_family = loaded.probabilities(features[:16])
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual_family, expected_family)


def test_torch_candidate_ranker_initializes_normalization_tensors() -> None:
    ranker = train_opening_candidate_mlp(
        np.zeros((8, 21), dtype=np.float32),
        np.asarray([0, 1, 0, 1, 0, 1, 0, 1]),
        np.asarray([0, 1, 0, 1, 0, 1, 0, 1]),
        device="cpu",
        epochs=1,
    )
    assert next(ranker.model.parameters()).device.type == "cpu"
    assert isinstance(ranker.feature_mean, torch.Tensor)

