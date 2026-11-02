from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.recursion.complexity import (
    COMPLEXITY_FEATURE_NAMES,
    POLICY_INPUT_FEATURE_NAMES,
    PolicyLabelCalibration,
    RecursionPolicy,
    complexity_features,
    complexity_score,
    derive_policy_labels,
    policy_loss,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10, "Bath": 10}),
            "furniture": Counter({"Bed": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_complexity_features_and_quantile_policy_labels() -> None:
    taxonomy = _taxonomy()
    footprint = torch.zeros((3, 32, 32))
    target = torch.zeros((3, 32, 32), dtype=torch.long)
    footprint[0, 2:30, 2:30] = 1
    target[0, 2:30, 2:30] = taxonomy.class_id("room:Bedroom")
    footprint[1, 2:30, 2:30] = 1
    target[1, 2:15, 2:30] = taxonomy.class_id("room:Bedroom")
    target[1, 15:30, 2:16] = taxonomy.class_id("room:Kitchen")
    target[1, 15:30, 16:30] = taxonomy.class_id("room:Bath")
    footprint[2] = torch.roll(torch.ones((4, 4)), 0, 0).repeat_interleave(8, 0).repeat_interleave(8, 1)
    target[2, 0:8, 0:8] = taxonomy.class_id("room:Bedroom")
    target[2, 8:16, 8:16] = taxonomy.class_id("room:Kitchen")
    target[2, 20:24, 20:24] = taxonomy.class_id("room:Bath")
    features = complexity_features(footprint, target, taxonomy)
    assert features.shape == (3, len(COMPLEXITY_FEATURE_NAMES))
    score = complexity_score(features)
    assert torch.isfinite(score).all()
    calibration = PolicyLabelCalibration.fit(features)
    labels = derive_policy_labels(features, calibration)
    assert labels.tolist() == [0, 1, 2]


def test_recursion_policy_returns_actions_and_trains() -> None:
    features = torch.randn(12, len(POLICY_INPUT_FEATURE_NAMES))
    labels = torch.arange(12) % 3
    policy = RecursionPolicy()
    policy.set_standardizer(features.mean(0), features.std(0))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1.0e-2)
    initial = float(policy_loss(policy(features), labels).detach())
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = policy_loss(policy(features), labels)
        loss.backward()
        optimizer.step()
    final = float(policy_loss(policy(features), labels).detach())
    decision = policy.decide(features)
    assert final < initial
    assert decision["action"].shape == (12,)
    assert decision["recursion_strength"].min() >= 0.0
    assert decision["recursion_strength"].max() <= 1.0
