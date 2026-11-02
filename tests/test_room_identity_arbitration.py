from __future__ import annotations

import torch

from floorflow.inference.room_identity_arbitration import (
    ROOM_IDENTITY_FEATURE_NAMES,
    arbitrate_room_logits,
    room_identity_arbitration_features,
)
from floorflow.recursion.trust_gate import RoomGraphTrustGate


def test_arbitration_features_are_target_free_and_finite() -> None:
    dense = torch.randn(2, 3, 16, 16)
    query = torch.randn(2, 3, 16, 16)
    routed = dense.clone()
    routed[:, 0, :, 8:] += 2.0
    footprint = torch.zeros(2, 16, 16)
    footprint[:, 2:14, 2:14] = 1.0
    features = room_identity_arbitration_features(
        dense,
        query,
        routed,
        footprint,
        room_ids=(1, 2, 3),
    )
    assert features.shape == (2, len(ROOM_IDENTITY_FEATURE_NAMES))
    assert torch.isfinite(features).all()


def test_arbitration_selects_dense_or_routed_by_gate_score() -> None:
    gate = RoomGraphTrustGate(
        feature_size=len(ROOM_IDENTITY_FEATURE_NAMES),
        hidden_size=8,
    )
    dense = torch.zeros(2, 3, 4, 4)
    routed = torch.ones(2, 3, 4, 4)
    features = torch.zeros(2, len(ROOM_IDENTITY_FEATURE_NAMES))
    with torch.no_grad():
        gate.network[-1].bias.fill_(0.0)
        gate.network[-1].weight.fill_(0.0)
    selected_logits, selected = arbitrate_room_logits(
        gate,
        dense,
        routed,
        features,
        threshold=0.5,
    )
    assert selected.tolist() == [True, True]
    assert torch.equal(selected_logits, routed)
