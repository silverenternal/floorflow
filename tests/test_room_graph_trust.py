from __future__ import annotations

import torch

from floorflow.recursion.trust_gate import (
    TRUST_FEATURE_NAMES,
    RoomGraphTrustGate,
    room_logit_trust_features,
)


def test_room_graph_trust_gate_features_and_scores() -> None:
    base = torch.zeros((2, 3, 8, 8))
    graph = torch.zeros_like(base)
    graph[:, 1] = 2.0
    footprint = torch.ones((2, 1, 8, 8))
    features = room_logit_trust_features(base, graph, footprint)
    assert features.shape == (2, len(TRUST_FEATURE_NAMES))
    assert torch.isfinite(features).all()
    gate = RoomGraphTrustGate()
    logits = gate(features)
    trust = gate.trust(features)
    assert logits.shape == (2,)
    assert ((trust >= 0.0) & (trust <= 1.0)).all()
