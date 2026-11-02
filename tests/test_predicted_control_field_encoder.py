from __future__ import annotations

import torch

from floorflow.model.predicted_field_encoder import (
    PredictedControlFieldEncoder,
)


def test_global_evidence_uses_envelope_geometry() -> None:
    encoder = PredictedControlFieldEncoder(
        (2, 3, 2, 2),
        4,
        grid=8,
        hidden_channels=8,
        evidence_context=True,
    )
    with torch.no_grad():
        encoder.evidence_fusion[-1].weight.normal_(0.0, 0.05)
    controls = {
        "room_logits": torch.randn(1, 2, 16, 16),
        "structure": torch.randn(1, 3, 16, 16),
        "furniture": torch.randn(1, 2, 16, 16),
        "support": torch.randn(1, 2, 16, 16),
        "program": torch.randn(1, 4),
    }
    center = torch.zeros(1, 16, 16)
    center[:, 4:12, 4:12] = 1.0
    edge = torch.zeros_like(center)
    edge[:, 1:15, 1:15] = 1.0

    _features, center_evidence, tokens = encoder.encode_details(
        controls,
        center,
    )
    _features, edge_evidence, _tokens = encoder.encode_details(
        controls,
        edge,
    )

    assert center_evidence is not None
    assert edge_evidence is not None
    assert tokens.shape == (1, 6, 8)
    assert not torch.allclose(center_evidence, edge_evidence)
