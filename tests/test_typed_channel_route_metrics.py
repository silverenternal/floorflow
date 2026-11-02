from __future__ import annotations

import torch

from floorflow.training.typed_channel_route_metrics import (
    typed_channel_route_metrics,
)


def test_route_metrics_are_flattened_and_nonfinite_values_are_skipped() -> None:
    output = {
        "stage_walls_route_diagnostics": {
            "route_budget_soft_mean": torch.tensor(3.5),
            "routing_entropy": torch.tensor(0.75),
            "attention_entropy": torch.tensor(float("nan")),
            "effective_expert_fraction": torch.tensor(0.25),
            "route_gate_mean": torch.tensor(0.5),
        },
        "stage_openings_route_diagnostics": None,
        "action_route_diagnostics": {
            "walls": {
                "route_budget_soft_mean": torch.tensor(2.0),
                "routing_entropy": torch.tensor(0.5),
                "effective_expert_fraction": torch.tensor(0.5),
                "route_gate_mean": torch.tensor(0.4),
            }
        },
        "channel_cross_attention_gate": torch.tensor(0.6),
        "channel_source_attention_entropy": torch.tensor(0.8),
        "channel_source_attention_gate": torch.tensor(
            [[[0.2], [0.4]]]
        ),
        "reliability": torch.tensor([[[[0.6]]]]),
        "mask_class_evidence": torch.tensor([[[3.0, 4.0]]]),
        "stage_rooms_logits": torch.zeros(1, 2, 2, 2),
        "stage_rooms_candidate_logits": torch.ones(1, 2, 2, 2),
        "room_adjacency_logits": torch.zeros(1, 2, 2),
    }
    metrics = typed_channel_route_metrics(output)
    assert metrics["stage_walls_route_budget"] == 3.5
    assert metrics["stage_walls_route_routing_entropy"] == 0.75
    assert metrics["stage_walls_route_effective_expert_fraction"] == 0.25
    assert "stage_walls_route_attention_entropy" not in metrics
    assert not any(key.startswith("stage_openings") for key in metrics)
    assert metrics["action_walls_route_budget"] == 2.0
    assert metrics["action_walls_route_routing_entropy"] == 0.5
    assert abs(metrics["channel_cross_attention_gate_mean"] - 0.6) < 1.0e-6
    assert abs(metrics["channel_source_attention_entropy"] - 0.8) < 1.0e-6
    assert abs(
        metrics["channel_source_attention_gate_mean"] - 0.3
    ) < 1.0e-6
    assert abs(metrics["channel_reliability_mean"] - 0.6) < 1.0e-6
    assert metrics["room_class_mask_region_evidence_l2"] == 5.0
    assert metrics["room_stage_candidate_delta_abs_mean"] == 1.0
    assert metrics["room_adjacency_probability_mean"] == 0.5
