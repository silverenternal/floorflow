from types import SimpleNamespace

import torch

from floorflow.training.raster_instance_evaluation import (
    blended_probabilities,
    identity_balance_score,
    mix_probability_endpoints,
    probability_endpoints,
    query_set_route_metrics,
)
from floorflow.training.raster_instance_setup import (
    materialize_float_outputs,
    weighted_loss_sum,
)


def test_identity_balance_rejects_count_only_improvement() -> None:
    reference = {
        "room_macro_iou": 0.30,
        "instance_room_f1": 0.30,
        "instance_count_abs_error": 4.0,
        "instance_type_accuracy": 0.60,
    }
    count_only = {
        "room_macro_iou": 0.29,
        "instance_room_f1": 0.29,
        "instance_count_abs_error": 3.0,
        "instance_type_accuracy": 0.60,
    }
    identity_gain = {
        "room_macro_iou": 0.31,
        "instance_room_f1": 0.31,
        "instance_count_abs_error": 3.9,
        "instance_type_accuracy": 0.60,
    }
    assert identity_balance_score(
        identity_gain,
        reference,
    ) > identity_balance_score(count_only, reference)
    assert identity_balance_score(
        identity_gain,
        reference,
        count_weight=0.1,
    ) > identity_balance_score(
        count_only,
        reference,
        count_weight=0.1,
    )


def test_weighted_loss_sum_uses_boundary_weight() -> None:
    args = SimpleNamespace(
        partition_weight=0.5,
        fused_identity_weight=0.5,
        dense_weight=0.5,
        query_set_weight=0.0,
        room_program_weight=0.0,
        teacher_weight=0.25,
        room_boundary_weight=0.5,
    )
    losses = {
        "room_mask": torch.tensor(2.0),
        "room_boundary": torch.tensor(4.0),
    }
    assert weighted_loss_sum(losses, args) == 4.0


def test_blend_endpoints_reuse_softmax_without_changing_readout() -> None:
    outputs = {
        "dense_logits": torch.randn(2, 4, 8, 8),
        "fused_logits": torch.randn(2, 4, 8, 8),
    }
    dense, fused = probability_endpoints(outputs)
    for weight in (0.0, 0.25, 0.5, 1.0):
        expected = (
            (1.0 - weight) * outputs["dense_logits"].softmax(dim=1)
            + weight * outputs["fused_logits"].softmax(dim=1)
        )
        assert torch.allclose(
            mix_probability_endpoints(dense, fused, weight),
            expected,
        )
        assert torch.allclose(
            blended_probabilities(outputs, weight),
            expected,
        )


def test_mixed_precision_outputs_cross_the_loss_boundary_as_fp32() -> None:
    value = {
        "logits": torch.ones(1, 2, dtype=torch.bfloat16),
        "indices": torch.ones(2, dtype=torch.long),
        "aux": [torch.ones(1, dtype=torch.float16)],
    }
    materialized = materialize_float_outputs(value)
    assert materialized["logits"].dtype == torch.float32
    assert materialized["indices"].dtype == torch.long
    assert materialized["aux"][0].dtype == torch.float32


def test_query_set_route_metrics_flatten_finite_diagnostics() -> None:
    metrics = query_set_route_metrics(
        {
            "query_set_diagnostics": {
                "route_entropy": torch.tensor(0.5),
                "route_attention_entropy": torch.tensor(float("nan")),
                "route_effective_expert_fraction": torch.tensor(0.25),
                "route_budget_mean": torch.tensor(2.5),
                "route_gate_mean": torch.tensor(0.75),
            }
        }
    )
    assert metrics == {
        "query_route_entropy": 0.5,
        "query_route_effective_expert_fraction": 0.25,
        "query_route_budget": 2.5,
        "query_route_gate_mean": 0.75,
    }
