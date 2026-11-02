from __future__ import annotations

import torch
from torch import nn

from floorflow.objectives.routing_regularization import (
    typed_routing_regularization,
)


class _ParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_typed_routing_regularization_penalizes_collapse() -> None:
    device = _device()
    model = _ParameterModel().to(device)
    weights = torch.zeros(2, 4, 4, 4, 4, device=device)
    weights[:, :, 0] = 1.0
    output = {
        "field_source_routing": {
            "source_weights": weights,
            "source_entropy": torch.tensor(0.1, device=device),
        },
        "channel_source_attention_entropy": torch.tensor(
            0.1,
            device=device,
        ),
        "action_route_diagnostics": {
            "walls": {
                "routing_entropy": torch.tensor(0.1, device=device),
                "effective_expert_fraction": torch.tensor(
                    0.05,
                    device=device,
                ),
            }
        },
        "query_set_diagnostics": {
            "route_entropy": torch.tensor(0.1, device=device),
            "route_attention_entropy": torch.tensor(0.1, device=device),
            "route_effective_expert_fraction": torch.tensor(
                0.05,
                device=device,
            ),
        },
    }
    loss, components = typed_routing_regularization(model, output)
    assert loss > 0
    assert components["routing_source_floor_loss"] > 0
    assert components["routing_stage_floor_loss"] > 0


def test_typed_routing_regularization_accepts_healthy_routing() -> None:
    device = _device()
    model = _ParameterModel().to(device)
    weights = torch.full((1, 4, 4, 4, 4), 0.25, device=device)
    output = {
        "field_source_routing": {
            "source_weights": weights,
            "source_entropy": torch.tensor(1.0, device=device),
        },
        "channel_source_attention_entropy": torch.tensor(
            0.8,
            device=device,
        ),
        "action_route_diagnostics": {
            "walls": {
                "routing_entropy": torch.tensor(0.8, device=device),
                "effective_expert_fraction": torch.tensor(
                    0.5,
                    device=device,
                ),
            }
        },
    }
    loss, _components = typed_routing_regularization(model, output)
    assert loss == 0
