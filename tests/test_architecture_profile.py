from __future__ import annotations

import torch

from floorflow.evaluation.architecture_profile import profile_callable, resource_dict


class _LinearModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def test_profile_callable_reports_parameters_and_macs() -> None:
    module = _LinearModule()
    inputs = torch.ones(2, 4)
    profile = profile_callable(
        module,
        lambda: module(inputs),
        device=torch.device("cpu"),
        warmup=1,
        repeats=2,
    )
    report = resource_dict(profile)
    assert report["parameters"] == 15
    assert report["macs"] == 24
    assert report["latency_ms_mean"] >= 0.0
    assert report["peak_memory_bytes"] is None
