from __future__ import annotations

import torch

from floorflow.objectives.gradient_balance import gradient_diagnostics


def test_gradient_diagnostics_reports_norms_and_cosines() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    first = (parameter * torch.tensor([1.0, 0.0])).sum()
    second = (parameter * torch.tensor([0.0, 1.0])).sum()
    diagnostics = gradient_diagnostics(
        {"first": first, "second": second},
        [parameter],
    )
    assert diagnostics["norms"]["first"] == 1.0
    assert diagnostics["norms"]["second"] == 1.0
    assert abs(diagnostics["pairwise_cosine"]["first__second"]) < 1.0e-6
