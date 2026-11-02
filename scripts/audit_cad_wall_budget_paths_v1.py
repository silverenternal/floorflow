#!/usr/bin/env python3
"""Audit first-step gradient and sample dependence of wall budget heads."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from floorflow.heads.cad_wall_candidate_scorer import CadWallCandidateScorer

ROOT = Path(__file__).resolve().parents[1]


def _grad_sum(module: torch.nn.Module) -> float:
    values = [
        parameter.grad.detach().abs().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(values).sum()) if values else 0.0


def main() -> None:
    torch.manual_seed(19)
    scorer = CadWallCandidateScorer(8, hidden_dim=16, graph_dim=12)
    valid = torch.ones(2, 3, dtype=torch.bool)
    endpoints = torch.rand(2, 3, 2, 2)
    evidence = torch.full((2, 3), 0.5)
    graph = torch.randn(2, 2, 12)
    first = scorer(
        torch.randn(2, 8, 8, 8),
        endpoints,
        valid,
        evidence,
        graph,
    )
    second = scorer(
        torch.randn(2, 8, 8, 8),
        endpoints,
        valid,
        evidence,
        graph,
    )
    loss = (
        first["budget_fraction"].sum()
        + first["exterior_budget_fraction"].sum()
        + first["interior_budget_fraction"].sum()
    )
    loss.backward()
    phase_grads = {
        name: {
            "first_layer": _grad_sum(head[0]),
            "last_layer": _grad_sum(head[2]),
        }
        for name, head in scorer.phase_budget.items()
    }
    checks = {
        "sample_dependent_budget": not torch.allclose(
            first["budget_fraction"],
            second["budget_fraction"],
        ),
        "global_budget_first_layer_live": (
            _grad_sum(scorer.budget[0]) > 0.0
        ),
        "global_budget_last_layer_live": (
            _grad_sum(scorer.budget[2]) > 0.0
        ),
        "phase_budget_live": all(
            values["first_layer"] > 0.0
            and values["last_layer"] > 0.0
            for values in phase_grads.values()
        ),
    }
    report = {
        "schema_version": "cad-wall-budget-path-audit-v1",
        "device": "cpu",
        "scope": "residential predicted-control wall candidate consumer",
        "checks": checks,
        "passed": all(checks.values()),
        "gradient_abs_sum": {
            "global_budget": {
                "first_layer": _grad_sum(scorer.budget[0]),
                "last_layer": _grad_sum(scorer.budget[2]),
            },
            **phase_grads,
        },
        "sample_budget_delta": float(
            (
                first["budget_fraction"]
                - second["budget_fraction"]
            ).abs().max().detach()
        ),
        "claim_boundary": "Path repair only; retraining and matched wall evaluation remain required.",
    }
    output = ROOT / "outputs/reports/cad_wall_budget_path_audit_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
