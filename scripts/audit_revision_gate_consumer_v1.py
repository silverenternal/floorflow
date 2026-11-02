#!/usr/bin/env python3
"""Audit revision-gate admission and fallback in the typed edit graph."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from floorflow.heads.revision_state_head import (
    RevisionStateHead,
    revision_state_loss,
)
from floorflow.inference.learned_revision import (
    consume_learned_revision,
)

ROOT = Path(__file__).resolve().parents[1]


def _revision(gate: float) -> dict[str, torch.Tensor]:
    return {
        "family_logits": torch.tensor([[4.0, 3.0, -4.0, -4.0]]),
        "region_logits": torch.full((1, 1, 8, 8), 4.0),
        "revision_gate": torch.tensor([gate]),
    }


def main() -> None:
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[:4] = 1
    labels[4:] = 2
    edit_region = np.zeros((8, 8), dtype=bool)
    edit_region[2:6, 2:6] = True
    changed = labels == 2
    arguments = {
        "room_ids": (1, 2),
        "wall_id": 3,
        "opening_ids": (4,),
        "furniture_ids": (5,),
    }
    admitted_graph, admitted = consume_learned_revision(
        labels,
        edit_region,
        changed,
        _revision(0.9),
        **arguments,
    )
    fallback_graph, fallback = consume_learned_revision(
        labels,
        edit_region,
        changed,
        _revision(0.1),
        **arguments,
    )
    head = RevisionStateHead(8, hidden_dim=16, operations=4, families=4)
    features = torch.randn(1, 8, 8, 8)
    region_logits = torch.zeros(1, 1, 8, 8)
    region_logits[:, :, 2:6, 2:6] = 1.0
    prediction = head(features, region_logits)
    loss, components = revision_state_loss(
        prediction,
        torch.tensor([1]),
        torch.tensor([0]),
        region_logits,
    )
    loss.backward()
    checks = {
        "high_gate_admits_learned_families": (
            admitted == ("room", "wall")
        ),
        "low_gate_uses_declared_families": fallback == ("room",),
        "low_gate_preserves_anchor": bool(fallback_graph.anchor.any()),
        "gate_changes_graph_families": (
            admitted_graph.changed_families
            != fallback_graph.changed_families
        ),
        "gate_loss_is_reported": "revision_gate_loss" in components,
        "gate_head_receives_gradient": bool(
            head.gate[-1].weight.grad is not None
            and head.gate[-1].weight.grad.abs().sum() > 0.0
        ),
    }
    report = {
        "schema_version": "revision-gate-consumer-audit-v1",
        "device": "cpu",
        "checks": checks,
        "passed": all(checks.values()),
        "admitted_families": list(admitted),
        "fallback_families": list(fallback),
        "claim_boundary": "Consumer wiring only; predicted edit metrics remain pending.",
    }
    output = ROOT / "outputs/reports/revision_gate_consumer_audit_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
