#!/usr/bin/env python3
"""Audit that furniture query budgets control canonical rendering."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.dynamic_object_instance import (
    DynamicFurnitureInstanceHead,
    render_object_queries,
    render_object_queries_soft,
)
from floorflow.inference.furniture_vector_readout import (
    read_furniture_queries,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.manual_seed(19)
    head = DynamicFurnitureInstanceHead(
        8,
        2,
        hidden_dim=16,
        max_queries=4,
        min_queries=2,
    )
    head_output = head(
        torch.randn(1, 8, 8, 8),
        torch.full((1, 1, 8, 8), 0.5),
        torch.ones(1, 1, 8, 8),
    )
    class_logits = torch.full((1, 2, 4), -5.0)
    class_logits[:, :, 0] = 5.0
    masks = torch.full((1, 2, 8, 8), -100.0)
    masks[0, 0, 1:3, 1:3] = 8.0
    masks[0, 1, 5:7, 5:7] = 8.0
    footprint = torch.ones(1, 1, 8, 8)
    active = torch.tensor([[True, False]])
    hard = render_object_queries(
        class_logits,
        masks,
        (3, 4),
        footprint,
        active_queries=active,
    )
    soft = render_object_queries_soft(
        class_logits,
        masks,
        footprint,
        active_queries=active,
    )
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 1}),
            "furniture": Counter({"Bed": 1, "Chair": 1}),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    vector_active_logits = torch.full(
        (2, len(taxonomy.furniture_ids) + 1),
        -8.0,
    )
    vector_active_logits[0, 0] = 1.0
    vector_active_logits[1, 1] = 8.0
    vector_active_masks = torch.full((2, 8, 8), -100.0)
    vector_active_masks[0, 1:5, 1:5] = 8.0
    vector_active_masks[1, 4:8, 4:8] = 8.0
    vector = read_furniture_queries(
        vector_active_logits,
        vector_active_masks,
        torch.full((8, 8), taxonomy.room_ids[0]),
        taxonomy,
        query_budget=1,
        active_queries=torch.tensor([[True, False]]),
        presence_threshold=0.1,
        mask_threshold=0.5,
        min_area=1,
    )
    checks = {
        "head_emits_positive_budget": bool(
            (head_output["query_budget"] >= head.min_queries).all()
        ),
        "head_emits_matching_active_mask": (
            head_output["active_queries"].shape
            == head_output["class_logits"].shape[:2]
        ),
        "hard_render_keeps_active_query": bool(
            (hard[0, 1:3, 1:3] == 3).all()
        ),
        "hard_render_drops_inactive_query": bool(
            hard[0, 5:7, 5:7].eq(0).all()
        ),
        "soft_render_drops_inactive_query": bool(
            soft[0, 1, 5:7, 5:7].eq(0).all()
        ),
        "vector_readout_uses_active_query": bool(
            len(vector) == 1
            and vector[0].label
            == taxonomy.class_names[taxonomy.furniture_ids[0]]
        ),
    }
    report = {
        "schema_version": "furniture-query-budget-consumer-audit-v1",
        "device": "cpu",
        "checks": checks,
        "passed": all(checks.values()),
        "query_budget": head_output["query_budget"].tolist(),
        "active_queries": head_output["active_queries"].tolist(),
        "claim_boundary": "Consumer wiring only; matched furniture metrics remain pending.",
    }
    output = ROOT / "outputs/reports/furniture_query_budget_consumer_audit_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
