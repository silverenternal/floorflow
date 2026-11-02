#!/usr/bin/env python3
"""Audit predicted-control opening and furniture deletion execution."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.predicted_control_edit import (
    project_predicted_instruction,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter({"Bed": 2}),
            "door": Counter({"Swing": 2}),
            "window": Counter({"Regular": 2}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    before = torch.full(
        (12, 12),
        taxonomy.room_ids[0],
        dtype=torch.long,
    )
    before[4:7, 4:7] = taxonomy.opening_ids[0]
    opening_controls = {
        "room_logits": torch.zeros(
            len(taxonomy.room_ids),
            12,
            12,
        ),
        "program": torch.ones(len(taxonomy.room_ids), dtype=torch.bool),
    }
    opening_after, mask = project_predicted_instruction(
        before,
        opening_controls,
        (3, 3, 6, 6),
        (12, 12),
        taxonomy,
        operation="opening_delete",
    )
    furniture_before = before.clone()
    furniture_before[4:7, 4:7] = taxonomy.furniture_ids[0]
    furniture_after, furniture_mask = project_predicted_instruction(
        furniture_before,
        opening_controls,
        (3, 3, 6, 6),
        (12, 12),
        taxonomy,
        operation="furniture_delete",
    )
    checks = {
        "opening_family_deleted_inside_region": bool(
            opening_after[4:7, 4:7].eq(0).all()
        ),
        "opening_room_pixels_preserved": bool(
            opening_after[0:3, 0:3]
            .eq(taxonomy.room_ids[0])
            .all()
        ),
        "opening_outside_region_exact": bool(
            torch.equal(opening_after[~mask], before[~mask])
        ),
        "furniture_family_deleted_inside_region": bool(
            furniture_after[4:7, 4:7].eq(0).all()
        ),
        "furniture_room_pixels_preserved": bool(
            furniture_after[0:3, 0:3]
            .eq(taxonomy.room_ids[0])
            .all()
        ),
        "furniture_outside_region_exact": bool(
            torch.equal(
                furniture_after[~furniture_mask],
                furniture_before[~furniture_mask],
            )
        ),
    }
    report = {
        "schema_version": "predicted-edit-object-delete-audit-v1",
        "device": "cpu",
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": "Execution-contract evidence only; matched edit metrics remain pending.",
    }
    output = ROOT / "outputs/reports/predicted_edit_object_delete_audit_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
