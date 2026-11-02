#!/usr/bin/env python3
"""Audit inactive predicted room queries across the editable state contract."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from floorflow.inference.predicted_control_state import (
    _mask_inactive_query_slots,
    predicted_control_typed_state,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    active = torch.tensor([[True, False, True]])
    payload = {
        "room_identity_tokens": torch.ones(1, 3, 4),
        "room_instance_ids": torch.tensor([[5, 6, 7]]),
        "room_instance_query_existence_logits": torch.ones(1, 3),
        "room_instance_class_logits": torch.ones(1, 3, 3),
        "room_instance_mask_logits": torch.ones(1, 3, 4, 4),
        "room_instance_aux_mask_logits": [
            torch.ones(1, 3, 4, 4),
        ],
        "room_adjacency_logits": torch.ones(1, 3, 3),
        "program_room_assignment_logits": torch.ones(1, 3, 2),
        "program_room_scheduled_logits": torch.ones(1, 3, 2),
    }
    _mask_inactive_query_slots(payload, active)
    full_output = {
        **payload,
        "action_stage_tokens": torch.zeros(1, 2, 4),
        "query_count_selection_mask": active,
    }
    state = predicted_control_typed_state(full_output)
    checks = {
        "inactive_instance_id_is_negative": (
            int(state.room_instance_ids[0, 1]) == -1
        ),
        "active_instance_ids_are_preserved": (
            state.room_instance_ids[0, [0, 2]].tolist() == [5, 7]
        ),
        "inactive_tokens_are_zero": (
            bool(torch.count_nonzero(state.room_features[0, 1]) == 0)
        ),
        "inactive_classes_are_no_object": (
            bool(state.room_class[0, 1, :-1].max() < -1.0e30)
            and bool(state.room_class[0, 1, -1] == 0.0)
        ),
        "inactive_adjacency_is_masked": (
            bool(state.room_adjacency_logits[0, 1].max() < -1.0e30)
            and bool(
                state.room_adjacency_logits[0, :, 1].max() < -1.0e30
            )
        ),
        "inactive_mask_is_masked": (
            bool(
                payload["room_instance_mask_logits"][0, 1].max()
                < -1.0e30
            )
        ),
        "inactive_aux_mask_is_masked": (
            bool(
                payload["room_instance_aux_mask_logits"][0][0, 1].max()
                < -1.0e30
            )
        ),
        "inactive_program_assignment_is_masked": (
            bool(
                payload["program_room_assignment_logits"][0, 1].max()
                < -1.0e30
            )
        ),
        "inactive_scheduled_assignment_is_masked": (
            bool(
                payload["program_room_scheduled_logits"][0, 1].max()
                < -1.0e30
            )
        ),
    }
    report = {
        "schema_version": "predicted-edit-ghost-state-audit-v1",
        "device": "cpu",
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": "State-contract evidence only; predicted editing metrics remain pending.",
    }
    output = ROOT / "outputs/reports/predicted_edit_ghost_state_audit_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
