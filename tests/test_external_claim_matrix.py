"""Tests for external claim branch isolation."""

from __future__ import annotations

from floorflow.evaluation.external_claim_matrix import build_claim_matrix


def test_only_matched_branch_can_enable_and_missing_methods_stay_closed() -> None:
    checks = {
        "room_iou": True,
        "node_f1": True,
    }
    matrix = build_claim_matrix(
        paired_checks=checks,
        matched_executable=True,
    )
    assert matrix["matched_cubicasa_reconstruction_head_to_head"]["enabled"]
    assert not matrix["complete_cad_generation_sota"]["enabled"]
    assert not matrix["rplan_generation_sota"]["enabled"]
    assert not matrix["house_mind_editing_defeat"]["enabled"]
