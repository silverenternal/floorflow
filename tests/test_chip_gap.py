"""Tests for chip gap decomposition helpers."""

from __future__ import annotations

import math

import pytest

from floorflow.evaluation.chip_gap import canonical_row


def _row() -> dict:
    return {
        "index": 0,
        "legalized_hard_cost": 1.3,
        "legalized_v_rel": 0.2,
        "legalized_soft_slots": 10.0,
        "legalized_grouping": 1,
        "legalized_boundary": 1,
        "legalized_mib": 0,
        "legalized_hpwl_gap": 0.2,
        "legalized_area_gap": 0.1,
        "legalized_b2b_hpwl_gap": 0.05,
        "legalized_p2b_hpwl_gap": 0.1,
        "legalized_rho_mae": 0.25,
    }


def test_soft_components_reconcile_to_v_rel() -> None:
    row = canonical_row(_row(), arm="candidate")

    assert row["grouping_contribution"] == pytest.approx(0.1)
    assert row["boundary_contribution"] == pytest.approx(0.1)
    assert row["mib_contribution"] == pytest.approx(0.0)
    assert row["soft_component_sum"] == pytest.approx(row["v_rel"])


def test_cost_reconstruction_uses_official_terms() -> None:
    row = canonical_row(_row(), arm="candidate")

    assert row["reconstructed_cost"] == pytest.approx(
        (1.0 + 0.5 * (0.2 + 0.1)) * math.exp(2.0 * 0.2)
    )
