from __future__ import annotations

from floorflow.chip_flow.checkpoint_selection import (
    rank_legalized_candidates,
    select_legalized_candidate,
)


def test_legalized_candidate_ranking_prefers_guarded_primary_metrics() -> None:
    rows = [
        {
            "step": 10,
            "legalized_feasible": True,
            "legalized_hard_cost": 1.1,
            "legalized_v_rel": 0.02,
            "legalized_displacement_mean_um": 1.0,
            "legalizer_rung": 2,
            "raw_total": 0.02,
        },
        {
            "step": 20,
            "legalized_feasible": True,
            "legalized_hard_cost": 1.0,
            "legalized_v_rel": 0.03,
            "legalized_displacement_mean_um": 2.0,
            "legalizer_rung": 3,
            "raw_total": 0.01,
        },
        {
            "step": 30,
            "legalized_feasible": False,
            "legalized_hard_cost": 0.9,
            "legalized_v_rel": 0.01,
            "legalized_displacement_mean_um": 0.0,
            "legalizer_rung": 0,
            "raw_total": 0.0,
        },
    ]
    ranked = rank_legalized_candidates(rows)
    assert [row["step"] for row in ranked] == [20, 10, 30]
    assert select_legalized_candidate(rows)["step"] == 20


def test_legalized_candidate_ranking_breaks_primary_ties_with_structure() -> None:
    base = {
        "legalized_feasible": True,
        "legalized_hard_cost": 1.0,
        "legalized_v_rel": 0.02,
        "legalized_displacement_mean_um": 1.0,
        "legalizer_rung": 2,
        "raw_total": 0.02,
    }
    area_poor = {
        **base,
        "step": 1,
        "legalized_area_gap": 0.2,
        "legalized_b2b_hpwl_gap": 0.01,
        "legalized_grouping": 0.0,
        "legalized_boundary": 0.0,
    }
    structural_better = {
        **base,
        "step": 2,
        "legalized_area_gap": 0.1,
        "legalized_b2b_hpwl_gap": 0.02,
        "legalized_grouping": 0.0,
        "legalized_boundary": 0.0,
    }
    assert select_legalized_candidate([area_poor, structural_better])["step"] == 2
