from __future__ import annotations

import numpy as np

from floorflow.chip_flow.rho_predictability import candidate_rank_metrics, case_flags


def test_candidate_rank_metrics_reward_correct_rho_order() -> None:
    metric = candidate_rank_metrics(
        [np.asarray([0.3, 0.1, 0.2])],
        [np.asarray([0.6, 0.2, 0.4])],
    )
    assert metric["cases"] == 1
    assert metric["top1_accuracy"] == 1.0
    assert metric["pair_accuracy"] == 1.0
    assert metric["mean_case_spearman"] == 1.0


def test_case_flags_require_every_structural_guard() -> None:
    reference = {
        "legalized_hard_cost": 1.0,
        "legalized_v_rel": 0.1,
        "legalized_area_gap": 0.1,
        "legalized_b2b_hpwl_gap": 0.1,
        "legalized_p2b_hpwl_gap": 0.1,
        "legalized_grouping": 1.0,
        "legalized_boundary": 1.0,
    }
    candidate = {**reference, "legalized_rho_mae": 0.0}
    reference["legalized_rho_mae"] = 0.1
    assert case_flags(candidate, reference) == (True, True, True)
    candidate["legalized_p2b_hpwl_gap"] = 0.2
    assert case_flags(candidate, reference) == (True, False, False)
