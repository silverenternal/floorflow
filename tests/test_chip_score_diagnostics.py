from types import SimpleNamespace

import numpy as np

from floorflow.chip_flow.score_diagnostics import score_diagnostics


def test_score_diagnostics_decomposes_soft_and_net_errors() -> None:
    boxes = np.array(
        [[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0]],
        dtype=np.float64,
    )
    gt_boxes = boxes.copy()
    score = SimpleNamespace(
        cost=1.25,
        area_gap=0.1,
        hpwl_gap=0.2,
        v_rel=0.3,
        soft=SimpleNamespace(grouping=1, mib=2, boundary=3, n_soft=10),
        feasibility=SimpleNamespace(
            overlap_count=0,
            area_violations=[],
            fixed_violations=[],
            preplaced_violations=[],
            aspect_violations=[],
        ),
    )
    result = score_diagnostics(
        "legalized",
        score,
        boxes,
        gt_boxes,
        np.array([[0.0, 1.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0]]),
    )
    assert result["legalized_area_gap"] == 0.1
    assert result["legalized_hpwl_gap"] == 0.2
    assert result["legalized_b2b_hpwl_gap"] == 0.0
    assert result["legalized_p2b_hpwl_gap"] == 0.0
    assert result["legalized_rho_mae"] == 0.0
    assert result["legalized_grouping"] == 1
    assert result["legalized_mib"] == 2
    assert result["legalized_boundary"] == 3
    assert result["legalized_soft_slots"] == 10
    assert result["legalized_fixed_violations"] == 0
    assert result["legalized_preplaced_violations"] == 0
