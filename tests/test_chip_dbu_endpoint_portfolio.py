from __future__ import annotations

import numpy as np
from trinity.floorplan.types import FloorplanInstance

from floorflow.chip_flow.dbu_endpoint_portfolio import (
    displacement_summary,
    project_dbu_endpoint_portfolio,
)


def _instance(boxes: np.ndarray) -> FloorplanInstance:
    count = len(boxes)
    return FloorplanInstance(
        block_count=count,
        area_targets=boxes[:, 2] * boxes[:, 3],
        constraints=np.zeros((count, 5), dtype=np.float64),
        b2b=np.empty((0, 3), dtype=np.float64),
        p2b=np.empty((0, 3), dtype=np.float64),
        pins_pos=np.empty((0, 2), dtype=np.float64),
        target_positions=np.full((count, 4), -1.0, dtype=np.float64),
        gt_positions=None,
    )


def test_displacement_summary_removes_global_translation() -> None:
    before = np.asarray([[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]])
    after = np.asarray([[3.0, 4.0, 1.0, 1.0], [5.0, 4.0, 1.0, 1.0]])

    p50, p95, maximum = displacement_summary(before, after)

    assert p50 == p95 == maximum == 0.0


def test_displacement_summary_keeps_nonrigid_motion() -> None:
    before = np.asarray([[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]])
    after = np.asarray([[3.0, 4.0, 1.0, 1.0], [9.0, 4.0, 1.0, 1.0]])

    p50, p95, maximum = displacement_summary(before, after)

    assert p50 == p95 == maximum == 2.0


def test_portfolio_returns_audited_endpoint_on_small_case() -> None:
    boxes = np.asarray(
        [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]],
        dtype=np.float64,
    )
    instance = _instance(boxes)
    result = project_dbu_endpoint_portfolio(
        boxes,
        instance,
        reference_boxes=boxes,
        projection_time_limit_ms=3000,
    )

    assert result.selected.result.accepted
    assert result.selected.result.overlap_after == 0
    assert result.selected.displacement_p95 >= 0.0
    assert result.candidates
    assert result.tried_routes[0] == "expanded_continuous"
