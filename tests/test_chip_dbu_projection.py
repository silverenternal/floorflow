from __future__ import annotations

import numpy as np
from trinity.floorplan.types import FloorplanInstance

from floorflow.chip_flow.dbu_projection import project_quantized_layout


def _instance(boxes: np.ndarray) -> FloorplanInstance:
    count = len(boxes)
    constraints = np.zeros((count, 5), dtype=np.float64)
    targets = np.full((count, 4), -1.0, dtype=np.float64)
    return FloorplanInstance(
        block_count=count,
        area_targets=boxes[:, 2] * boxes[:, 3],
        constraints=constraints,
        b2b=np.empty((0, 3), dtype=np.float64),
        p2b=np.empty((0, 3), dtype=np.float64),
        pins_pos=np.empty((0, 2), dtype=np.float64),
        target_positions=targets,
        gt_positions=None,
    )


def test_projection_resolves_overlap_on_integer_grid() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )
    result = project_quantized_layout(
        boxes,
        _instance(boxes),
        projection_time_limit_ms=3000,
    )

    assert result.accepted
    assert result.route in {
        "scale_pack_integer_projection_sequential",
        "scale_pack_integer_projection_contact_tree",
    }
    assert result.overlap_after == 0
    assert result.area_violations == 0
    assert result.fixed_violations == 0
    assert result.preplaced_violations == 0
    assert result.max_grid_residual < 1.0e-6
    np.testing.assert_allclose(result.boxes[:, 2:], boxes[:, 2:])


def test_projection_rejects_nonpositive_resolution() -> None:
    boxes = np.asarray([[0.0, 0.0, 2.0, 2.0]], dtype=np.float64)
    try:
        project_quantized_layout(
            boxes,
            _instance(boxes),
            resolution=0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive resolution must fail")
