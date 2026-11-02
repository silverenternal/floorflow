from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from trinity.floorplan.scoring.severity import score_severity
from trinity.floorplan.types import FloorplanInstance, Placement

from floorflow.chip_flow.official_cost_surrogate import (
    differentiable_soft_severity,
    official_cost_objective,
    official_cost_surrogate,
)


def _batch() -> dict:
    target = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]]]
    )
    constraints = torch.zeros(1, 2, 5)
    constraints[..., 3] = 1
    return {
        "areas": torch.ones(1, 2),
        "boxes": target,
        "constraints": constraints,
        "b2b": torch.tensor([[[0.0, 1.0, 1.0]]]),
        "p2b": torch.empty(1, 0, 3),
        "pins": torch.empty(1, 0, 2),
        "n_b2b": torch.ones(1, dtype=torch.long),
        "n_p2b": torch.zeros(1, dtype=torch.long),
        "n_pins": torch.zeros(1, dtype=torch.long),
    }


def test_official_cost_proxy_is_one_at_target_and_monotone() -> None:
    batch = _batch()
    target = batch["boxes"]
    exact = official_cost_surrogate(target, target, batch)
    worse_boxes = target.clone()
    worse_boxes[0, 1, 0] = 5.0
    worse = official_cost_surrogate(worse_boxes, target, batch)
    assert torch.allclose(exact.total, torch.tensor(1.0), atol=1.0e-6)
    assert exact.soft_v_rel < 1.0e-6
    assert worse.total > exact.total
    assert worse.hpwl_gap > 0
    assert worse.grouping > 0


def test_official_cost_objective_is_opt_in_differentiable() -> None:
    batch = _batch()
    target = batch["boxes"]
    prediction = target.clone().requires_grad_(True)
    disabled = SimpleNamespace(official_cost_weight=0.0)
    loss, terms = official_cost_objective(prediction, target, batch, disabled)
    assert loss == 0
    assert terms == {}
    enabled = SimpleNamespace(
        official_cost_weight=0.2,
        official_cost_violation_temperature=0.1,
    )
    loss, terms = official_cost_objective(prediction, target, batch, enabled)
    assert loss > 0
    assert terms["official_cost_current_weight"] == torch.tensor(0.2)
    loss.backward()
    assert prediction.grad is not None


def test_official_cost_objective_supports_delayed_annealing() -> None:
    batch = _batch()
    target = batch["boxes"]
    prediction = target.clone().requires_grad_(True)
    args = SimpleNamespace(
        official_cost_weight=0.0,
        official_cost_final_weight=0.2,
        official_cost_delay_steps=10,
        official_cost_anneal_steps=10,
        official_cost_violation_temperature=0.1,
    )
    early, _terms = official_cost_objective(
        prediction,
        target,
        batch,
        args,
        step=0,
    )
    late, terms = official_cost_objective(
        prediction,
        target,
        batch,
        args,
        step=20,
    )
    assert early == 0
    assert late > 0
    assert terms["official_cost_current_weight"] == torch.tensor(0.2)


def test_differentiable_soft_severity_matches_trinity_definition() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 2.0, 1.0],
            [3.0, 0.0, 1.0, 2.0],
            [3.0, 3.0, 1.0, 1.0],
            [0.0, 3.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    constraints = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    instance = FloorplanInstance(
        block_count=4,
        area_targets=boxes[:, 2] * boxes[:, 3],
        constraints=constraints,
        b2b=np.empty((0, 3)),
        p2b=np.empty((0, 3)),
        pins_pos=np.empty((0, 2)),
        target_positions=boxes,
    )
    reference = score_severity(
        Placement(boxes, instance),
        hpwl_gap=0.0,
        area_gap=0.0,
    )
    actual = differentiable_soft_severity(
        torch.from_numpy(boxes)[None],
        torch.from_numpy(instance.area_targets)[None],
        torch.from_numpy(constraints)[None],
    )

    dtype = actual["grouping"].dtype
    torch.testing.assert_close(
        actual["grouping"][0],
        torch.tensor(reference.grouping, dtype=dtype),
    )
    torch.testing.assert_close(
        actual["mib"][0],
        torch.tensor(reference.mib, dtype=dtype),
    )
    torch.testing.assert_close(
        actual["boundary"][0],
        torch.tensor(reference.boundary, dtype=dtype),
    )
    torch.testing.assert_close(
        actual["n_soft"][0],
        torch.tensor(reference.n_soft, dtype=dtype),
    )
    torch.testing.assert_close(
        actual["severity_relative"][0],
        torch.tensor(reference.sev_rel, dtype=dtype),
    )
