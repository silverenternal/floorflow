import torch

from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutOpening,
    ReadoutWall,
)
from floorflow.heads.opening_interval_head import (
    WallIntervalHead,
    interval_targets_from_plan,
    wall_interval_loss,
)


def _plan() -> RasterWallPlan:
    return RasterWallPlan(
        frame=64.0,
        walls=[
            ReadoutWall(start=(8.0, 50.0), end=(56.0, 50.0), thickness=3.0),
            ReadoutWall(start=(8.0, 10.0), end=(56.0, 10.0), thickness=3.0),
        ],
        rooms=[],
        openings=[
            ReadoutOpening(
                center=(32.0, 50.0),
                width=8.0,
                kind="door",
                wall_index=0,
            )
        ],
    )


def test_interval_targets_are_derived_from_the_plan() -> None:
    targets = interval_targets_from_plan(_plan())
    assert float(targets["presence"].sum()) == 1.0
    assert 0.0 < float(targets["start"][0]) < 1.0
    assert float(targets["length"][0]) > 0.0
    assert int(targets["type"][0]) == 0


def test_wall_interval_head_is_trainable() -> None:
    head = WallIntervalHead(16, num_types=2, heads=4)
    wall_tokens = torch.randn(1, 2, 16, requires_grad=True)
    opening_tokens = torch.randn(1, 2, 16, requires_grad=True)
    endpoints = torch.tensor(
        [[[8.0, 50.0, 56.0, 50.0], [8.0, 10.0, 56.0, 10.0]]]
    )
    predictions = head(wall_tokens, opening_tokens, endpoints)
    targets = interval_targets_from_plan(_plan())
    loss, components = wall_interval_loss(predictions, targets)
    assert torch.isfinite(loss)
    assert set(components) == {
        "opening_interval_presence_loss",
        "opening_interval_start_loss",
        "opening_interval_length_loss",
        "opening_interval_type_loss",
    }
    loss.backward()
    assert head.interval[2].weight.grad is not None
