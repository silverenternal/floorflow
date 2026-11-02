"""Tests for many-opening-to-host-wall routing."""

from collections import Counter

import pytest
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.opening_host_router import (
    OpeningHostRouter,
    opening_host_router_loss,
    opening_route_metrics,
    routed_opening_rows,
    slice_tensor_predictions,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 5}),
            "furniture": Counter(),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_router_supports_multiple_openings_on_one_wall() -> None:
    router = OpeningHostRouter(16, num_types=2)
    walls = torch.randn(1, 3, 16, requires_grad=True)
    openings = torch.randn(1, 4, 16, requires_grad=True)
    endpoints = torch.tensor(
        [
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            ]
        ]
    )
    output = router(walls, openings, endpoints)
    loss, components = opening_host_router_loss(
        output,
        target_hosts=torch.tensor([[0, 0, 1, 2]]),
        target_center_fraction=torch.tensor([[0.2, 0.7, 0.5, 0.4]]),
        target_width_fraction=torch.tensor([[0.1, 0.2, 0.2, 0.1]]),
        target_types=torch.tensor([[0, 1, 0, 1]]),
        matched_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    loss.backward()
    assert output["host_logits"].shape == (1, 4, 3)
    assert output["center_xy"].shape == (1, 4, 2)
    assert set(components) == {
        "opening_query_presence_loss",
        "opening_host_loss",
        "opening_center_fraction_loss",
        "opening_width_fraction_loss",
        "opening_type_loss",
    }
    assert walls.grad is not None
    assert openings.grad is not None


def test_router_handles_empty_wall_batch_without_nan() -> None:
    router = OpeningHostRouter(8, num_types=2)
    output = router(
        torch.zeros(1, 0, 8),
        torch.randn(1, 2, 8),
        torch.zeros(1, 0, 2, 2),
    )
    assert torch.isfinite(output["host_logits"]).all()
    assert torch.isfinite(output["center_xy"]).all()


def test_geometry_prior_routes_to_nearest_segment() -> None:
    router = OpeningHostRouter(8, num_types=2)
    endpoints = torch.tensor(
        [
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.4], [1.0, 0.4]],
            ]
        ]
    )
    output = router(
        torch.zeros(1, 2, 8),
        torch.zeros(1, 1, 8),
        endpoints,
        opening_centers=torch.tensor([[[0.5, 0.36]]]),
        opening_widths=torch.tensor([[0.2]]),
    )
    assert output["host_index"].item() == 1
    assert output["center_fraction"].item() == pytest.approx(0.5, abs=1.0e-6)
    assert output["width_fraction"].item() == pytest.approx(0.2, abs=1.0e-6)


def test_opening_router_hybrid_mita_is_dynamic_and_trainable() -> None:
    router = OpeningHostRouter(
        16,
        num_types=2,
        route_attention=True,
        route_router_count=4,
        route_top_k=2,
    )
    output = router(
        torch.randn(2, 8, 16),
        torch.randn(2, 5, 16),
        torch.rand(2, 8, 2, 2),
        wall_mask=torch.ones(2, 8, dtype=torch.bool),
        opening_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    diagnostics = output["route_diagnostics"]
    assert torch.isfinite(diagnostics["routing_entropy"])
    assert torch.isfinite(diagnostics["route_budget_soft_mean"])
    output["host_logits"].square().mean().backward()
    assert router.route is not None
    assert router.route.q_proj.weight.grad is not None
    assert router.route_budget[0].weight.grad is not None
    metrics = opening_route_metrics(output)
    assert metrics["route_budget"] > 0.0
    assert 0.0 <= metrics["route_entropy"] <= 1.0


def test_routed_rows_are_a_typed_cad_consumer() -> None:
    taxonomy = _taxonomy()
    predictions = {
        "presence_logits": torch.tensor([[4.0, 4.0]]),
        "opening_mask": torch.tensor([[True, True]]),
        "host_index": torch.tensor([[0, 0]]),
        "host_endpoints": torch.tensor(
            [[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]]]
        ),
        "center_fraction": torch.tensor([[0.25, 0.75]]),
        "width_fraction": torch.tensor([[0.1, 0.2]]),
        "center_xy": torch.tensor([[[0.25, 0.0], [0.75, 0.0]]]),
        "type_logits": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]]
        ),
    }
    rows = routed_opening_rows(predictions, 0, taxonomy)
    assert len(rows) == 2
    assert [row.wall_index for row in rows] == [0, 0]


def test_routed_rows_scale_normalized_geometry_to_frame() -> None:
    taxonomy = _taxonomy()
    predictions = {
        "presence_logits": torch.tensor([[4.0]]),
        "opening_mask": torch.tensor([[True]]),
        "host_index": torch.tensor([[0]]),
        "host_endpoints": torch.tensor(
            [[[[0.0, 0.0], [1.0, 0.0]]]]
        ),
        "center_fraction": torch.tensor([[0.5]]),
        "width_fraction": torch.tensor([[0.2]]),
        "center_xy": torch.tensor([[[0.5, 0.0]]]),
        "type_logits": torch.tensor([[[1.0, 0.0]]]),
    }
    row = routed_opening_rows(predictions, 0, taxonomy, frame=100.0)[0]
    assert row.center == (50.0, 0.0)
    assert row.width == pytest.approx(20.0, abs=1.0e-6)


def test_routed_rows_can_preserve_detection_intervals() -> None:
    taxonomy = _taxonomy()
    predictions = {
        "presence_logits": torch.tensor([[4.0]]),
        "opening_mask": torch.tensor([[True]]),
        "host_index": torch.tensor([[0]]),
        "host_endpoints": torch.tensor(
            [[[[0.0, 0.0], [1.0, 0.0]]]]
        ),
        "width_fraction": torch.tensor([[0.8]]),
        "center_xy": torch.tensor([[[0.8, 0.0]]]),
        "type_logits": torch.tensor([[[1.0, 0.0]]]),
    }
    row = routed_opening_rows(
        predictions,
        0,
        taxonomy,
        frame=100.0,
        query_centers=torch.tensor([[[0.25, 0.0]]]),
        query_widths=torch.tensor([[0.1]]),
        query_families=torch.tensor([[1]]),
        preserve_query_intervals=True,
    )[0]
    assert row.center == (25.0, 0.0)
    assert row.width == pytest.approx(10.0)
    assert row.kind == "window"


def test_tensor_prediction_slice_drops_route_diagnostics() -> None:
    sliced = slice_tensor_predictions(
        {
            "presence_logits": torch.tensor([[1.0], [2.0]]),
            "scalar": torch.tensor(3.0),
            "route_diagnostics": {"routing_entropy": torch.tensor(0.5)},
        },
        1,
    )
    assert torch.equal(sliced["presence_logits"], torch.tensor([[2.0]]))
    assert "route_diagnostics" not in sliced
    assert "scalar" not in sliced


def test_routed_rows_consume_learned_intervals_by_default() -> None:
    taxonomy = _taxonomy()
    predictions = {
        "presence_logits": torch.tensor([[4.0]]),
        "opening_mask": torch.tensor([[True]]),
        "host_index": torch.tensor([[0]]),
        "host_endpoints": torch.tensor(
            [[[[0.0, 0.0], [1.0, 0.0]]]]
        ),
        "center_fraction": torch.tensor([[0.8]]),
        "width_fraction": torch.tensor([[0.7]]),
        "center_xy": torch.tensor([[[0.8, 0.0]]]),
        "type_logits": torch.tensor([[[1.0, 0.0]]]),
    }
    row = routed_opening_rows(
        predictions,
        0,
        taxonomy,
        frame=100.0,
        query_centers=torch.tensor([[[0.25, 0.0]]]),
        query_widths=torch.tensor([[0.1]]),
    )[0]
    assert row.center == pytest.approx((80.0, 0.0))
    assert row.width == pytest.approx(70.0)


def test_routed_rows_drop_openings_on_degenerate_hosts() -> None:
    taxonomy = _taxonomy()
    predictions = {
        "presence_logits": torch.tensor([[4.0]]),
        "opening_mask": torch.tensor([[True]]),
        "host_index": torch.tensor([[0]]),
        "host_endpoints": torch.tensor(
            [[[[0.5, 0.5], [0.5, 0.5]]]]
        ),
        "width_fraction": torch.tensor([[0.2]]),
        "center_xy": torch.tensor([[[0.5, 0.5]]]),
        "type_logits": torch.tensor([[[1.0, 0.0]]]),
    }
    assert routed_opening_rows(predictions, 0, taxonomy, frame=100.0) == []
