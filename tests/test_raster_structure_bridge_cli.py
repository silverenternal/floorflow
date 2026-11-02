import sys

import pytest

from floorflow.training.raster_structure_bridge_cli import (
    parse_args,
    validate_args,
)


def test_typed_structure_contract_options_are_available(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--structure-contract",
            "typed_v2",
            "--room-query-checkpoint",
            "query.pt",
            "--room-query-blend",
            "0.75",
            "--room-blend-jitter",
            "0.1",
            "--structure-relation-weight",
            "0.5",
            "--structure-global-route-attention",
            "--host-conditioning-v2",
            "--structure-route-grid",
            "8",
            "--structure-route-router-count",
            "8",
            "--structure-route-top-k",
            "4",
        ],
    )
    args = parse_args()
    validate_args(args)
    assert args.structure_contract == "typed_v2"
    assert args.room_query_blend == 0.75
    assert args.room_blend_jitter == 0.1
    assert args.epochs == 12
    assert args.structure_global_route_attention
    assert args.host_conditioning_v2
    assert args.structure_route_grid == 8
    assert args.structure_route_router_count == 8
    assert args.structure_route_top_k == 4


def test_structure_blend_jitter_requires_blend(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--room-blend-jitter", "0.1"],
    )
    args = parse_args()
    with pytest.raises(ValueError):
        validate_args(args)


def test_structure_global_routing_requires_typed_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--structure-global-route-attention"],
    )
    args = parse_args()
    with pytest.raises(ValueError, match="typed_v2"):
        validate_args(args)


def test_typed_host_conditioning_requires_typed_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--host-conditioning-v2"],
    )
    args = parse_args()
    with pytest.raises(ValueError, match="typed_v2"):
        validate_args(args)
