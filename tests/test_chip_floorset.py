import json

import pytest
import torch

from floorflow.chip_flow.cache import PackedCaseStore, write_packed_split
from floorflow.chip_flow.dataset import select_legacy_splits
from floorflow.chip_flow.floorset import (
    b2b_edges,
    case_from_current,
    case_from_legacy,
    edge_codes_to_boundary,
    p2b_edges,
)


def _legacy_columns() -> tuple[torch.Tensor, ...]:
    tree = torch.zeros(1, 2, 3, dtype=torch.long)
    sizes_pos = torch.tensor(
        [[[4, 2, 0, 0], [2, 4, 4, 0], [2, 2, 0, 2]]],
        dtype=torch.long,
    )
    pins = torch.tensor([[[8, 1], [1, 4]]], dtype=torch.long)
    b2b = torch.tensor(
        [[[0, 1, 0], [1, 0, 1], [0, 1, 0]]],
        dtype=torch.bool,
    )
    p2b = torch.tensor(
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        dtype=torch.bool,
    )
    edges = torch.tensor([[[0, 1], [2, 0], [1, 1]]], dtype=torch.long)
    preplaced = torch.tensor([False])
    fixed = torch.tensor([True])
    mib = torch.tensor([[0, 1, 1]], dtype=torch.long)
    cluster = torch.tensor([[1, 0, 0]], dtype=torch.float32)
    return tree, sizes_pos, pins, b2b, p2b, edges, preplaced, fixed, mib, cluster


def test_edge_codes_convert_to_boundary_bits() -> None:
    codes = torch.tensor([[0, 0], [1, 0], [2, 2], [1, 1]])
    assert edge_codes_to_boundary(codes).tolist() == [0.0, 4.0, 10.0, 5.0]


def test_adjacencies_become_flow_physics_edges() -> None:
    columns = _legacy_columns()
    assert b2b_edges(columns[3][0]).tolist() == [
        [0.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
    ]
    assert p2b_edges(columns[4][0]).tolist() == [
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 1.0],
    ]


def test_legacy_case_recovers_xywh_areas_and_constraints() -> None:
    case = case_from_legacy(_legacy_columns(), 0)
    assert case["boxes"].tolist() == [
        [0.0, 0.0, 4.0, 2.0],
        [4.0, 0.0, 2.0, 4.0],
        [0.0, 2.0, 2.0, 2.0],
    ]
    assert case["areas"].tolist() == [8.0, 8.0, 4.0]
    assert case["constraints"][:, :4].tolist() == [
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ]
    assert case["constraints"][:, 4].tolist() == [1.0, 8.0, 5.0]
    assert case["features"].shape == (3, 10)


def test_current_seven_column_case_recovers_native_constraints() -> None:
    raw = torch.tensor(
        [
            [
                [8, 1, 0, 0, 1, 1],
                [8, 0, 0, 1, 0, 2],
                [4, 0, 0, 0, 2, 5],
            ]
        ],
        dtype=torch.float32,
    )
    b2b = torch.tensor([[[0, 1, 0.5], [1, 2, 0.25]]])
    p2b = torch.tensor([[[0, 0, 0.75]]])
    pins = torch.zeros(1, 1, 2)
    tree = torch.zeros(1, 2, 3)
    fp_sol = torch.tensor(
        [[[4, 2, 0, 0], [4, 2, 4, 0], [2, 2, 0, 2]]],
        dtype=torch.float32,
    )
    metrics = torch.zeros(1, 8)
    case = case_from_current(
        (raw, b2b, p2b, pins, tree, fp_sol, metrics),
        0,
    )
    assert case["areas"].tolist() == [8.0, 8.0, 4.0]
    expected = raw[0, :, 1:].clone()
    expected[:, 2] = torch.tensor([1.0, 1.0, 0.0])
    assert case["constraints"].tolist() == expected.tolist()
    assert case["b2b"].tolist() == b2b[0].tolist()


def test_current_layout_rejects_noncontiguous_invalid_blocks() -> None:
    raw = torch.tensor(
        [
            [
                [0, 1, 0, 0, 0, 0],
                [8, 0, 0, 0, 0, 1],
            ]
        ],
        dtype=torch.float32,
    )
    zeros = torch.zeros(1, 1, 3)
    fp_sol = torch.tensor([[[4, 2, 0, 0], [4, 2, 4, 0]]], dtype=torch.float32)
    with pytest.raises(ValueError, match="contiguous"):
        case_from_current(
            (raw, zeros, zeros, zeros[..., :2], zeros, fp_sol, zeros),
            0,
        )


def test_split_selection_is_deterministic_and_disjoint() -> None:
    counts = list(range(21, 121)) * 2
    first = select_legacy_splits(
        counts,
        dev_per_count=100,
        dev_random=0,
        train_size=200,
        layouts_per_shard=100,
    )
    second = select_legacy_splits(
        counts,
        dev_per_count=100,
        dev_random=0,
        train_size=200,
        layouts_per_shard=100,
    )
    assert first[:2] == second[:2]
    assert not set(first[0]).intersection(first[1])
    assert len(first[0]) == 200
    assert len(first[1]) == 10000


def test_packed_cache_round_trip_preserves_fields(tmp_path) -> None:
    columns = _legacy_columns()
    cases = [case_from_legacy(columns, 0) for _ in range(2)]
    cases[1]["b2b"] = cases[1]["b2b"][:1]
    metadata = write_packed_split(cases, tmp_path)
    store = PackedCaseStore(tmp_path)
    batch = store.batch(store.indices(3))
    assert metadata["cases"] == 2
    assert batch["boxes"].shape == (2, 3, 4)
    assert batch["constraints"].shape == (2, 3, 5)
    assert batch["b2b"].shape == (2, 2, 3)
    assert batch["b2b"][1, 1].tolist() == [0.0, 0.0, 0.0]
    assert len(store) == 2


def test_packed_cache_rejects_metadata_count_mismatch(tmp_path) -> None:
    cases = [case_from_legacy(_legacy_columns(), 0)]
    write_packed_split(cases, tmp_path)
    path = tmp_path / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata count"):
        PackedCaseStore(tmp_path)
