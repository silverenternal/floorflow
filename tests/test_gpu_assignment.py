from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from floorflow.objectives.gpu_assignment import (
    assignment_index_tensors,
    exact_assignment_batch,
    padded_exact_assignment_batch,
)


def test_exact_assignment_batch_matches_scipy_for_ragged_costs() -> None:
    torch.manual_seed(5)
    costs = [torch.rand(3, 2), torch.rand(2, 4), torch.empty(0, 3)]
    actual = exact_assignment_batch(costs)
    for cost, (actual_rows, actual_columns) in zip(costs, actual):
        if not cost.numel():
            assert actual_rows.size == actual_columns.size == 0
            continue
        expected_rows, expected_columns = linear_sum_assignment(
            cost.numpy()
        )
        assert np.array_equal(actual_rows, expected_rows)
        assert np.array_equal(actual_columns, expected_columns)


def test_assignment_index_tensors_pack_once_and_mask_padding() -> None:
    assignments = [
        (np.array([2, 0]), np.array([1, 3])),
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
        (np.array([4]), np.array([0])),
    ]
    rows, columns, valid = assignment_index_tensors(
        assignments,
        device=torch.device("cpu"),
    )
    assert rows.shape == columns.shape == (3, 2)
    assert valid.tolist() == [[True, True], [False, False], [True, False]]
    assert rows[0, :2].tolist() == [2, 0]
    assert columns[0, :2].tolist() == [1, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_exact_assignment_batch_cuda_contract() -> None:
    costs = [
        torch.tensor([[4.0, 1.0], [0.0, 5.0]], device="cuda:0"),
        torch.tensor([[2.0]], device="cuda:0"),
    ]
    assignments = exact_assignment_batch(costs)
    assert assignments[0][0].tolist() == [0, 1]
    assert assignments[0][1].tolist() == [1, 0]
    assert assignments[1][0].tolist() == [0]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_padded_exact_assignment_matches_rectangular_scipy(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(17)
    costs = [
        torch.rand(5, 3, device=device),
        torch.rand(2, 7, device=device),
        torch.empty(0, 4, device=device),
    ]
    actual = padded_exact_assignment_batch(costs)
    for cost, (actual_rows, actual_columns) in zip(costs, actual):
        if not cost.numel():
            assert actual_rows.size == actual_columns.size == 0
            continue
        expected_rows, expected_columns = linear_sum_assignment(
            cost.cpu().numpy()
        )
        actual_pair = dict(zip(actual_rows.tolist(), actual_columns.tolist()))
        expected_pair = dict(
            zip(expected_rows.tolist(), expected_columns.tolist())
        )
        assert actual_pair == expected_pair
