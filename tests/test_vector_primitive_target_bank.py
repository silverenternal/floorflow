"""Tests for canonical vector-target case indexing."""

from pathlib import Path

import pytest
import torch

from floorflow.data.vector_primitive_target_bank import (
    case_name_from_geometry,
    opening_host_targets,
)


def test_case_name_matches_predicted_control_schema() -> None:
    relative = Path("colorful/11261/Floor_2.geometry.npz")
    assert (
        case_name_from_geometry(relative)
        == "colorful__11261__Floor-2"
    )


def test_opening_host_targets_choose_nearest_valid_wall() -> None:
    walls = torch.tensor(
        [
            [
                [[0.0, 0.2], [1.0, 0.2]],
                [[0.8, 0.0], [0.8, 1.0]],
            ]
        ]
    )
    wall_exists = torch.tensor([[1.0, 1.0]])
    centers = torch.tensor([[[0.5, 0.8]]])
    opening_exists = torch.tensor([[1.0]])

    hosts, intervals = opening_host_targets(
        walls,
        wall_exists,
        centers,
        opening_exists,
    )

    assert hosts.item() == 1
    assert intervals.item() == pytest.approx(0.8)
