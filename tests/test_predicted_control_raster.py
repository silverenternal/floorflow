"""Tests for cached predicted non-room control rasters."""

from pathlib import Path

import numpy as np
import torch

from floorflow.data.predicted_control_raster import (
    predicted_control_raster_path,
    predicted_furniture_probability,
)


def test_predicted_control_path_uses_split_and_sample_key(
    tmp_path: Path,
) -> None:
    path = predicted_control_raster_path(tmp_path, "val", "case/floor__F-1")
    assert path == tmp_path / "val" / "case__floor__F-1.npz"


def test_predicted_control_payload_has_expected_shape(
    tmp_path: Path,
) -> None:
    path = predicted_control_raster_path(tmp_path, "val", "case__F-1")
    path.parent.mkdir(parents=True)
    image = np.zeros((3, 256, 256), dtype=np.float16)
    np.savez_compressed(path, image=image)
    with np.load(path) as payload:
        assert payload["image"].shape == (3, 256, 256)
        assert payload["image"].dtype == np.float16


def test_furniture_probability_excludes_room_and_footprint_channels() -> None:
    furniture = torch.zeros(1, 5, 4, 4)
    furniture[:, 0] = 1.0
    furniture[:, 3, 1:3, 1:3] = 0.8
    furniture[:, -1] = 1.0
    probability = predicted_furniture_probability(furniture, 2)
    assert torch.allclose(
        probability[0, 1:3, 1:3].mean(),
        torch.tensor(0.8),
    )
    assert float(probability[0, :, 0].mean()) == 0.0
