from pathlib import Path

import pytest
import torch

from floorflow.training.raster_instance_evaluation import resolve_query_blend


def test_query_blend_prefers_checkpoint_selection(tmp_path: Path) -> None:
    checkpoint = tmp_path / "query.pt"
    torch.save({"blend": 0.75}, checkpoint)
    weight, source = resolve_query_blend(checkpoint)
    assert weight == 0.75
    assert source == "checkpoint"


def test_query_blend_override_is_validated(tmp_path: Path) -> None:
    checkpoint = tmp_path / "query.pt"
    torch.save({"blend": 0.75}, checkpoint)
    weight, source = resolve_query_blend(checkpoint, 0.5)
    assert weight == 0.5
    assert source == "override"
    with pytest.raises(ValueError):
        resolve_query_blend(checkpoint, 1.1)
