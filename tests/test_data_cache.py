from pathlib import Path

import numpy as np
import pytest

from floorflow.data.dataset import FloorplanSample, load_or_rasterize


def test_resolution_mismatch_never_overwrites_shared_cache(tmp_path: Path) -> None:
    cache = tmp_path / "Floor_1.npz"
    original_target = np.ones((4, 4), dtype=np.uint8)
    original_footprint = np.ones((4, 4), dtype=np.uint8)
    np.savez_compressed(
        cache,
        target=original_target,
        footprint=original_footprint,
    )
    sample = FloorplanSample(
        relative_case=Path("case"),
        svg_path=tmp_path / "missing.svg",
        cache_path=cache,
        floor_id="Floor-1",
    )
    with pytest.raises(ValueError, match="cache resolution mismatch"):
        load_or_rasterize(sample, image_size=8, write_cache=True)
    with np.load(cache) as payload:
        assert payload["target"].shape == (4, 4)
        assert np.array_equal(payload["target"], original_target)
