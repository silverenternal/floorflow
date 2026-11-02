from __future__ import annotations

from pathlib import Path

from floorflow.data.raster_vector_geometry_dataset import (
    RasterVectorGeometryDataset,
)

ROOT = Path(__file__).resolve().parents[1]


def test_native_raster_is_paired_with_vector_targets() -> None:
    dataset = RasterVectorGeometryDataset(
        ROOT / "outputs/cache/cubicasa5k_geometry_v1",
        ROOT / ".data/cubicasa5k",
        "val",
        limit=1,
    )
    item = dataset[0]
    assert item["image"].shape == (3, 256, 256)
    assert item["target"].shape == (256, 256)
    assert "room_polygons" in item

    shifted = RasterVectorGeometryDataset(
        ROOT / "outputs/cache/cubicasa5k_geometry_v1",
        ROOT / ".data/cubicasa5k",
        "val",
        limit=1,
        offset=1,
    )
    assert dataset[0]["name"] != shifted[0]["name"]
