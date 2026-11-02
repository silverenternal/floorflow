from __future__ import annotations

from pathlib import Path

import numpy as np

from floorflow.data.classes import (
    BACKGROUND,
    DOOR,
    FURNITURE_OTHER,
    LIVING,
    NUM_TARGET_CHANNELS,
    WALL,
    semantic_channel_prior,
)
from floorflow.data.svg import _parse_path_points, rasterize_cubicasa_svg
from floorflow.representation.masks import fill_enclosed_regions


def test_rasterize_cubicasa_semantics_and_footprint(tmp_path: Path) -> None:
    svg = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 80">
  <g class="Floorplan">
    <g id="room" class="Space LivingRoom">
      <polygon points="10,10 90,10 90,70 10,70"/>
    </g>
    <g id="Wall">
      <polygon points="10,10 90,10 90,14 10,14"/>
      <g id="Door"><polygon points="40,10 55,10 55,14 40,14"/></g>
    </g>
    <g class="FixedFurniture Sofa" transform="matrix(1,0,0,1,30,30)">
      <g class="BoundaryPolygon"><polygon points="0,0 20,0 20,10 0,10"/></g>
    </g>
  </g>
</svg>
"""
    path = tmp_path / "model.svg"
    path.write_text(svg, encoding="utf-8")
    target, footprint = rasterize_cubicasa_svg(path, image_size=128)
    assert target.shape == footprint.shape == (128, 128)
    assert footprint.mean() > 0.3
    assert set(np.unique(target)).issuperset({WALL, DOOR, LIVING, FURNITURE_OTHER})
    assert target[0, 0] == BACKGROUND
    assert footprint[0, 0] == 0
    native_target, native_footprint = rasterize_cubicasa_svg(
        path,
        output_shape=(80, 120),
    )
    assert native_target.shape == native_footprint.shape == (80, 120)


def test_fill_outline_region() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:28, 6:26] = 0
    mask[4, 6:26] = 1
    mask[27, 6:26] = 1
    mask[4:28, 6] = 1
    mask[4:28, 25] = 1
    filled = fill_enclosed_regions(mask)
    assert filled[16, 16] == 1
    assert filled[0, 0] == 0


def test_parse_svg_path_endpoints() -> None:
    points = _parse_path_points("M0,0 L10,0 Q10,10 0,10 Z")
    assert points.shape[0] >= 3
    assert np.allclose(points[:2], [[0, 0], [10, 0]])
    partial = _parse_path_points("M0,0 L10,0 LNaN,NaN")
    assert np.allclose(partial, [[0, 0], [10, 0]])


def test_semantic_channel_prior_shape_and_symmetry() -> None:
    prior = semantic_channel_prior()
    assert len(prior) == NUM_TARGET_CHANNELS
    assert all(len(row) == NUM_TARGET_CHANNELS for row in prior)
    assert prior == tuple(zip(*prior))
