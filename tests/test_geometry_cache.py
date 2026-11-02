from collections import Counter
from pathlib import Path

import numpy as np

from floorflow.data.geometry_cache import (
    build_floorplan_geometry,
    geometry_cache_path,
    load_geometry_cache,
    unpack_polygons,
    write_geometry_cache,
)
from floorflow.geometry.geometry_graph import wall_plan_from_geometry
from floorflow.evaluation.fpeval_bridge import to_fpeval_plan
from fpeval.verify import verify
from floorflow.data.taxonomy import build_taxonomy


SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 80">
  <g class="Floorplan">
    <g id="Floor-1" transform="translate(2,3)">
      <g class="Space Bedroom">
        <polygon points="8,7 48,7 48,67 8,67"/>
      </g>
      <g class="Space Kitchen">
        <polygon points="48,7 88,7 88,67 48,67"/>
      </g>
      <g id="Wall" class="Wall External">
        <polygon points="8,7 88,7 88,12 8,12"/>
        <polygon points="8,62 88,62 88,67 8,67"/>
        <polygon points="8,7 13,7 13,67 8,67"/>
        <polygon points="83,7 88,7 88,67 83,67"/>
      </g>
      <g id="Wall" class="Wall">
        <polygon points="46,7 50,7 50,67 46,67"/>
        <g id="Door" class="Door Swing Beside">
          <polygon points="46,53 50,53 50,61 46,61"/>
        </g>
      </g>
      <g class="FixedFurniture Sink">
        <g class="BoundaryPolygon">
          <polygon points="65,20 78,20 78,28 65,28"/>
        </g>
      </g>
    </g>
  </g>
</svg>
"""


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Kitchen": 4}),
            "furniture": Counter({"Sink": 4}),
            "door": Counter({"Swing Beside": 4}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _geometry(tmp_path: Path):
    path = tmp_path / "model.svg"
    path.write_text(SVG, encoding="utf-8")
    return build_floorplan_geometry(
        path,
        floor_id="Floor-1",
        image_size=128,
        taxonomy=_taxonomy(),
    )


def test_geometry_cache_preserves_instances_and_masks(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    assert geometry.target.shape == (128, 128)
    assert len(unpack_polygons(geometry.rooms)) == 2
    assert len(unpack_polygons(geometry.walls)) == 5
    assert len(unpack_polygons(geometry.openings)) == 1
    assert len(unpack_polygons(geometry.furniture)) == 1
    assert geometry.wall_centerlines.shape == (5, 2, 2)
    assert np.all(geometry.wall_thickness > 0)
    assert 0 <= geometry.opening_wall_indices[0] < 5
    visible_rooms = np.isin(geometry.target, (3, 4))
    assert geometry.room_support.sum() >= visible_rooms.sum()
    assert geometry.occlusion_mask.sum() > 0
    assert np.all(geometry.occlusion_mask[visible_rooms] == 0)


def test_geometry_cache_roundtrip_and_no_overwrite(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    path = geometry_cache_path(
        tmp_path / "cache",
        "test",
        "case",
        "Floor-1",
    )
    assert write_geometry_cache(geometry, path)
    assert not write_geometry_cache(geometry, path)
    loaded = load_geometry_cache(path)
    assert loaded["cache_version"].item() == "geometry-v1"
    assert np.array_equal(loaded["target"], geometry.target)
    assert np.array_equal(loaded["room_support"], geometry.room_support)
    assert np.array_equal(loaded["room_offsets"], geometry.rooms.offsets)


def test_wall_graph_keeps_source_rooms_and_verifies(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    plan = to_fpeval_plan(
        wall_plan_from_geometry(geometry, _taxonomy())
    )
    result = verify(plan)
    assert result.valid, result.issues
    assert result.watertight, result.issues
    assert len(plan.rooms) == 2
