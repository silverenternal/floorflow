from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from floorflow.data.svg import list_cubicasa_floors, rasterize_cubicasa_svg
from floorflow.data.taxonomy import build_taxonomy


def test_taxonomy_keeps_common_and_groups_rare_labels() -> None:
    counts = {
        "room": Counter({"Kitchen": 200, "Bar": 2}),
        "furniture": Counter({"Sink": 150, "Dishwasher": 1}),
        "door": Counter({"Swing Beside": 100}),
        "window": Counter({"Regular": 100}),
    }
    taxonomy = build_taxonomy(counts, room_min_count=100, furniture_min_count=100)
    assert "room:Kitchen" in taxonomy.class_names
    assert "room_rare" in taxonomy.class_names
    assert "furniture:Sink" in taxonomy.class_names
    assert "furniture_rare" in taxonomy.class_names
    assert taxonomy.room_map["Bar"] == "room_rare"


def test_floor_selection_does_not_overlay_storeys(tmp_path: Path) -> None:
    svg = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <g class="Floor">
    <g id="Floor-1" class="Floorplan Floor-1">
      <g class="Space Kitchen"><polygon points="10,10 90,10 90,50 10,50"/></g>
    </g>
    <g id="Floor-2" class="Floorplan Floor-2">
      <g class="Space Bedroom"><polygon points="10,55 90,55 90,90 10,90"/></g>
    </g>
  </g>
</svg>
"""
    path = tmp_path / "model.svg"
    path.write_text(svg, encoding="utf-8")
    assert list_cubicasa_floors(path) == ["Floor-1", "Floor-2"]
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Kitchen": 10, "Bedroom": 10}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    floor1, _ = rasterize_cubicasa_svg(path, image_size=128, floor_id="Floor-1", taxonomy=taxonomy)
    floor2, _ = rasterize_cubicasa_svg(path, image_size=128, floor_id="Floor-2", taxonomy=taxonomy)
    kitchen = taxonomy.class_id("room:Kitchen")
    bedroom = taxonomy.class_id("room:Bedroom")
    assert np.any(floor1 == kitchen)
    assert not np.any(floor1 == bedroom)
    assert np.any(floor2 == bedroom)
    assert not np.any(floor2 == kitchen)
