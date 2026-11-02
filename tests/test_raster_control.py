from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from floorflow.data.dataset import (
    FloorplanSample,
    filter_unsupervised_samples,
    room_instance_targets,
)
from floorflow.data.raster_control import (
    cubicasa_floor_crops,
    load_cubicasa_floor_rgb,
    non_room_control_labels,
)
from floorflow.data.taxonomy import FloorplanTaxonomy


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "room:LivingRoom",
            "room:Bedroom",
            "door:Swing Beside",
        ),
        room_map={"LivingRoom": "room:LivingRoom", "Bedroom": "room:Bedroom"},
        furniture_map={},
        door_map={"Swing Beside": "door:Swing Beside"},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def test_non_room_control_removes_all_room_class_pixels() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((12, 12), dtype=np.int64)
    footprint = np.ones_like(labels, dtype=bool)
    labels[1:5, 1:5] = taxonomy.class_id("room:LivingRoom")
    labels[6:10, 6:10] = taxonomy.class_id("room:Bedroom")
    labels[5, 2:10] = taxonomy.class_id("wall")
    labels[2:4, 7:9] = taxonomy.class_id("door:Swing Beside")
    labels[0, 0] = taxonomy.class_id("room:Bedroom")
    footprint[0, 0] = False

    controls = non_room_control_labels(labels, footprint, taxonomy)

    assert not np.isin(controls, taxonomy.room_ids).any()
    assert controls[5, 5] == taxonomy.class_id("wall")
    assert controls[3, 7] == taxonomy.class_id("door:Swing Beside")
    assert controls[0, 0] == 0


def test_page_floor_detection_selects_target_free_drawing(tmp_path) -> None:
    image = Image.new("L", (320, 180), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 100, 170), outline=0, width=3)
    draw.rectangle((210, 20, 309, 160), outline=0, width=3)
    path = tmp_path / "page.png"
    image.convert("RGB").save(path)

    boxes = cubicasa_floor_crops(path, 2)
    assert len(boxes) == 2
    assert boxes[0][0] < boxes[1][0]
    selected = load_cubicasa_floor_rgb(
        path,
        floor_index=1,
        floor_count=2,
    )
    assert selected.shape[0] > 0 and selected.shape[1] > 0


def test_unsupervised_cache_record_is_removed(tmp_path) -> None:
    valid = tmp_path / "valid.npz"
    empty = tmp_path / "empty.npz"
    np.savez_compressed(
        valid,
        target=np.array([[0, 2], [2, 0]], dtype=np.uint8),
        footprint=np.ones((2, 2), dtype=np.uint8),
    )
    np.savez_compressed(
        empty,
        target=np.zeros((2, 2), dtype=np.uint8),
        footprint=np.ones((2, 2), dtype=np.uint8),
    )
    samples = [
        FloorplanSample(Path("case"), Path("model.svg"), valid),
        FloorplanSample(Path("case"), Path("model.svg"), empty),
    ]
    dataset = type(
        "Dataset",
        (),
        {
            "samples": samples,
            "image_size": 2,
            "taxonomy": _taxonomy(),
        },
    )()
    removed = filter_unsupervised_samples(dataset)
    assert removed == ["case__Floor-1"]
    assert dataset.samples == samples[:1]


def test_room_instance_targets_are_built_without_device_round_trip() -> None:
    taxonomy = _taxonomy()
    target = np.zeros((16, 16), dtype=np.int64)
    target[1:8, 1:8] = taxonomy.class_id("room:LivingRoom")
    target[9:15, 9:15] = taxonomy.class_id("room:Bedroom")
    classes, masks, valid, target_counts = room_instance_targets(
        target,
        taxonomy,
        max_rooms=4,
    )
    assert classes.device.type == "cpu"
    assert masks.device.type == "cpu"
    assert masks.dtype == torch.uint8
    assert valid.tolist() == [True, True, False, False]
    assert target_counts.item() == 2
    assert masks.shape == (4, 16, 16)
