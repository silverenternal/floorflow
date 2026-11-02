from __future__ import annotations

import json
from collections import Counter

import numpy as np

from floorflow.data.synthetic_data import SyntheticFloorplanDataset, convert_fpgen_gt
from floorflow.data.taxonomy import build_taxonomy


def _payload() -> dict:
    return {
        "meta": {"img_w": 1024, "img_h": 768},
        "rooms": [
            {
                "label": "BEDROOM 1",
                "polygon": [[100, 100], [500, 100], [500, 650], [100, 650]],
            }
        ],
        "walls": [
            {
                "start": [100, 100],
                "end": [500, 100],
                "thickness": 20,
                "openings": [
                    {"type": "window", "center": [300, 100], "width": 80},
                ],
            }
        ],
        "fixtures": [
            {
                "kind": "toilet",
                "quad": [[200, 200], [260, 200], [260, 280], [200, 280]],
            }
        ],
    }


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter({"Toilet": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_fpgen_converter_preserves_room_and_objects(tmp_path) -> None:
    taxonomy = _taxonomy()
    path = tmp_path / "sample.gt.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    target, footprint = convert_fpgen_gt(path, taxonomy, image_size=128)
    assert footprint.any()
    assert (target == taxonomy.class_id("room:Bedroom")).any()
    assert (target == taxonomy.class_id("furniture:Toilet")).any()
    assert (target == taxonomy.class_id("window:Regular")).any()


def test_synthetic_dataset_interface(tmp_path) -> None:
    taxonomy = _taxonomy()
    taxonomy.write(tmp_path / "taxonomy.json")
    (tmp_path / "train").mkdir()
    target = np.zeros((128, 128), dtype=np.uint8)
    target[10:100, 10:100] = taxonomy.class_id("room:Bedroom")
    footprint = np.zeros((128, 128), dtype=np.uint8)
    footprint[10:100, 10:100] = 1
    np.savez_compressed(tmp_path / "train" / "000000.npz", target=target, footprint=footprint)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"splits": {"train": ["train/000000.npz"]}}),
        encoding="utf-8",
    )
    dataset = SyntheticFloorplanDataset(tmp_path, "train")
    item_target, item_footprint, floor_id, name = dataset[0]
    assert item_target.shape == (128, 128)
    assert item_footprint.sum() == footprint.sum()
    assert floor_id == 1
    assert name == "000000"
