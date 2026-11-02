from __future__ import annotations

import torch

from floorflow.data.local_control_edits import (
    edit_furniture_delete,
    edit_opening_delete,
    edit_room_add,
    edit_room_remove,
    edit_room_type,
)
from floorflow.data.taxonomy import FloorplanTaxonomy


def _controls(taxonomy: FloorplanTaxonomy) -> dict[str, torch.Tensor]:
    room_count = len(taxonomy.room_ids)
    controls = {
        "room_logits": torch.zeros(room_count, 32, 32),
        "program": torch.zeros(room_count, dtype=torch.bool),
        "structure": torch.zeros(
            room_count + len(taxonomy.opening_ids) + 3,
            32,
            32,
        ),
        "furniture": torch.zeros(
            room_count + len(taxonomy.furniture_ids) + 1,
            32,
            32,
        ),
        "support": torch.zeros(room_count + 4, 32, 32),
    }
    return controls


def test_room_type_edit_changes_one_component() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    controls = _controls(taxonomy)
    controls["room_logits"][0, 2:9, 2:9] = 4.0
    controls["room_logits"][1, 20:27, 20:27] = 4.0
    controls["room_logits"][2, 10:17, 20:27] = 4.0
    edit = edit_room_type(controls, taxonomy, seed=3)
    assert edit is not None
    changed = edit.controls["room_logits"] != controls["room_logits"]
    assert int(changed.sum()) > 0
    assert not torch.equal(edit.controls["room_logits"], controls["room_logits"])


def test_opening_and_furniture_deletes_are_local() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    room_count = len(taxonomy.room_ids)
    controls = _controls(taxonomy)
    controls["structure"][room_count + 1, 4:6, 4:6] = 1.0
    opening = edit_opening_delete(controls, taxonomy, seed=1)
    assert opening is not None
    assert float(opening.controls["structure"][room_count + 1, 4, 4]) == 0.0
    controls = _controls(taxonomy)
    controls["furniture"][room_count, 10:14, 10:14] = 1.0
    furniture = edit_furniture_delete(controls, taxonomy, seed=1)
    assert furniture is not None
    assert float(furniture.controls["furniture"][room_count, 10, 10]) == 0.0


def test_room_add_and_remove_update_program() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    controls = _controls(taxonomy)
    controls["room_logits"][0, 4:20, 4:20] = 4.0
    controls["room_logits"][1, 4:20, 4:20] = 5.0
    controls["program"][0] = True
    controls["program"][1] = True
    added = edit_room_add(controls, taxonomy, seed=2)
    assert added is not None
    assert int(added.controls["program"].sum()) == int(controls["program"].sum()) + 1
    removed = edit_room_remove(controls, taxonomy, seed=2)
    assert removed is not None
    assert int(removed.controls["program"].sum()) < int(controls["program"].sum())
