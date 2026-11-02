import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.support_structure_fusion import (
    apply_room_support_to_visible_plan,
    fuse_support_deletion_with_structure,
)
from floorflow.representation.layered_floorplan import LayeredFloorplan


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "room:LivingRoom",
            "door:Door",
        ),
        room_map={"LivingRoom": "room:LivingRoom"},
        furniture_map={},
        door_map={"Door": "door:Door"},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def test_deleted_support_is_absorbed_by_structure_evidence() -> None:
    taxonomy = _taxonomy()
    footprint = torch.ones(1, 1, 2, 3, dtype=torch.bool)
    base = LayeredFloorplan(
        room_labels=torch.full((1, 2, 3), 2),
        wall=torch.zeros(1, 2, 3, dtype=torch.long),
        opening=torch.zeros(1, 2, 3, dtype=torch.long),
        furniture=torch.zeros(1, 2, 3, dtype=torch.long),
        footprint=footprint[:, 0],
        taxonomy=taxonomy,
    )
    gated = base.room_labels.clone()
    gated[0, 1, 1:] = 0
    structure = {
        "wall_logits": torch.full((1, 1, 2, 3), 4.0),
        "opening_logits": torch.tensor([[[[0.0, 9.0]]]]).repeat(1, 1, 2, 1),
        "opening_type_logits": torch.zeros(1, 1, 2, 3),
    }
    fused = fuse_support_deletion_with_structure(
        base,
        gated,
        structure,
        footprint,
        taxonomy,
        wall_threshold=0.5,
        opening_threshold=0.5,
    )
    assert fused.room_labels[0, 1, 1].item() == 0
    assert fused.wall[0, 1, 1].item() == 1
    assert fused.opening[0, 1, 1].item() == taxonomy.opening_ids[0]


def test_room_support_cleanup_preserves_non_room_plan_layers() -> None:
    taxonomy = _taxonomy()
    labels = torch.tensor(
        [
            [
                [2, 2, 1],
                [2, 1, taxonomy.opening_ids[0]],
            ]
        ]
    )
    keep = torch.tensor([[[True, False, False], [True, False, False]]])
    output = apply_room_support_to_visible_plan(
        labels,
        keep,
        torch.tensor(taxonomy.room_ids),
    )
    assert output.tolist() == [[[2, 0, 1], [2, 1, taxonomy.opening_ids[0]]]]
