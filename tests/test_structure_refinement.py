from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.geometry_heads import wall_junction_targets
from floorflow.objectives.control_repair import (
    structure_losses,
    wall_centerline_target,
)
from floorflow.objectives.structure_refinement import (
    predicted_opening_wall_containment,
    room_boundary_target,
    typed_structure_relation_loss,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Bath": 2}),
            "furniture": Counter(),
            "door": Counter({"Door": 2}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_wall_junction_targets_are_finite() -> None:
    wall = torch.zeros(1, 1, 16, 16)
    wall[:, :, 8, 2:14] = 1.0
    wall[:, :, 2:14, 8] = 1.0
    footprint = torch.ones(1, 16, 16)
    junction, degree = wall_junction_targets(wall, footprint)
    assert junction.shape == (1, 1, 16, 16)
    assert degree.shape == (1, 16, 16)
    assert torch.isfinite(junction).all()
    assert degree.max() >= 3


def test_wall_centerline_target_is_thin_and_supervised() -> None:
    wall = torch.zeros(1, 1, 16, 16)
    wall[:, :, 7:10, 2:14] = 1.0
    wall[:, :, 6:11, 7:10] = 1.0
    centerline = wall_centerline_target(wall)
    assert centerline.shape == wall.shape
    assert 0 < centerline.sum() < wall.sum()
    predictions = {
        "wall_logits": torch.randn(1, 1, 16, 16, requires_grad=True),
        "opening_logits": torch.randn(1, 1, 16, 16, requires_grad=True),
        "opening_type_logits": torch.randn(1, 2, 16, 16),
        "centerline_logits": torch.randn(
            1,
            1,
            16,
            16,
            requires_grad=True,
        ),
    }
    target = torch.zeros(1, 5, 16, 16)
    target[:, 2:3] = wall
    loss, components = structure_losses(
        predictions,
        target,
        torch.ones(1, 16, 16, dtype=torch.bool),
        room_count=2,
        opening_ids=torch.tensor([0, 1]),
        centerline_weight=0.5,
    )
    assert torch.isfinite(loss)
    assert components["wall_centerline"] > 0
    loss.backward()
    assert predictions["centerline_logits"].grad is not None


def test_opening_wall_containment_resizes_to_footprint_resolution() -> None:
    predictions = {
        "opening_logits": torch.randn(1, 1, 8, 8, requires_grad=True),
        "wall_logits": torch.randn(1, 1, 8, 8, requires_grad=True),
    }
    loss = predicted_opening_wall_containment(
        predictions,
        torch.ones(1, 32, 32),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert predictions["opening_logits"].grad is not None


def test_typed_relation_resizes_footprint_for_junction_targets() -> None:
    taxonomy = _taxonomy()
    room_count = len(taxonomy.room_ids)
    wall = torch.zeros(1, 1, 16, 16)
    wall[:, :, 8, 2:14] = 1.0
    labels = torch.full((1, 32, 32), taxonomy.room_ids[0], dtype=torch.long)
    labels[:, :, 16:] = taxonomy.room_ids[1]
    footprint = torch.ones(1, 32, 32)
    target = torch.cat(
        [
            torch.nn.functional.one_hot(
                torch.zeros(1, 16, 16, dtype=torch.long),
                room_count,
            ).permute(0, 3, 1, 2).float(),
            wall,
            torch.zeros(1, 1, 16, 16),
            torch.zeros(1, len(taxonomy.opening_ids), 16, 16),
            torch.ones(1, 1, 16, 16),
        ],
        dim=1,
    )
    predictions = {
        "wall_logits": torch.randn(1, 1, 16, 16),
        "opening_logits": torch.randn(1, 1, 16, 16),
        "opening_type_logits": torch.randn(
            1,
            len(taxonomy.opening_ids),
            16,
            16,
        ),
        "junction_logits": torch.randn(1, 1, 16, 16),
        "junction_degree_logits": torch.randn(1, 5, 16, 16),
        "room_boundary_logits": torch.randn(1, 1, 16, 16),
    }
    loss, _ = typed_structure_relation_loss(
        predictions,
        torch.randn(1, room_count, 16, 16),
        target,
        labels,
        footprint,
        taxonomy,
        room_count=room_count,
    )
    assert torch.isfinite(loss)


def test_typed_structure_relation_loss_reaches_all_heads() -> None:
    taxonomy = _taxonomy()
    room_count = len(taxonomy.room_ids)
    wall = torch.zeros(1, 1, 16, 16)
    wall[:, :, 8, 2:14] = 1.0
    footprint = torch.ones(1, 16, 16)
    labels = torch.full((1, 16, 16), taxonomy.room_ids[0], dtype=torch.long)
    labels[:, :, 8:] = taxonomy.room_ids[1]
    labels[:, 8, 2:14] = taxonomy.class_id("wall")
    labels[:, 12:14, 7:9] = taxonomy.opening_ids[0]
    predictions = {
        "wall_logits": torch.randn(1, 1, 16, 16, requires_grad=True),
        "opening_logits": torch.randn(1, 1, 16, 16, requires_grad=True),
        "opening_type_logits": torch.randn(
            1,
            len(taxonomy.opening_ids),
            16,
            16,
            requires_grad=True,
        ),
        "junction_logits": torch.randn(1, 1, 16, 16, requires_grad=True),
        "junction_degree_logits": torch.randn(
            1,
            5,
            16,
            16,
            requires_grad=True,
        ),
        "room_boundary_logits": torch.randn(
            1,
            1,
            16,
            16,
            requires_grad=True,
        ),
    }
    room_logits = torch.randn(1, room_count, 16, 16, requires_grad=True)
    structure_target = torch.cat(
        [
            torch.nn.functional.one_hot(
                torch.zeros(1, 16, 16, dtype=torch.long),
                room_count,
            ).permute(0, 3, 1, 2).float(),
            wall,
            torch.zeros(1, 1, 16, 16),
            torch.zeros(1, len(taxonomy.opening_ids), 16, 16),
            footprint[:, None],
        ],
        dim=1,
    )
    loss, components = typed_structure_relation_loss(
        predictions,
        room_logits,
        structure_target,
        labels,
        footprint,
        taxonomy,
        room_count=room_count,
    )
    assert torch.isfinite(loss)
    assert components["structure_junction_loss"] > 0
    loss.backward()
    assert predictions["junction_logits"].grad is not None
    assert predictions["room_boundary_logits"].grad is not None
    boundary = room_boundary_target(labels, footprint, taxonomy.room_ids)
    assert boundary.shape == (1, 1, 16, 16)
    assert boundary.max() == 1
