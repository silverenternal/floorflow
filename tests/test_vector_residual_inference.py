"""Tests for edit-local vector residual inference."""

from __future__ import annotations

import numpy as np
import torch
from shapely.geometry import Polygon

from floorflow.evaluation.vector_edit_protocol import (
    VectorOpening,
    VectorPlan,
    VectorRoom,
    VectorWall,
)
from floorflow.heads.vector_proposal_refiner import VectorProposalRefiner
from floorflow.inference.vector_residual_refiner import refine_vector_plan


def test_refinement_only_changes_edit_intersecting_primitives() -> None:
    mask_edit = np.zeros((32, 32), dtype=bool)
    mask_edit[5:15, 5:15] = True
    mask_static = np.zeros((32, 32), dtype=bool)
    mask_static[20:30, 20:30] = True
    plan = VectorPlan(
        rooms=(
            VectorRoom("room:LivingRoom", Polygon([(4, 4), (14, 4), (14, 14), (4, 14)]), mask_edit),
            VectorRoom("room:Bedroom", Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]), mask_static),
        ),
        walls=(
            VectorWall((5, 5), (5, 14), 1.0, "interior"),
            VectorWall((20, 20), (20, 30), 1.0, "interior"),
        ),
        openings=(
            VectorOpening((10, 10), 2.0, "door"),
            VectorOpening((25, 25), 2.0, "window"),
        ),
        wall_mask=np.zeros((32, 32), dtype=bool),
        opening_mask=np.zeros((32, 32), dtype=bool),
        source=None,
    )
    refiner = VectorProposalRefiner(in_channels=8, hidden_dim=16)
    refiner.room_residual.bias.data.fill_(1.0)
    refiner.wall_residual.bias.data.fill_(1.0)
    refiner.opening_residual.bias.data.fill_(1.0)
    result = refine_vector_plan(
        plan,
        mask_edit,
        torch.randn(1, 8, 32, 32),
        refiner,
    )

    assert result.affected_rooms == 1
    assert result.affected_walls == 1
    assert result.affected_openings == 1
    assert not result.plan.rooms[0].polygon.equals_exact(plan.rooms[0].polygon, 1.0e-9)
    assert result.plan.rooms[1].polygon.equals_exact(plan.rooms[1].polygon, 1.0e-9)
    assert result.plan.walls[1] == plan.walls[1]
    assert result.plan.openings[1] == plan.openings[1]
