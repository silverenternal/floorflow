"""Tests for room-partition wall candidate generation."""

from collections import Counter

import numpy as np
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.cad_plan_metrics import cad_wall_plan_metrics
from floorflow.geometry.cad_wall_candidates import (
    footprint_boundary_candidates,
    room_boundary_candidates,
    wall_candidate_evidence,
)
from floorflow.inference.final_stage_wall_source import (
    final_stage_wall_segments,
)
from floorflow.model.cad_layers import CadLayeredOutput
from floorflow.training.cad_wall_candidate_training import (
    build_candidate_batch,
    compose_source_candidate_plans,
)


def test_room_boundaries_are_deduplicated_and_split() -> None:
    left = {
        "polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
    }
    right = {
        "polygon": [[5, 0], [10, 0], [10, 5], [5, 5]],
    }
    walls = room_boundary_candidates([left, right], minimum_length=0.5)
    shared = [wall for wall in walls if {wall.start, wall.end} == {(5.0, 0.0), (5.0, 5.0)}]
    assert len(shared) == 1
    assert len(walls) == 7


def test_candidate_evidence_samples_wall_probability() -> None:
    footprint = np.ones((16, 16), dtype=np.uint8)
    walls = footprint_boundary_candidates(footprint, minimum_length=1.0)
    probability = np.zeros((16, 16), dtype=np.float32)
    probability[0, :] = 1.0
    evidence = wall_candidate_evidence(walls, probability, samples=3)
    assert evidence.max() == 1.0
    assert evidence.min() < 1.0


def test_pixel_staircase_is_simplified_into_continuous_exterior_walls() -> None:
    staircase = [[0, y] for y in range(10)] + [[x, 10] for x in range(10)]
    walls = room_boundary_candidates(
        [{"polygon": staircase + [[10, 0], [0, 0]], "holes": []}],
        minimum_length=1.0,
    )
    vertical = [wall for wall in walls if wall.start[0] == wall.end[0] == 0.0]
    assert len(vertical) == 1
    assert abs(vertical[0].start[1] - vertical[0].end[1]) >= 9.0


def test_skeleton_portfolio_keeps_boundary_control_separate() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter({"Sink": 2}),
            "door": Counter({"Swing": 2}),
            "window": Counter({"Regular": 2}),
        },
        room_min_count=1,
    )
    room_count = len(taxonomy.room_ids)
    room_logits = torch.full((1, room_count, 16, 16), -4.0)
    room_logits[:, 0, :, :8] = 4.0
    room_logits[:, 1, :, 8:] = 4.0
    host_wall = torch.zeros(1, 1, 16, 16)
    host_wall[:, :, :2] = 1.0
    stage_wall = torch.zeros(1, 1, 16, 16)
    stage_wall[:, :, -2:] = 1.0
    openings = len(taxonomy.opening_ids)
    structure = torch.zeros(1, room_count + openings + 3, 16, 16)
    structure[:, room_count : room_count + 1] = host_wall
    layers = CadLayeredOutput(
        room_logits=room_logits,
        wall_logits=stage_wall * 8.0 - 4.0,
        opening_logits=torch.zeros(1, openings, 16, 16),
        furniture_logits=torch.zeros(
            1,
            len(taxonomy.furniture_ids),
            16,
            16,
        ),
    )
    controls = {
        "structure": structure,
    }
    candidates = build_candidate_batch(
        controls,
        {"cad_layers": layers},
        torch.ones(1, 16, 16),
        taxonomy,
        minimum_room_area=1,
        max_walls=128,
        device=torch.device("cpu"),
        candidate_mode="skeleton_portfolio",
    )
    assert candidates.valid.any()
    assert candidates.control_valid is not None
    assert candidates.control_endpoints is not None
    assert candidates.proposal_source is not None
    assert int(candidates.proposal_source.max()) >= 2
    assert 2 in set(candidates.proposal_source[candidates.valid].tolist())
    assert 5 in set(candidates.proposal_source[candidates.valid].tolist())
    assert not torch.equal(
        candidates.endpoints,
        candidates.control_endpoints,
    )


def test_final_stage_wall_source_is_deterministic_and_nonempty() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter({"Sink": 2}),
            "door": Counter({"Swing": 2}),
            "window": Counter({"Regular": 2}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    room_count = len(taxonomy.room_ids)
    room_logits = torch.full((1, room_count, 16, 16), -4.0)
    room_logits[:, 0, :, :8] = 4.0
    room_logits[:, 1, :, 8:] = 4.0
    stage_wall = torch.zeros(1, 1, 16, 16)
    stage_wall[:, :, :2] = 1.0
    output = {
        "cad_layers": CadLayeredOutput(
            room_logits=room_logits,
            wall_logits=stage_wall * 8.0 - 4.0,
            opening_logits=torch.zeros(
                1,
                len(taxonomy.opening_ids),
                16,
                16,
            ),
            furniture_logits=torch.zeros(
                1,
                len(taxonomy.furniture_ids),
                16,
                16,
            ),
        )
    }
    first = final_stage_wall_segments(
        output,
        torch.ones(1, 16, 16),
        taxonomy,
    )[0]
    second = final_stage_wall_segments(
        output,
        torch.ones(1, 16, 16),
        taxonomy,
    )[0]
    assert first
    assert first == second


def test_source_candidate_plans_are_repeatable() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter({"Sink": 2}),
            "door": Counter({"Swing": 2}),
            "window": Counter({"Regular": 2}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    room_count = len(taxonomy.room_ids)
    opening_count = len(taxonomy.opening_ids)
    room_logits = torch.full((1, room_count, 16, 16), -4.0)
    room_logits[:, 0, :, :8] = 4.0
    room_logits[:, 1, :, 8:] = 4.0
    wall_logits = torch.full((1, 1, 16, 16), -4.0)
    wall_logits[:, :, :2] = 4.0
    output = {
        "cad_layers": CadLayeredOutput(
            room_logits=room_logits,
            wall_logits=wall_logits,
            opening_logits=torch.zeros(1, opening_count, 16, 16),
            furniture_logits=torch.zeros(
                1,
                len(taxonomy.furniture_ids),
                16,
                16,
            ),
        )
    }
    structure = torch.zeros(1, room_count + opening_count + 3, 16, 16)
    controls = {"structure": structure}
    opening_probability = torch.zeros(1, 2, 16, 16)
    candidates = build_candidate_batch(
        controls,
        output,
        torch.ones(1, 16, 16),
        taxonomy,
        minimum_room_area=1,
        max_walls=128,
        device=torch.device("cpu"),
        candidate_mode="skeleton_portfolio",
    )
    first = compose_source_candidate_plans(
        candidates,
        opening_probability,
        16.0,
        taxonomy,
    )[0]
    second = compose_source_candidate_plans(
        candidates,
        opening_probability,
        16.0,
        taxonomy,
    )[0]
    assert first.walls
    assert [(wall.start, wall.end, wall.wall_type) for wall in first.walls] == [
        (wall.start, wall.end, wall.wall_type) for wall in second.walls
    ]
    metrics = cad_wall_plan_metrics(
        first,
        torch.zeros(16, 16, dtype=torch.long).numpy(),
        taxonomy,
    )
    assert 0.0 <= metrics["wall"] <= 1.0
    assert metrics["valid"] == 1.0
