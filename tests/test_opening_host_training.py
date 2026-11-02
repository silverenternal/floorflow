"""Tests for predicted opening-query host supervision."""

from collections import Counter

import pytest
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.geometry.wall_graph_readout import ReadoutWall
from floorflow.training.cad_wall_candidate_training import CandidateBatch
from floorflow.training.opening_host_training import (
    OpeningQueries,
    candidate_host_walls,
    detection_opening_queries,
    match_opening_host_targets,
    opening_query_maps,
    target_opening_host_targets,
    target_opening_queries,
)


def test_opening_query_maps_preserve_independent_foreground() -> None:
    foreground = torch.full((1, 8, 8), 0.8)
    family_logits = torch.full((1, 2, 8, 8), -4.0)
    family_logits[:, 0] = 4.0
    controls = {
        "opening_family_probability": torch.zeros(1, 2, 8, 8),
        "opening_foreground_probability": foreground,
        "opening_family_logits": family_logits,
    }
    result_foreground, result_family = opening_query_maps(
        controls,
        frame=8,
    )
    assert torch.allclose(result_foreground, foreground)
    assert torch.equal(result_family, family_logits)


def test_detection_queries_use_independent_foreground_and_family() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 1}),
            "furniture": Counter(),
            "door": Counter({"Swing": 1}),
            "window": Counter({"Regular": 1}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    foreground = torch.zeros(1, 12, 12)
    foreground[:, 4, 1:11] = 0.9
    family_logits = torch.zeros(1, 2, 12, 12)
    family_logits[:, 0] = 4.0
    family_logits[:, 1] = -4.0
    controls = {
        "opening_family_probability": torch.zeros(1, 2, 12, 12),
        "opening_foreground_probability": foreground,
        "opening_family_logits": family_logits,
    }
    candidates = CandidateBatch(
        endpoints=torch.tensor([[[[1.0, 4.0], [11.0, 4.0]]]]),
        valid=torch.tensor([[True]]),
        evidence=torch.tensor([[0.9]]),
        exterior=torch.tensor([[False]]),
        rooms=[[]],
        host_walls=[[ReadoutWall((1.0, 4.0), (11.0, 4.0), 3.0)]],
    )
    queries = detection_opening_queries(
        controls,
        candidates,
        taxonomy,
        frame=12,
        threshold=0.5,
        max_openings=4,
        device=torch.device("cpu"),
    )
    assert bool(queries.valid.any())


def test_opening_query_matches_target_host_candidate() -> None:
    endpoints = torch.tensor(
        [
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            ]
        ]
    )
    candidates = CandidateBatch(
        endpoints=endpoints,
        valid=torch.tensor([[True, True]]),
        evidence=torch.tensor([[0.8, 0.7]]),
        exterior=torch.tensor([[True, True]]),
        rooms=[[{"polygon": [[0, 0], [1, 0], [1, 1]]}]],
        host_walls=[[]],
    )
    queries = OpeningQueries(
        centers=torch.tensor([[[0.6, 1.0]]]),
        widths=torch.tensor([[0.2]]),
        family_logits=torch.tensor([[[2.0, 0.0]]]),
        valid=torch.tensor([[True]]),
    )
    targets = {
        "wall_endpoints": endpoints,
        "wall_exists": torch.tensor([[1.0, 1.0]]),
        "opening_centers": torch.tensor([[[0.6, 1.0]]]),
        "opening_exists": torch.tensor([[1.0]]),
        "opening_hosts": torch.tensor([[1]]),
        "opening_widths": torch.tensor([[0.2]]),
        "opening_classes": torch.tensor([[0]]),
    }
    taxonomy = build_taxonomy(
        {
            "room": Counter(),
            "furniture": Counter(),
            "door": Counter({"Swing": 1}),
            "window": Counter({"Regular": 1}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    matched = match_opening_host_targets(queries, candidates, targets, taxonomy)
    assert matched.matched.tolist() == [[True]]
    assert matched.hosts.tolist() == [[1]]
    assert matched.center_fraction.item() == pytest.approx(0.6)
    assert matched.width_fraction.item() == pytest.approx(0.2)


def test_target_opening_queries_and_hosts_use_canonical_geometry() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter({"Swing Beside": 2}),
            "window": Counter({"Regular": 2}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    endpoints = torch.tensor(
        [[[[0.0, 0.0], [1.0, 0.0]]]]
    )
    candidates = CandidateBatch(
        endpoints=endpoints,
        valid=torch.tensor([[True]]),
        evidence=torch.tensor([[1.0]]),
        exterior=torch.tensor([[True]]),
        rooms=[[]],
        host_walls=[[]],
    )
    targets = {
        "wall_endpoints": endpoints,
        "wall_exists": torch.tensor([[1.0]]),
        "opening_centers": torch.tensor([[[0.25, 0.0]]]),
        "opening_exists": torch.tensor([[1.0]]),
        "opening_hosts": torch.tensor([[0]]),
        "opening_widths": torch.tensor([[0.2]]),
        "opening_classes": torch.tensor([[0]]),
    }
    queries = target_opening_queries(
        targets,
        taxonomy,
        device=torch.device("cpu"),
    )
    hosts = target_opening_host_targets(
        candidates,
        targets,
        candidate_mask=torch.tensor([[True]]),
    )
    assert queries.valid.tolist() == [[True]]
    assert hosts.hosts.tolist() == [[0]]
    assert hosts.center_fraction.item() == pytest.approx(0.25)


def test_host_matching_rejects_invalid_and_wrong_family_queries() -> None:
    endpoints = torch.tensor(
        [[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]]
    )
    candidates = CandidateBatch(
        endpoints=endpoints,
        valid=torch.tensor([[True, True]]),
        evidence=torch.tensor([[1.0, 1.0]]),
        exterior=torch.tensor([[True, True]]),
        rooms=[[]],
        host_walls=[[]],
    )
    taxonomy = build_taxonomy(
        {
            "room": Counter(),
            "furniture": Counter(),
            "door": Counter({"Swing": 1}),
            "window": Counter({"Regular": 1}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    window_local = next(
        index
        for index, class_id in enumerate(taxonomy.opening_ids)
        if taxonomy.class_names[class_id].startswith("window:")
    )
    queries = OpeningQueries(
        centers=torch.tensor([[[0.5, 0.5], [0.5, 0.5]]]),
        widths=torch.tensor([[0.2, 0.2]]),
        family_logits=torch.tensor([[[0.0, 2.0], [2.0, 0.0]]]),
        valid=torch.tensor([[True, False]]),
    )
    targets = {
        "wall_endpoints": endpoints,
        "wall_exists": torch.tensor([[1.0, 1.0]]),
        "opening_centers": torch.tensor([[[0.5, 0.5]]]),
        "opening_exists": torch.tensor([[1.0]]),
        "opening_hosts": torch.tensor([[0]]),
        "opening_widths": torch.tensor([[0.2]]),
        "opening_classes": torch.tensor([[window_local]]),
    }
    matched = match_opening_host_targets(queries, candidates, targets, taxonomy)
    assert matched.matched.tolist() == [[True, False]]
    assert matched.hosts.tolist() == [[0, 0]]


def test_host_matching_handles_no_compatible_family_pairs() -> None:
    endpoints = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]])
    candidates = CandidateBatch(
        endpoints=endpoints,
        valid=torch.tensor([[True]]),
        evidence=torch.tensor([[1.0]]),
        exterior=torch.tensor([[True]]),
        rooms=[[]],
        host_walls=[[]],
    )
    taxonomy = build_taxonomy(
        {
            "room": Counter(),
            "furniture": Counter(),
            "door": Counter({"Swing": 1}),
            "window": Counter({"Regular": 1}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    window_local = next(
        index
        for index, class_id in enumerate(taxonomy.opening_ids)
        if taxonomy.class_names[class_id].startswith("window:")
    )
    queries = OpeningQueries(
        centers=torch.tensor([[[0.5, 0.0]]]),
        widths=torch.tensor([[0.2]]),
        family_logits=torch.tensor([[[2.0, 0.0]]]),
        valid=torch.tensor([[True]]),
    )
    targets = {
        "wall_endpoints": endpoints,
        "wall_exists": torch.tensor([[1.0]]),
        "opening_centers": torch.tensor([[[0.5, 0.0]]]),
        "opening_exists": torch.tensor([[1.0]]),
        "opening_hosts": torch.tensor([[0]]),
        "opening_widths": torch.tensor([[0.2]]),
        "opening_classes": torch.tensor([[window_local]]),
    }
    matched = match_opening_host_targets(queries, candidates, targets, taxonomy)
    assert not matched.matched.any()


def test_candidate_host_walls_use_only_final_selected_candidates() -> None:
    candidates = CandidateBatch(
        endpoints=torch.tensor(
            [
                [
                    [[0.0, 0.0], [1.0, 0.0]],
                    [[0.0, 0.5], [1.0, 0.5]],
                ]
            ]
        ),
        valid=torch.tensor([[True, True]]),
        evidence=torch.tensor([[0.9, 0.8]]),
        exterior=torch.tensor([[True, False]]),
        rooms=[[]],
        host_walls=[[]],
    )
    walls = candidate_host_walls(
        candidates,
        torch.tensor([[True, False]]),
        frame=100.0,
    )[0]
    assert len(walls) == 1
    assert walls[0].start == (0.0, 0.0)
    assert walls[0].end == (100.0, 0.0)
    assert walls[0].wall_type == "exterior"
