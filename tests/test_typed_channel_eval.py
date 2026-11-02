from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.representation.layered_floorplan import LayeredFloorplan
from floorflow.training.typed_channel_eval import (
    accumulate_train_component_totals,
    empty_train_component_totals,
    family_metrics,
    materialize_train_component_totals,
    selection_score,
)


def test_furniture_selection_weight_changes_checkpoint_ranking() -> None:
    wall_heavy = {
        "room_macro_iou": 0.30,
        "instance_f1": 0.22,
        "instance_type_accuracy": 0.65,
        "wall_f1": 0.50,
        "opening_f1": 0.21,
        "furniture_f1": 0.04,
        "room_f1": 0.87,
        "instance_count_abs_error": 3.6,
    }
    furniture_heavy = {
        **wall_heavy,
        "wall_f1": 0.46,
        "furniture_f1": 0.12,
    }
    assert selection_score(wall_heavy) > selection_score(furniture_heavy)
    assert selection_score(
        furniture_heavy,
        furniture_weight=0.5,
    ) > selection_score(
        wall_heavy,
        furniture_weight=0.5,
    )


def test_train_component_totals_cover_every_logged_objective() -> None:
    totals = empty_train_component_totals()
    assert {
        "ce",
        "room_iou",
        "tversky",
        "boundary",
        "family_iou",
        "stage_family",
        "stage_wall",
        "stage_wall_precision",
        "stage_opening",
        "stage_opening_precision",
        "stage_opening_host",
        "stage_opening_type",
        "stage_opening_family",
        "stage_furniture",
        "routing",
        "furniture_query",
        "instance_room_class",
        "instance_room_mask",
        "instance_room_partition",
        "room_adjacency",
        "room_adjacency_total",
        "room_adjacency_consistency_loss",
        "room_adjacency_valid_pairs",
        "room_adjacency_assigned_queries",
        "room_type_adjacency",
        "room_type_adjacency_loss",
        "room_type_adjacency_valid_pairs",
        "room_type_adjacency_target_mean",
        "room_type_node",
        "room_type_node_loss",
        "room_query_set_total",
        "room_type_adjacency_total",
        "room_type_node_total",
        "dense_type_adjacency",
        "dense_type_node",
        "dense_type_total",
        "dense_type_edge_target_mean",
    } <= totals.keys()


def test_component_totals_accumulate_on_device_before_one_sync() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    totals = empty_train_component_totals(device)
    accumulate_train_component_totals(
        totals,
        {"routing": torch.tensor(2.0, device=device)},
        batch_size=4,
    )
    accumulate_train_component_totals(
        totals,
        {"routing": torch.tensor(4.0, device=device)},
        batch_size=2,
    )
    assert totals["routing"].device == device
    values = materialize_train_component_totals(totals, samples=6)
    assert values["routing"] == 16.0 / 6.0


def test_selection_prefers_canonical_graph_quality_at_equal_layers() -> None:
    weak_graph = {
        "room_macro_iou": 0.30,
        "instance_f1": 0.20,
        "instance_type_accuracy": 0.65,
        "wall_f1": 0.45,
        "opening_f1": 0.15,
        "furniture_f1": 0.0,
        "room_f1": 0.87,
        "instance_count_abs_error": 5.0,
        "node_f1": 0.70,
        "edge_overlap": 0.30,
        "ged": 9.0,
    }
    strong_graph = {
        **weak_graph,
        "node_f1": 0.80,
        "edge_overlap": 0.45,
        "ged": 6.0,
    }
    assert selection_score(strong_graph) > selection_score(weak_graph)


def test_gate_balanced_selection_does_not_hide_opening_and_graph_gaps() -> None:
    room_only = {
        "room_macro_iou": 0.70,
        "instance_f1": 0.60,
        "instance_type_accuracy": 0.85,
        "instance_count_abs_error": 1.0,
        "wall_f1": 0.60,
        "opening_f1": 0.0,
        "furniture_f1": 0.0,
        "node_f1": 0.82,
        "edge_overlap": 0.10,
        "ged": 12.0,
    }
    balanced = {
        **room_only,
        "room_macro_iou": 0.45,
        "instance_f1": 0.50,
        "instance_type_accuracy": 0.75,
        "wall_f1": 0.58,
        "opening_f1": 0.35,
        "furniture_f1": 0.40,
        "edge_overlap": 0.50,
        "ged": 6.0,
    }
    assert selection_score(
        balanced,
        furniture_weight=1.0,
    ) > selection_score(
        room_only,
        furniture_weight=1.0,
    )


def test_selection_consumes_vector_opening_proxy() -> None:
    weak = {
        "room_macro_iou": 0.35,
        "instance_f1": 0.28,
        "instance_type_accuracy": 0.67,
        "instance_count_abs_error": 3.0,
        "wall_f1": 0.49,
        "opening_f1": 0.20,
        "opening_vector_f1_proxy": 0.0,
        "node_f1": 0.73,
        "edge_overlap": 0.36,
        "ged": 7.7,
    }
    strong = {**weak, "opening_vector_f1_proxy": 0.34}
    assert selection_score(strong) > selection_score(weak)


def test_family_metrics_report_precision_and_recall() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 1}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    footprint = torch.ones(1, 2, 2, dtype=torch.bool)
    wall = torch.tensor([[[True, True], [False, False]]])
    empty = torch.zeros(1, 2, 2, dtype=torch.long)
    prediction = LayeredFloorplan(
        room_labels=empty,
        wall=wall,
        opening=empty,
        furniture=empty,
        footprint=footprint,
        taxonomy=taxonomy,
    )
    target = empty.clone()
    target[0, 0, 0] = taxonomy.class_id("wall")
    metrics = family_metrics(
        prediction,
        target,
        footprint,
        taxonomy,
    )
    assert metrics["wall_precision"] == 0.5
    assert metrics["wall_recall"] == 1.0
