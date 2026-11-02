"""Tests for predicted-room fragment graph arbitration."""

from collections import Counter

import numpy as np
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.query_mask_graph import build_query_mask_graph
from floorflow.heads.room_fragment_graph import (
    build_fragment_graph,
    render_fragment_graph,
)
from floorflow.representation.core import extract_room_instances


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Bath": 4}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _graph_payload():
    taxonomy = _taxonomy()
    bedroom = taxonomy.class_id("room:Bedroom")
    labels = np.zeros((16, 16), dtype=np.int64)
    labels[2:14, 1:6] = bedroom
    labels[2:14, 9:14] = bedroom
    target = np.zeros_like(labels)
    target[2:14, 1:14] = bedroom
    footprint = np.ones_like(labels, dtype=bool)
    probability = np.zeros((len(taxonomy.room_ids), 16, 16), dtype=np.float32)
    probability[0, labels == bedroom] = 0.9
    graph = build_fragment_graph(
        labels,
        probability,
        target,
        footprint,
        taxonomy,
        neighbors=4,
    )
    assert graph is not None
    return graph, taxonomy


def test_fragment_graph_labels_same_target_for_merge() -> None:
    graph, _taxonomy_value = _graph_payload()
    assert graph.node_features.shape[0] == 2
    assert graph.node_target.all()
    assert graph.edge_target.any()


def test_fragment_graph_merges_accepted_same_target_edges() -> None:
    graph, taxonomy = _graph_payload()
    rendered = render_fragment_graph(
        graph,
        torch.full((2,), 8.0),
        torch.full((graph.edge_target.numel(),), 8.0),
        taxonomy,
        node_threshold=0.5,
        edge_threshold=0.5,
    )
    assert len(extract_room_instances(rendered, taxonomy)) == 1


def test_fragment_graph_without_merge_keeps_two_instances() -> None:
    graph, taxonomy = _graph_payload()
    rendered = render_fragment_graph(
        graph,
        torch.full((2,), 8.0),
        torch.full((graph.edge_target.numel(),), -8.0),
        taxonomy,
        node_threshold=0.5,
        edge_threshold=0.5,
    )
    assert len(extract_room_instances(rendered, taxonomy)) == 2


def test_query_mask_graph_uses_soft_masks_and_merges_duplicates() -> None:
    taxonomy = _taxonomy()
    bedroom = taxonomy.class_id("room:Bedroom")
    target = np.zeros((16, 16), dtype=np.int64)
    target[2:14, 1:14] = bedroom
    footprint = np.ones((16, 16), dtype=bool)
    mask_logits = torch.full((2, 16, 16), -8.0)
    mask_logits[0, 2:14, 1:8] = 8.0
    mask_logits[1, 2:14, 7:14] = 8.0
    class_logits = torch.full((2, len(taxonomy.room_ids) + 1), -8.0)
    class_logits[:, 0] = 8.0
    existence_logits = torch.tensor([4.0, 4.0])
    graph = build_query_mask_graph(
        mask_logits,
        class_logits,
        existence_logits,
        target,
        footprint,
        taxonomy,
        neighbors=2,
    )
    assert graph is not None
    assert graph.graph_kind == "query"
    assert graph.node_target.all()
    assert graph.edge_target.any()
    rendered = render_fragment_graph(
        graph,
        torch.full((2,), 8.0),
        torch.full((graph.edge_target.numel(),), 8.0),
        taxonomy,
        query_score_rendering=True,
        fill_footprint=True,
    )
    assert len(extract_room_instances(rendered, taxonomy)) == 1
