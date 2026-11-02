from __future__ import annotations

from collections import Counter

import torch
from torch import nn

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.recursive_room_prior import (
    predict_recursive_room_prior,
)
from floorflow.recursion.complexity import RecursionPolicy
from floorflow.recursion.graph import (
    build_partition_graph,
    graph_refine_region_logits,
)
from floorflow.recursion.graph_adapter import RecursiveSemanticGraphAdapter
from floorflow.recursion.partition import (
    decode_partitioned,
    merge_region_labels,
    merge_region_logits,
    propose_partition,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Bed": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_recursive_partition_covers_footprint_without_overlap() -> None:
    footprint = torch.zeros((1, 64, 64))
    footprint[:, 4:60, 6:58] = 1
    direct = propose_partition(footprint, 0.0, max_depth=3)
    recursive = propose_partition(footprint, 1.0, max_depth=3)
    assert len(direct) == 1
    assert len(recursive) > 1
    counts = torch.stack([region.mask for region in recursive]).sum(dim=0)
    assert torch.equal(counts.bool(), footprint[0].bool())
    assert counts.max() == 1
    assert all(region.area > 0 for region in recursive)
    assert max(region.depth for region in recursive) >= 1


def test_partitioned_decoder_merges_and_repairs_seams() -> None:
    taxonomy = _taxonomy()
    footprint = torch.zeros((1, 64, 64))
    footprint[:, 4:60, 6:58] = 1
    regions = propose_partition(footprint, 1.0, max_depth=2)
    room_id = taxonomy.class_id("room:Bedroom")

    def decode(crop):
        return torch.full_like(crop, room_id, dtype=torch.long)

    merged, local = decode_partitioned(
        footprint,
        regions,
        decode,
        taxonomy=taxonomy,
    )
    assert len(local) == len(regions)
    assert torch.equal((merged > 0), footprint[0].bool())
    assert torch.all(merged[footprint[0].bool()] == room_id)

    region_labels = [
        torch.full(
            (
                region.bbox[2] - region.bbox[0],
                region.bbox[3] - region.bbox[1],
            ),
            room_id,
        )
        for region in regions
    ]
    assembled = merge_region_labels(footprint, regions, region_labels)
    assert assembled.shape == footprint[0].shape
    assert torch.equal(assembled.bool(), footprint[0].bool())
    logits = [
        torch.ones(
            2,
            region.bbox[2] - region.bbox[0],
            region.bbox[3] - region.bbox[1],
        )
        for region in regions
    ]
    merged_logits = merge_region_logits(footprint, regions, logits)
    assert merged_logits.shape == (2, 64, 64)
    assert torch.equal(
        torch.isfinite(merged_logits).any(dim=0),
        footprint[0].bool(),
    )


def test_recursive_room_prior_decodes_from_footprint_only() -> None:
    class DummySegmenter(nn.Module):
        def forward(self, footprint):
            return torch.zeros(
                footprint.shape[0],
                2,
                *footprint.shape[-2:],
                device=footprint.device,
            )

    footprint = torch.zeros((1, 64, 64))
    footprint[:, 4:60, 6:58] = 1
    prior = predict_recursive_room_prior(
        RecursionPolicy(),
        DummySegmenter(),
        footprint,
        image_size=32,
    )
    assert prior.logits.shape == (1, 2, 64, 64)
    assert prior.action.shape == (1,)
    assert prior.regions


def test_partition_graph_message_passing_shapes() -> None:
    footprint = torch.zeros((1, 64, 64))
    footprint[:, 4:60, 6:58] = 1
    regions = propose_partition(footprint, 1.0, max_depth=2)
    graph = build_partition_graph(regions)
    assert graph.adjacency.shape == (len(regions), len(regions))
    assert torch.allclose(graph.adjacency, graph.adjacency.T)
    local_logits = [
        torch.randn(
            3,
            region.bbox[2] - region.bbox[0],
            region.bbox[3] - region.bbox[1],
        )
        for region in regions
    ]
    refined, diagnostics = graph_refine_region_logits(
        local_logits,
        graph,
        blend=0.2,
        margin_threshold=0.5,
    )
    assert len(refined) == len(local_logits)
    assert all(
        value.shape == original.shape
        for value, original in zip(refined, local_logits, strict=True)
    )
    assert diagnostics["nodes"] == len(regions)
    assert diagnostics["updated_nodes"] <= diagnostics["nodes"]


def test_recursive_graph_adapter_uses_cross_attention_and_zero_init() -> None:
    footprint = torch.zeros((1, 64, 64))
    footprint[:, 4:60, 6:58] = 1
    regions = propose_partition(footprint, 1.0, max_depth=2)
    graph = build_partition_graph(regions)
    local_logits = [
        torch.randn(
            3,
            region.bbox[2] - region.bbox[0],
            region.bbox[3] - region.bbox[1],
        )
        for region in regions
    ]
    adapter = RecursiveSemanticGraphAdapter(3, hidden_size=16)
    refined, diagnostics = adapter(local_logits, graph)
    assert all(
        torch.equal(value, original)
        for value, original in zip(refined, local_logits, strict=True)
    )
    assert diagnostics["nodes"] == len(regions)
    with torch.no_grad():
        adapter.class_head.weight.normal_()
    changed, _ = adapter(local_logits, graph)
    assert any(
        not torch.equal(value, original)
        for value, original in zip(changed, local_logits, strict=True)
    )
