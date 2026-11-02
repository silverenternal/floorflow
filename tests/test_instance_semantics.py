from __future__ import annotations

import torch

from floorflow.inference.instance_semantics import (
    consolidate_room_instances,
    consolidate_room_instances_with_prior,
    enforce_room_instance_count,
    graph_refine_room_instances,
    merge_room_instances_to_count,
)


def test_instance_vote_assigns_one_class_per_component() -> None:
    labels = torch.zeros(1, 8, 16, dtype=torch.long)
    labels[:, :, :8] = 10
    labels[:, :, 8:] = 20
    logits = torch.zeros(1, 2, 8, 16)
    logits[:, 1, :, :8] = 5.0
    logits[:, 0, :, 8:] = 5.0
    output = consolidate_room_instances(
        labels,
        logits,
        torch.ones(1, 8, 16),
        (10, 20),
        min_area=1,
    )
    assert (output[:, :, :8] == 20).all()
    assert (output[:, :, 8:] == 10).all()


def test_instance_prior_vote_uses_calibrated_retrieval_evidence() -> None:
    labels = torch.zeros(1, 8, 16, dtype=torch.long)
    labels[:, :, :8] = 10
    labels[:, :, 8:] = 20
    model_logits = torch.zeros(1, 2, 8, 16)
    retrieval = torch.zeros_like(model_logits)
    retrieval[:, 0, :, :8] = 2.0
    retrieval[:, 1, :, 8:] = 2.0
    output = consolidate_room_instances_with_prior(
        labels,
        model_logits,
        retrieval,
        torch.ones(1, 1, 8, 16),
        (10, 20),
        torch.tensor([1.0, 1.0]),
        min_area=1,
    )
    assert (output[:, :, :8] == 10).all()
    assert (output[:, :, 8:] == 20).all()


def test_graph_context_keeps_valid_room_labels() -> None:
    labels = torch.zeros(1, 8, 16, dtype=torch.long)
    labels[:, :, :8] = 10
    labels[:, :, 8:] = 20
    logits = torch.zeros(1, 2, 8, 16)
    logits[:, 0, :, :8] = 2.0
    logits[:, 1, :, 8:] = 2.0
    output = graph_refine_room_instances(
        labels,
        logits,
        logits,
        torch.ones(1, 8, 16),
        (10, 20),
        torch.ones(2),
        neighbor_weight=0.1,
        steps=1,
        min_area=1,
    )
    assert torch.isin(output, torch.tensor([0, 10, 20])).all()


def test_room_count_enforcement_merges_excess_instance() -> None:
    labels = torch.zeros(1, 8, 12, dtype=torch.long)
    labels[:, :, :4] = 10
    labels[:, :, 4:8] = 20
    labels[:, :, 8:] = 10
    logits = torch.zeros(1, 2, 8, 12)
    logits[:, 0, :, :4] = 4.0
    logits[:, 1, :, 4:8] = 3.0
    logits[:, 0, :, 8:] = 1.0
    output = enforce_room_instance_count(
        labels,
        logits,
        torch.ones(1, 8, 12),
        (10, 20),
        target_count=2,
        min_area=1,
    )
    from floorflow.representation.core import extract_room_instances

    class Taxonomy:
        room_ids = (10, 20)

    assert len(extract_room_instances(output[0].numpy(), Taxonomy())) == 2


def test_nearest_instance_merge_handles_background_gaps_and_3d_logits() -> None:
    labels = torch.zeros(8, 20, dtype=torch.long)
    labels[:, :4] = 10
    labels[:, 8:12] = 10
    labels[:, 16:] = 20
    logits = torch.zeros(2, 8, 20)
    logits[0, :, :4] = 4.0
    logits[0, :, 8:12] = 3.0
    logits[1, :, 16:] = 4.0
    output = merge_room_instances_to_count(
        labels[None],
        logits,
        torch.ones(1, 8, 20),
        (10, 20),
        target_count=2,
        min_area=1,
    )

    class Taxonomy:
        room_ids = (10, 20)

    from floorflow.representation.core import extract_room_instances

    assert len(extract_room_instances(output[0].numpy(), Taxonomy())) == 2
