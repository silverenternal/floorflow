"""Tests for dense-query support arbitration."""

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.dense_query_support_gate import (
    DenseQuerySupportGate,
    SelectiveRoomSupportDeletion,
    apply_room_support_gate,
    query_mask_support_probability,
    render_support_conditioned_room_labels,
    selective_deletion_loss,
    support_gate_loss,
    support_gate_targets,
    topology_safe_keep_mask,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Bath": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_gate_is_exactly_fixed_blend_at_initialization() -> None:
    taxonomy = _taxonomy()
    gate = DenseQuerySupportGate(
        len(taxonomy.room_ids),
        fixed_blend=0.7,
        hidden_channels=8,
    )
    dense = torch.rand(1, len(taxonomy.room_ids), 8, 8).softmax(dim=1)
    query = torch.rand_like(dense).softmax(dim=1)
    support = torch.rand(1, 8, 8)
    footprint = torch.ones(1, 8, 8)
    output = gate(dense, query, support, footprint)
    assert torch.allclose(output["gate"], torch.full((1, 8, 8), 0.7))
    assert torch.allclose(output["probability"], 0.3 * dense + 0.7 * query)


def test_query_union_support_has_a_downstream_label_consumer() -> None:
    mask_logits = torch.full((1, 2, 2, 2), -20.0)
    mask_logits[0, 0, 0, 0] = 20.0
    mask_logits[0, 1, 0, 1] = 20.0
    support = query_mask_support_probability(mask_logits, mode="union")
    labels = torch.tensor([[[3, 4], [5, 6]]])
    footprint = torch.ones(1, 2, 2, dtype=torch.bool)
    gated = apply_room_support_gate(
        labels,
        support,
        footprint,
        threshold=0.5,
    )
    assert support[0, 0, 0] > 0.99
    assert support[0, 1, 1] < 1.0e-6
    assert gated.tolist() == [[[3, 4], [0, 0]]]


def test_selective_deletion_preserves_agreed_confident_room_core() -> None:
    taxonomy = _taxonomy()
    gate = SelectiveRoomSupportDeletion(
        len(taxonomy.room_ids),
        hidden_channels=8,
        preserve_confidence=0.85,
    )
    dense = torch.tensor(
        [
            [
                    [[0.9, 0.1], [0.8, 0.2]],
                    [[0.1, 0.9], [0.2, 0.8]],
            ]
        ]
    )
    query = dense.clone()
    support = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    footprint = torch.ones(1, 2, 2, dtype=torch.bool)
    output = gate(
        dense,
        query,
        support,
        footprint,
        keep_threshold=0.4,
    )
    assert output["keep_mask"].tolist() == [
        [[True, True], [False, False]]
    ]
    labels = torch.tensor([[[3, 4], [5, 6]]])
    gated = apply_room_support_gate(
        labels,
        output["keep_mask"].float(),
        footprint,
        threshold=0.5,
    )
    assert gated.tolist() == [[[3, 4], [0, 0]]]


def test_selective_deletion_loss_reaches_parameters() -> None:
    taxonomy = _taxonomy()
    gate = SelectiveRoomSupportDeletion(
        len(taxonomy.room_ids),
        hidden_channels=8,
    )
    dense = torch.rand(1, len(taxonomy.room_ids), 8, 8).softmax(dim=1)
    query = torch.rand_like(dense).softmax(dim=1)
    query_support = torch.rand(1, 8, 8)
    footprint = torch.ones(1, 8, 8, dtype=torch.bool)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, 2:6, 2:6] = taxonomy.room_ids[0]
    output = gate(dense, query, query_support, footprint)
    loss, components = selective_deletion_loss(
        output,
        target,
        footprint,
        taxonomy.room_ids,
    )
    loss.backward()
    assert loss >= 0.0
    assert components["selective_deletion_bce"] >= 0.0
    assert gate.network[-1].weight.grad is not None


def test_topology_safe_keep_rejects_room_splitting_deletion() -> None:
    labels = torch.tensor([[[1, 1, 1, 1], [0, 0, 0, 0]]])
    proposed = torch.tensor([[[True, False, True, True], [False, False, False, False]]])
    safe = topology_safe_keep_mask(labels, proposed)
    assert safe.tolist() == [[[True, True, True, True], [False, False, False, False]]]


def test_topology_safe_keep_allows_whole_component_deletion() -> None:
    labels = torch.tensor([[[1, 1, 0, 2], [0, 0, 0, 2]]])
    proposed = torch.tensor([[[True, True, False, False], [False, False, False, False]]])
    safe = topology_safe_keep_mask(labels, proposed)
    assert safe.tolist() == [[[True, True, False, False], [False, False, False, False]]]


def test_topology_safe_keep_restores_entire_large_room() -> None:
    labels = torch.ones(1, 8, 8, dtype=torch.long)
    proposed = torch.zeros(1, 8, 8, dtype=torch.bool)
    safe = topology_safe_keep_mask(labels, proposed)
    assert bool(safe.all())


def test_support_label_renderer_keeps_footprint_contract() -> None:
    probability = torch.tensor(
        [
            [
                [[0.9, 0.9], [0.9, 0.9]],
                [[0.1, 0.1], [0.1, 0.1]],
            ]
        ]
    )
    keep = torch.tensor([[[True, True], [True, True]]])
    room_ids = torch.tensor([3, 4])
    footprint = torch.tensor([[[True, True], [True, False]]])
    labels = render_support_conditioned_room_labels(
        probability,
        keep,
        room_ids,
        footprint,
        topology_safe=True,
    )
    assert labels.tolist() == [[[3, 3], [3, 0]]]


def test_arbitration_targets_mark_branch_disagreement() -> None:
    taxonomy = _taxonomy()
    bed = taxonomy.class_id("room:Bedroom")
    bath = taxonomy.class_id("room:Bath")
    target = torch.tensor([[bed, bed, bath, bath, 0]])
    dense = torch.zeros(1, len(taxonomy.room_ids), 5)
    query = torch.zeros_like(dense)
    bed_index = taxonomy.room_ids.index(bed)
    bath_index = taxonomy.room_ids.index(bath)
    dense[0, bed_index, :3] = 10.0
    dense[0, bath_index, 3:] = 10.0
    query[0, bed_index, 0:2] = 10.0
    query[0, bath_index, 2] = 10.0
    query[0, bed_index, 3] = 10.0
    room_target, arbitration = support_gate_targets(
        target,
        dense,
        query,
        taxonomy.room_ids,
    )
    assert room_target.tolist() == [[True, True, True, True, False]]
    assert arbitration[0, 2].item() == 1
    assert arbitration[0, 3].item() == 0


def test_support_gate_loss_reaches_gate_parameters() -> None:
    taxonomy = _taxonomy()
    gate = DenseQuerySupportGate(
        len(taxonomy.room_ids),
        fixed_blend=0.5,
        hidden_channels=8,
    )
    dense = torch.rand(1, len(taxonomy.room_ids), 8, 8).softmax(dim=1)
    query = torch.rand_like(dense).softmax(dim=1)
    support = torch.rand(1, 8, 8)
    footprint = torch.ones(1, 8, 8)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, 2:6, 2:6] = taxonomy.room_ids[0]
    output = gate(dense, query, support, footprint)
    loss, components = support_gate_loss(
        output,
        target,
        dense,
        query,
        footprint,
        taxonomy.room_ids,
    )
    loss.backward()
    assert components["support_gate_loss"] >= 0.0
    assert gate.residual_scale.grad is not None
