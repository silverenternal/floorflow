from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.room_adjacency import (
    RoomAdjacencyHead,
    RoomGraphConditioner,
    mask_contact_logits,
    room_graph_weights,
)
from floorflow.heads.room_mask_fusion import (
    apply_class_count_prior,
    fused_room_mask_logits,
)
from floorflow.model.room_instance_identity_stage import (
    RoomInstanceIdentityStage,
)
from floorflow.objectives.room_adjacency import (
    dense_room_type_graph_losses,
    room_adjacency_consistency_loss,
    room_adjacency_proxy_metrics,
    room_type_adjacency_consistency_loss,
    room_type_node_consistency_loss,
    soft_mask_adjacency,
    type_adjacency_probability,
)


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_soft_mask_adjacency_is_symmetric_and_contact_sensitive() -> None:
    device = _device()
    masks = torch.full((1, 2, 16, 16), -8.0, device=device)
    masks[:, 0, :, :8] = 8.0
    masks[:, 1, :, 7:] = 8.0
    contact = soft_mask_adjacency(masks, grid=16)
    separated = masks.clone()
    separated[:, 1] = -8.0
    separated_contact = soft_mask_adjacency(separated, grid=16)
    assert torch.allclose(contact, contact.transpose(1, 2))
    assert contact[0, 0, 1] > separated_contact[0, 0, 1]


def test_mask_aware_adjacency_consumes_geometric_contact() -> None:
    device = _device()
    masks = torch.full(
        (1, 2, 16, 16),
        -8.0,
        device=device,
        requires_grad=True,
    )
    with torch.no_grad():
        masks[:, 0, :, :8] = 8.0
        masks[:, 1, :, 8:] = 8.0
    contact = mask_contact_logits(masks)
    separated = torch.full_like(masks.detach(), -8.0)
    separated[:, 0, :, :8] = 8.0
    separated[:, 1, :, 9:] = 8.0
    separated_contact = mask_contact_logits(separated)
    assert torch.allclose(contact, contact.transpose(1, 2), atol=1.0e-6)
    assert contact[0, 0, 1] > separated_contact[0, 0, 1]
    assert torch.isfinite(
        torch.diagonal(contact, dim1=-2, dim2=-1)
    ).all()
    head = RoomAdjacencyHead(4, mask_aware=True).to(device)
    query_features = torch.zeros(1, 2, 4, device=device)
    logits = head(query_features, masks)
    separated_logits = head(query_features, separated)
    assert logits[0, 0, 1] > separated_logits[0, 0, 1]
    logits.sum().backward()
    assert head.contact_scale is not None
    assert head.contact_scale.grad is not None
    assert masks.grad is not None
    assert torch.count_nonzero(masks.grad) > 0


def test_relational_adjacency_uses_contact_and_presence() -> None:
    device = _device()
    masks = torch.full((1, 3, 16, 16), -8.0, device=device)
    masks[:, 0, :, :8] = 8.0
    masks[:, 1, :, 8:] = 8.0
    queries = torch.randn(1, 3, 8, device=device)
    presence = torch.tensor([[1.0, 1.0, 0.0]], device=device)
    head = RoomAdjacencyHead(
        8,
        mask_aware=True,
        relational=True,
        relational_hidden=16,
    ).to(device)
    logits = head(queries, masks, query_presence=presence)
    loss = logits[0, :2, :2].sum()
    loss.backward()
    assert head.relational_pair is not None
    assert head.relational_pair[0].weight.grad is not None
    assert head.relational_pair[0].weight.grad.abs().sum() > 0
    assert head.relational_scale is not None
    assert head.relational_scale.grad is not None


def test_relational_head_does_not_shift_shared_initialization() -> None:
    torch.manual_seed(19)
    control = RoomInstanceIdentityStage(
        8,
        3,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        predict_room_adjacency=True,
        enable_query_set_head=True,
        query_set_hidden=16,
        query_set_pool=8,
        query_set_layers=1,
    )
    torch.manual_seed(19)
    candidate = RoomInstanceIdentityStage(
        8,
        3,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        predict_room_adjacency=True,
        enable_query_set_head=True,
        query_set_hidden=16,
        query_set_pool=8,
        query_set_layers=1,
        mask_aware_room_adjacency=True,
        relational_room_adjacency=True,
    )
    candidate_state = candidate.state_dict()
    for name, value in control.state_dict().items():
        if name in candidate_state:
            assert torch.equal(value, candidate_state[name]), name


def test_graph_residual_warm_start_restores_nonzero_output_weights() -> None:
    conditioner = RoomGraphConditioner(8, 8, 3)
    with torch.no_grad():
        conditioner.class_residual.weight.zero_()
        conditioner.mask_residual.weight.zero_()
    conditioner.reset_graph_residual_heads()
    assert conditioner.class_residual.weight.abs().sum() > 0
    assert conditioner.mask_residual.weight.abs().sum() > 0


def test_area_preserving_graph_residual_redistributes_owner_mass() -> None:
    torch.manual_seed(7)
    conditioner = RoomGraphConditioner(
        8,
        8,
        3,
        area_preserving_mask_residual=True,
    )
    queries = torch.randn(1, 3, 8)
    adjacency = torch.randn(1, 3, 3)
    features = torch.randn(1, 8, 4, 4)
    output = conditioner(
        queries,
        adjacency,
        features,
        query_presence=torch.tensor([[1.0, 0.0, 1.0]]),
    )
    residual = output["mask_residual"]
    assert torch.count_nonzero(residual[0, 1]) == 0
    assert residual[0, [0, 2]].sum(dim=(1, 2)).abs().max() < 1.0e-6
    assert residual.sum(dim=1).abs().max() < 1.0e-6
    class_residual = output["class_residual"]
    assert torch.count_nonzero(class_residual[0, 1]) == 0
    assert class_residual.sum(dim=1).abs().max() < 1.0e-6
    (
        output["mask_residual"].square().mean()
        + class_residual.square().mean()
    ).backward()
    assert conditioner.mask_residual.weight.grad is not None
    assert conditioner.class_residual.weight.grad is not None
    assert torch.count_nonzero(conditioner.mask_residual.weight.grad) > 0
    assert torch.count_nonzero(conditioner.class_residual.weight.grad) > 0


def test_room_graph_weights_suppress_absent_query_slots() -> None:
    weights = room_graph_weights(
        torch.zeros(1, 3, 3),
        query_presence=torch.tensor([[1.0, 0.0, 1.0]]),
    )
    assert torch.count_nonzero(weights[0, :, 1]) == 0
    assert torch.count_nonzero(weights[0, 1, :]) == 0
    assert weights[0, 0, 2] > 0.0
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.tensor([[1.0, 0.0, 1.0]]),
    )


def test_room_adjacency_loss_reaches_query_head() -> None:
    device = _device()
    stage = RoomInstanceIdentityStage(
        8,
        3,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        predict_room_adjacency=True,
        enable_query_set_head=True,
        query_set_hidden=16,
        query_set_pool=8,
        query_set_layers=1,
        query_set_routing_attention=False,
        count_aware_second_graph=True,
    ).to(device)
    output = stage(
        torch.randn(1, 8, 16, 16, device=device),
        torch.randn(1, 3, 16, 16, device=device),
    )
    loss, components = room_adjacency_consistency_loss(output, 4)
    loss = (
        loss
        + output["class_logits"].square().mean()
        + output["mask_logits"].square().mean()
        + output["room_count_logits"].square().mean()
    )
    assert output["room_adjacency_logits"].shape == (1, 4, 4)
    assert output["room_count_logits"].shape == (1, 5)
    assert output["query_count_selection_mask"].shape == (1, 4)
    assert "room_graph_second_pass_scale" in output
    assert torch.allclose(
        output["room_graph_query_presence"],
        output["query_count_selection_mask"].float(),
    )
    dense_room_logits = torch.zeros(1, 3, 16, 16, device=device)
    dense_room_logits[:, 0] = 2.0
    dense = stage(
        torch.zeros(1, 8, 16, 16, device=device),
        dense_room_logits,
    )
    assert not torch.allclose(
        dense["room_instance_logits"],
        torch.zeros_like(dense["room_instance_logits"]),
    )
    assert dense["room_owner_boundary"].shape == (
        dense["room_instance_logits"].shape[0],
        *dense["room_instance_logits"].shape[-2:],
    )
    owner_class_spread = (
        dense["room_owner_fused_logits"].amax(dim=1)
        - dense["room_owner_fused_logits"].amin(dim=1)
    )
    assert owner_class_spread.max() > 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert stage.room_adjacency_head.projection.weight.grad is not None
    assert (
        stage.room_graph_conditioner.mask_residual.weight.grad is not None
    )
    assert (
        stage.room_graph_conditioner.query_update[0].weight.grad is not None
    )
    assert (
        stage.room_graph_conditioner.query_update[0].weight.grad.abs().sum() > 0
    )
    assert (
        stage.room_graph_conditioner.class_residual.weight.grad is not None
    )
    assert stage.query_set_head.count[-1].weight.grad is not None
    assert stage.class_count_prior_scale.grad is not None
    assert components["room_adjacency_valid_pairs"] > 0
    metrics = room_adjacency_proxy_metrics(
        output["room_adjacency_logits"],
        output["mask_logits"],
    )
    assert 0.0 <= metrics["room_adjacency_f1"] <= 1.0
    assert metrics["room_adjacency_mae"] >= 0.0


def test_class_count_prior_calibrates_foreground_classes() -> None:
    class_logits = torch.zeros(1, 4, 3)
    class_count_logits = torch.full((1, 2, 5), -8.0)
    class_count_logits[0, 0, 3] = 8.0
    class_count_logits[0, 1, 0] = 8.0
    adjusted = apply_class_count_prior(
        class_logits,
        class_count_logits,
        scale=0.5,
    )
    assert adjusted[0, 0, 0] > adjusted[0, 0, 1]
    assert torch.equal(adjusted[..., -1], class_logits[..., -1])


def test_adjacency_loss_uses_target_room_edges() -> None:
    masks = torch.full((1, 4, 16, 16), -8.0)
    masks[:, 0, :, :8] = 8.0
    masks[:, 1, :, 7:] = 8.0
    target_masks = masks[:, :2].sigmoid()
    logits = torch.randn(1, 4, 4, requires_grad=True)
    output = {
        "room_adjacency_logits": logits,
        "room_instance_mask_logits": masks,
        "room_instance_query_existence_logits": torch.tensor(
            [[8.0, 8.0, -8.0, -8.0]]
        ),
    }
    loss, components = room_adjacency_consistency_loss(
        output,
        4,
        target_masks,
        torch.tensor([[True, True]]),
    )
    assert torch.isfinite(loss)
    assert components["room_adjacency_valid_pairs"] > 0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) > 0


def test_adjacency_supervision_does_not_vanish_when_presence_is_low() -> None:
    masks = torch.full((1, 4, 16, 16), -8.0)
    masks[:, 0, :, :8] = 8.0
    masks[:, 1, :, 7:] = 8.0
    output = {
        "room_adjacency_logits": torch.randn(
            1,
            4,
            4,
            requires_grad=True,
        ),
        "room_instance_mask_logits": masks,
        "room_graph_query_presence": torch.zeros(1, 4),
    }
    loss, components = room_adjacency_consistency_loss(
        output,
        4,
        masks[:, :2].sigmoid(),
        torch.tensor([[True, True]]),
    )
    assert components["room_adjacency_valid_pairs"] > 0
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert output["room_adjacency_logits"].grad is not None


def test_room_fusion_uses_selected_slot_without_double_presence_gate() -> None:
    predictions = {
        "mask_logits": torch.tensor([[[[8.0]], [[8.0]]]]),
        "class_logits": torch.tensor(
            [[[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0]]]
        ),
        "query_existence_logits": torch.tensor([[8.0, -8.0]]),
        "query_count_selection_mask": torch.tensor([[True, True]]),
    }
    without_presence = fused_room_mask_logits(
        predictions,
        2,
        fusion_mode="sigmoid",
    )
    with_presence = fused_room_mask_logits(
        predictions,
        2,
        fusion_mode="sigmoid_existence",
    )
    assert with_presence.shape == without_presence.shape
    without_margin = without_presence[0, 0] - without_presence[0, 1]
    with_margin = with_presence[0, 0] - with_presence[0, 1]
    assert with_margin == without_margin


def test_type_adjacency_loss_trains_the_evaluated_edge_space() -> None:
    masks = torch.full((1, 3, 8, 8), -8.0)
    masks[:, 0, :, :3] = 8.0
    masks[:, 1, :, 2:6] = 8.0
    masks[:, 2, :, 5:] = 8.0
    logits = (masks > 0).float()
    classes = F.one_hot(
        torch.tensor([[0, 1, 2]]),
        num_classes=3,
    ).float()
    query_adjacency = torch.tensor(
        [[
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]]
    )
    type_edges = type_adjacency_probability(
        query_adjacency,
        classes,
    )
    assert type_edges[0, 0, 1] > 0.9
    assert type_edges[0, 1, 2] > 0.9
    assert type_edges[0, 0, 2] < 0.1

    output = {
        "room_adjacency_logits": logits.new_zeros((1, 3, 3))
        .requires_grad_(),
        "room_instance_class_logits": torch.cat(
            [torch.eye(3) * 2.0, torch.full((3, 1), -2.0)],
            dim=-1,
        )[None].requires_grad_(),
    }
    loss, components = room_type_adjacency_consistency_loss(
        output,
        3,
        torch.tensor([[0, 1, 2]]),
        torch.tensor([[True, True, True]]),
        torch.ones(1, 3, 8, 8),
        grid=8,
    )
    assert torch.isfinite(loss)
    assert components["room_type_adjacency_valid_pairs"] > 0
    loss.backward()
    assert torch.count_nonzero(
        output["room_adjacency_logits"].grad
    ) > 0
    assert torch.count_nonzero(
        output["room_instance_class_logits"].grad
    ) > 0


def test_type_node_loss_trains_presence_and_class_set() -> None:
    class_logits = torch.tensor(
        [
            [4.0, -4.0, -4.0, -4.0],
            [-4.0, 4.0, -4.0, -4.0],
            [-4.0, -4.0, -4.0, 4.0],
        ]
    )[None].requires_grad_()
    existence = torch.tensor([[4.0, 4.0, 4.0]]).requires_grad_()
    loss, components = room_type_node_consistency_loss(
        {
            "room_instance_class_logits": class_logits,
            "room_instance_query_existence_logits": existence,
        },
        3,
        torch.tensor([[0, 1, 2]]),
        torch.tensor([[True, True, True]]),
    )
    assert torch.isfinite(loss)
    assert components["room_type_node_target_mean"] > 0
    loss.backward()
    assert torch.count_nonzero(class_logits.grad) > 0
    assert torch.count_nonzero(existence.grad) > 0


def test_dense_type_graph_loss_reaches_final_partition_probability() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    probability = torch.full(
        (1, len(taxonomy.room_ids), 8, 8),
        0.1,
        requires_grad=True,
    )
    probability.data[:, 0, :, :4] = 0.8
    probability.data[:, 1, :, 4:] = 0.8
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :, :4] = taxonomy.room_ids[0]
    target[:, :, 4:] = taxonomy.room_ids[1]
    loss, components, terms = dense_room_type_graph_losses(
        probability,
        target,
        taxonomy,
        return_loss_terms=True,
    )
    assert torch.isfinite(loss)
    assert components["dense_type_edge_target_mean"] == 1.0
    assert set(terms) == {
        "dense_type_adjacency",
        "dense_type_node",
    }
    loss.backward()
    assert probability.grad is not None
    assert torch.count_nonzero(probability.grad) > 0


def test_dense_type_graph_overlap_term_penalizes_missed_edges() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    matched = torch.full(
        (1, len(taxonomy.room_ids), 8, 8),
        0.01,
    )
    matched[:, 0, :, :4] = 0.99
    matched[:, 1, :, 4:] = 0.99
    missed = torch.full_like(matched, 0.01)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :, :4] = taxonomy.room_ids[0]
    target[:, :, 4:] = taxonomy.room_ids[1]
    _, matched_components = dense_room_type_graph_losses(
        matched,
        target,
        taxonomy,
        adjacency_weight=0.0,
        node_weight=0.0,
        adjacency_overlap_weight=1.0,
    )
    _, missed_components = dense_room_type_graph_losses(
        missed,
        target,
        taxonomy,
        adjacency_weight=0.0,
        node_weight=0.0,
        adjacency_overlap_weight=1.0,
    )
    assert (
        matched_components["dense_type_edge_dice_loss"]
        < missed_components["dense_type_edge_dice_loss"]
    )


def test_dense_type_edge_matches_two_pixel_room_adjacency_tolerance() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    probability = torch.full(
        (1, len(taxonomy.room_ids), 8, 8),
        0.1,
        requires_grad=True,
    )
    probability.data[:, 0, :, :3] = 0.8
    probability.data[:, 1, :, 5:] = 0.8
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :, :3] = taxonomy.room_ids[0]
    target[:, :, 5:] = taxonomy.room_ids[1]
    _loss, components = dense_room_type_graph_losses(
        probability,
        target,
        taxonomy,
        adjacency_tolerance=2,
    )
    assert components["dense_type_edge_target_mean"] == 1.0


def test_dense_type_edge_penalizes_false_positive_contact() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter(
                {"Bedroom": 2, "Kitchen": 2, "Living": 2}
            ),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    target = torch.zeros(1, 12, 12, dtype=torch.long)
    target[:, :, :4] = taxonomy.room_ids[0]
    target[:, :, 4:8] = taxonomy.room_ids[1]
    target[:, :, 8:] = taxonomy.room_ids[2]
    clean = torch.full((1, 3, 12, 12), 0.01)
    clean[:, 0, :, :4] = 0.98
    clean[:, 1, :, 4:8] = 0.98
    clean[:, 2, :, 8:] = 0.98
    crossed = clean.clone()
    crossed[:, 0, :, 8:] = 0.45
    clean_loss, _ = dense_room_type_graph_losses(
        clean,
        target,
        taxonomy,
        node_weight=0.0,
    )
    crossed_loss, _ = dense_room_type_graph_losses(
        crossed,
        target,
        taxonomy,
        node_weight=0.0,
    )
    assert crossed_loss > clean_loss


