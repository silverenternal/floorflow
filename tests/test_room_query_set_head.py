import torch

from floorflow.heads import room_query_set_objective as query_objective_module
from floorflow.heads.room_query_set_head import (
    RoomQuerySetHead,
    _one_to_one_query_targets,
    query_count_selection_from_presence,
    room_query_set_loss,
)
from floorflow.heads.room_query_set_objective import (
    _normalized_cardinality_distance,
)
from floorflow.objectives.room_query_matching import RoomQueryMatch


def test_room_query_set_loss_reaches_all_cardinality_heads() -> None:
    head = RoomQuerySetHead(
        8,
        6,
        3,
        6,
        hidden_dim=16,
        pool_size=4,
    )
    features = torch.randn(2, 6, 12, 12)
    queries = torch.randn(2, 6, 8)
    masks = torch.randn(2, 6, 12, 12)
    base_existence = torch.zeros(2, 6)
    predictions = head(features, queries, masks, base_existence)
    predictions["mask_logits"] = masks
    target_classes = torch.tensor(
        [[0, 0, 2, 2, 1, 0], [1, 2, 0, 0, 0, 0]]
    )
    target_valid = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, False, False, False],
        ]
    )
    loss, components = room_query_set_loss(
        predictions,
        target_classes,
        target_valid,
        target_masks=torch.rand(2, 6, 12, 12),
        selection_alignment_weight=0.2,
        presence_rank_weight=0.2,
        presence_count_regression_weight=0.5,
    )
    assert torch.isfinite(loss)
    assert components["room_query_count_loss"] >= 0
    assert 0.0 <= components["room_query_count_selected_target_recall"] <= 1.0
    assert 0.0 <= components["room_query_count_selected_background_rate"] <= 1.0
    assert components["room_query_count_selection_available"] == 1.0
    assert 0.0 <= components["room_query_mask_only_assignment_disagreement"] <= 1.0
    assert components["room_query_selection_alignment"] >= 0.0
    assert components["room_query_presence_rank"] > 0.0
    assert components["room_query_presence_count_regression"] >= 0.0
    loss.backward()
    assert head.presence[-1].weight.grad is not None
    assert head.count[-1].weight.grad is not None
    assert head.class_count.weight.grad is not None
    assert head.presence_count_scale.grad is not None


def test_query_set_initialization_is_exact_class_existence_prior() -> None:
    head = RoomQuerySetHead(4, 3, 2, 4, hidden_dim=8, pool_size=2)
    outputs = head(
        torch.randn(1, 3, 8, 8),
        torch.randn(1, 4, 4),
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 4),
    )
    assert outputs["query_existence_logits"].shape == (1, 4)
    assert outputs["room_count_logits"].shape == (1, 5)
    assert outputs["room_class_count_logits"].shape == (1, 2, 5)
    assert outputs["query_count_candidate_mask"].shape == (1, 4)
    assert outputs["query_count_soft_selection"].shape == (1, 4)
    assert torch.all(outputs["query_count_soft_selection"] >= 0.0)
    assert torch.all(outputs["query_count_soft_selection"] <= 1.0)


def test_query_set_initial_count_prior_is_centered_near_eight() -> None:
    head = RoomQuerySetHead(
        8,
        8,
        3,
        32,
        hidden_dim=16,
        pool_size=4,
    )
    expected = (
        head.count[-1].bias.softmax(dim=0)
        * torch.arange(33, dtype=head.count[-1].bias.dtype)
    ).sum().detach()
    assert 6.5 < float(expected) < 9.5


def test_class_count_prior_allocates_a_sparse_room_budget() -> None:
    head = RoomQuerySetHead(
        8,
        8,
        3,
        32,
        hidden_dim=16,
        pool_size=4,
        class_count_prior=torch.tensor([4.0, 1.0, 0.0]),
    )
    logits = head.class_count.bias.reshape(3, 33)
    expected = (
        logits.softmax(dim=-1)
        * torch.arange(33, dtype=logits.dtype)
    ).sum(dim=-1)
    assert 5.0 < float(expected.detach().sum()) < 9.5
    assert expected[0] > expected[1] > expected[2]
    assert expected[2] < 0.1


def test_class_count_prior_can_be_restored_after_checkpoint_load() -> None:
    head = RoomQuerySetHead(
        8,
        8,
        3,
        16,
        hidden_dim=16,
        pool_size=4,
        class_count_prior=torch.tensor([4.0, 1.0, 0.0]),
    )
    expected = head.class_count.bias.detach().clone()
    with torch.no_grad():
        head.class_count.bias.zero_()
    head.reset_class_count_bias()
    assert torch.equal(head.class_count.bias, expected)


def test_class_cardinality_consistency_connects_count_and_query_classes() -> None:
    predictions = {
        "query_existence_logits": torch.zeros(1, 3),
        "class_logits": torch.tensor(
            [[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]],
            requires_grad=True,
        ),
        "room_class_count_logits": torch.tensor(
            [[[0.0, 0.0, 4.0, 0.0], [4.0, -4.0, 0.0, 0.0]]],
            requires_grad=True,
        ),
    }
    loss, components = room_query_set_loss(
        predictions,
        torch.zeros(1, 3, dtype=torch.long),
        torch.ones(1, 3, dtype=torch.bool),
        existence_weight=0.0,
        count_weight=0.0,
        class_count_weight=0.0,
        class_cardinality_consistency_weight=1.0,
    )
    assert components["room_query_class_cardinality_consistency"] >= 0
    loss.backward()
    assert predictions["class_logits"].grad is not None
    assert (
        predictions["room_class_count_logits"].grad is not None
    )
    assert predictions["class_logits"].grad.abs().sum() > 0
    assert predictions["room_class_count_logits"].grad.abs().sum() > 0


def test_class_cardinality_scale_is_finite_and_bounded() -> None:
    query_counts = torch.ones(2, 18)
    expected_counts = torch.full((2, 18), 32.0)
    distance = _normalized_cardinality_distance(
        query_counts,
        expected_counts,
    )
    assert torch.isfinite(distance)
    assert 0.0 <= float(distance) <= 2.0


def test_query_set_loss_exposes_atomic_terms() -> None:
    predictions = {
        "query_existence_logits": torch.randn(1, 4, requires_grad=True),
        "class_logits": torch.randn(1, 4, 3, requires_grad=True),
        "mask_logits": torch.randn(1, 4, 8, 8, requires_grad=True),
        "room_count_logits": torch.randn(1, 5, requires_grad=True),
        "room_class_count_logits": torch.randn(1, 2, 5, requires_grad=True),
    }
    target_masks = torch.zeros(1, 4, 8, 8)
    target_masks[:, 0, :, :4] = 1.0
    target_masks[:, 1, :, 4:] = 1.0
    _, _, terms = room_query_set_loss(
        predictions,
        torch.tensor([[0, 1, 0, 0]]),
        torch.tensor([[True, True, False, False]]),
        target_masks=target_masks,
        return_loss_terms=True,
    )
    assert set(terms) == {
        "query_existence",
        "query_count",
        "query_class_count",
        "query_class_cardinality",
        "query_slot_class_anchor",
        "query_support_assignment",
        "query_selection_alignment",
        "query_presence_rank",
        "query_presence_count",
    }
    sum(terms.values()).backward()
    assert predictions["query_existence_logits"].grad is not None
    assert predictions["room_class_count_logits"].grad is not None


def test_soft_count_selection_has_expected_cardinality_and_gradient() -> None:
    head = RoomQuerySetHead(4, 3, 2, 4, hidden_dim=8, pool_size=2)
    features = torch.randn(1, 3, 8, 8, requires_grad=True)
    outputs = head(
        features,
        torch.randn(1, 4, 4),
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 4),
    )
    soft = outputs["query_count_soft_selection"]
    count_probability = outputs["room_count_logits"].softmax(dim=-1)
    expected_count = (
        count_probability
        * torch.arange(5, dtype=count_probability.dtype)
    ).sum(dim=-1)
    assert torch.allclose(soft.sum(dim=-1), expected_count)
    weights = torch.arange(4, dtype=soft.dtype)
    (soft * weights).sum().backward()
    assert head.count[-1].weight.grad is not None


def test_query_count_selection_uses_current_presence_without_official_head() -> None:
    presence = torch.tensor([[5.0, 4.0, 0.0, -1.0]])
    count_logits = torch.full((1, 5), -20.0)
    count_logits[0, 2] = 20.0
    outputs = query_count_selection_from_presence(presence, count_logits)
    assert outputs["selection"][0].tolist() == [True, True, False, False]
    assert outputs["ordered_indices"][0].tolist() == [0, 1, 2, 3]
    assert torch.allclose(
        outputs["soft_selection"].sum(dim=-1),
        torch.tensor([2.0]),
    )


def test_query_set_attention_uses_global_and_query_context() -> None:
    head = RoomQuerySetHead(
        16,
        8,
        3,
        6,
        hidden_dim=32,
        pool_size=8,
        attention_layers=1,
        attention_heads=4,
    )
    with torch.no_grad():
        head.global_query.weight.normal_(0.0, 0.02)
    outputs = head(
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 6, 16),
        torch.randn(2, 6, 16, 16),
        torch.randn(2, 6),
    )
    assert outputs["query_existence_logits"].shape == (2, 6)
    loss = (
        outputs["query_existence_logits"].square().mean()
        + outputs["room_count_logits"].square().mean()
    )
    loss.backward()
    assert head.set_attention[0].in_proj_weight.grad is not None
    assert head.global_query.weight.grad is not None


def test_query_set_mita_route_has_dynamic_budget_and_gradients() -> None:
    head = RoomQuerySetHead(
        16,
        8,
        3,
        8,
        hidden_dim=32,
        pool_size=8,
        attention_layers=1,
        attention_heads=4,
        routing_attention=True,
        route_router_count=6,
        route_top_k=3,
    )
    outputs = head(
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 8, 16),
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 8),
    )
    diagnostics = outputs["query_set_diagnostics"]
    for key in (
        "route_entropy",
        "route_attention_entropy",
        "route_effective_expert_fraction",
        "route_budget_mean",
        "route_gate_mean",
    ):
        assert torch.isfinite(diagnostics[key])
    (
        outputs["query_existence_logits"].square().mean()
        + outputs["room_count_logits"].square().mean()
    ).backward()
    assert head.route_attention[0].q_proj.weight.grad is not None
    assert head.route_budget[0][0].weight.grad is not None
    assert head.route_gate[0][0].weight.grad is not None


def test_query_set_loss_anchors_slot_class_residual() -> None:
    predictions = {
        "query_existence_logits": torch.zeros(1, 3),
        "slot_class_residual": torch.full((1, 3, 4), 0.1),
    }
    loss, components = room_query_set_loss(
        predictions,
        torch.zeros(1, 3, dtype=torch.long),
        torch.tensor([[True, False, False]]),
        existence_weight=0.0,
        count_weight=0.0,
        class_count_weight=0.0,
        slot_class_anchor_weight=1.0,
    )
    assert torch.allclose(loss, torch.tensor(0.01))
    assert torch.allclose(
        components["room_query_slot_class_anchor"],
        torch.tensor(0.01),
    )


def test_query_existence_target_is_one_to_one_for_duplicate_masks() -> None:
    # Two queries overlap the same target. Only one may become positive;
    # otherwise the count gate can select duplicate room masks.
    iou = torch.tensor(
        [[[0.95], [0.90], [0.05]]],
    )
    target, assignment = _one_to_one_query_targets(
        iou,
        torch.tensor([[True]]),
    )
    assert target.shape == (1, 3)
    assert float(target.sum()) == 1.0
    assert int(assignment.sum()) == 1


def test_query_existence_target_ignores_invalid_rooms() -> None:
    target, assignment = _one_to_one_query_targets(
        torch.tensor([[[0.9, 0.8], [0.2, 0.7]]]),
        torch.tensor([[False, True]]),
    )
    assert int(assignment.sum()) == 1
    assert float(target.sum()) == 1.0


def test_query_presence_uses_shared_identity_match(
    monkeypatch,
) -> None:
    mask_logits = torch.full((1, 3, 4, 4), -8.0)
    mask_logits[0, 1, :, :2] = 8.0
    mask_logits[0, 2, :, 2:] = 8.0
    target_masks = torch.zeros(1, 2, 4, 4)
    target_masks[0, 0, :, :2] = 1.0
    target_masks[0, 1, :, 2:] = 1.0
    predictions = {
        "query_existence_logits": torch.zeros(1, 3),
        "class_logits": torch.zeros(1, 3, 3),
        "mask_logits": mask_logits,
    }
    shared = [
        RoomQueryMatch(
            predicted_indices=[0, 1],
            target_indices=[0, 1],
        )
    ]
    monkeypatch.setattr(
        query_objective_module,
        "match_room_mask_queries",
        lambda *_args, **_kwargs: shared,
    )

    _loss, components = room_query_set_loss(
        predictions,
        torch.tensor([[0, 1]]),
        torch.tensor([[True, True]]),
        target_masks=target_masks,
        count_weight=0.0,
        class_count_weight=0.0,
        class_cardinality_consistency_weight=0.0,
    )

    assert (
        components["room_query_mask_only_assignment_disagreement"]
        == 1.0
    )


def test_class_count_target_accepts_noncontiguous_valid_slots() -> None:
    predictions = {
        "query_existence_logits": torch.zeros(1, 3),
        "room_instance_class_logits": torch.zeros(1, 3, 3),
        "room_instance_mask_logits": torch.zeros(1, 3, 4, 4),
        "room_count_logits": torch.zeros(1, 4),
        "room_class_count_logits": torch.zeros(1, 2, 4),
    }
    target_classes = torch.tensor([[0, 1, 1]])
    target_valid = torch.tensor([[True, False, True]])
    _loss, components = room_query_set_loss(
        predictions,
        target_classes,
        target_valid,
        existence_weight=0.0,
        count_weight=0.0,
        class_count_weight=1.0,
    )
    assert torch.isfinite(
        components["room_query_class_count_loss"]
    )
