from __future__ import annotations

import torch

from floorflow.heads.room_adjacency import (
    RoomGraphConditioner,
    owner_swap_metrics,
)
from floorflow.heads.room_owner_readout import room_owner_readout


def test_owner_diagnostic_separates_relative_identity_from_absolute_confidence() -> None:
    prediction = {
        "mask_logits": torch.tensor([[[[-8.0]], [[-16.0]]]]),
        "class_logits": torch.tensor(
            [[[4.0, -4.0], [-4.0, 4.0]]]
        ),
        "query_existence_logits": torch.tensor([[4.0, -4.0]]),
        "query_count_selection_mask": torch.tensor([[1.0, 0.0]]),
    }
    owner = room_owner_readout(
        prediction,
        2,
        fallback_logits=torch.zeros(1, 2, 1, 1),
        soft_foreground=True,
    )
    assert owner["owner_confidence"].mean() < 0.01
    assert owner["owner_relative_confidence"].mean() > 0.99
    assert not owner["owner_foreground"].any()


def test_relative_owner_mode_releases_low_absolute_confidence() -> None:
    prediction = {
        "mask_logits": torch.tensor([[[[-8.0]], [[-16.0]]]]),
        "class_logits": torch.tensor(
            [[[4.0, -4.0], [-4.0, 4.0]]]
        ),
        "query_existence_logits": torch.tensor([[4.0, -4.0]]),
        "query_count_selection_mask": torch.tensor([[1.0, 0.0]]),
    }
    owner = room_owner_readout(
        prediction,
        2,
        fallback_logits=torch.zeros(1, 2, 1, 1),
        soft_foreground=True,
        confidence_mode="relative",
    )
    assert owner["owner_foreground"].all()
    assert owner["owner_soft_weight"].mean() > 0.99
    assert owner["owner_preserving_logits"][0, 0, 0, 0] > (
        owner["owner_preserving_logits"][0, 1, 0, 0]
    )


def test_owner_readout_preserves_all_classes_from_selected_query() -> None:
    prediction = {
        "mask_logits": torch.tensor([[[[8.0]], [[-8.0]]]]),
        "class_logits": torch.tensor(
            [[[4.0, -4.0, -4.0], [-4.0, 4.0, -4.0]]]
        ),
    }
    owner = room_owner_readout(
        prediction,
        2,
        fallback_logits=torch.zeros(1, 2, 1, 1),
        confidence_mode="relative",
    )
    logits = owner["owner_preserving_logits"]
    assert logits.shape == (1, 2, 1, 1)
    assert logits[0, 0, 0, 0] > logits[0, 1, 0, 0]
    assert logits[0, 0, 0, 0] - logits[0, 1, 0, 0] > 7.0


def test_owner_mode_rejects_unknown_confidence_contract() -> None:
    prediction = {
        "mask_logits": torch.zeros(1, 1, 1, 1),
        "class_logits": torch.zeros(1, 1, 1),
    }
    try:
        room_owner_readout(
            prediction,
            1,
            confidence_mode="unknown",
        )
    except ValueError as error:
        assert "absolute or relative" in str(error)
    else:
        raise AssertionError("unknown owner confidence mode must fail")


def test_owner_swap_metrics_ignore_absent_and_unsupported_queries() -> None:
    before = torch.full((1, 2, 4, 4), -8.0)
    before[:, 0, :2] = 8.0
    before[:, 1, 2:] = 8.0
    after = before.clone()
    after[:, 0, 1] = -8.0
    after[:, 1, 1] = 8.0
    before[:, :, 3, 3] = -16.0
    after[:, :, 3, 3] = -12.0
    metrics = owner_swap_metrics(
        before,
        after,
        torch.tensor([[1.0, 1.0]]),
    )
    assert 0.0 < metrics["owner_swap_fraction"] < 1.0
    absent = owner_swap_metrics(
        before,
        after,
        torch.tensor([[0.0, 0.0]]),
    )
    assert absent["owner_swap_fraction"] == 0.0
    assert absent["owner_swap_support_fraction"] == 0.0


def test_owner_margin_gate_suppresses_confident_graph_swaps() -> None:
    torch.manual_seed(7)
    conditioner = RoomGraphConditioner(
        8,
        8,
        3,
        area_preserving_mask_residual="area_preserving_margin",
    )
    queries = torch.randn(1, 3, 8)
    adjacency = torch.randn(1, 3, 3)
    features = torch.randn(1, 8, 4, 4)
    present = torch.ones(1, 3)
    ambiguous = torch.zeros(1, 3, 4, 4)
    confident = ambiguous.clone()
    confident[:, 0] = 8.0
    confident[:, 1:] = -8.0
    ambiguous_output = conditioner(
        queries,
        adjacency,
        features,
        query_presence=present,
        mask_logits=ambiguous,
    )
    confident_output = conditioner(
        queries,
        adjacency,
        features,
        query_presence=present,
        mask_logits=confident,
    )
    assert (
        ambiguous_output["mask_residual"].abs().mean()
        > confident_output["mask_residual"].abs().mean()
    )


def test_class_only_graph_mode_disables_mask_owner_edits() -> None:
    conditioner = RoomGraphConditioner(
        8,
        8,
        3,
        area_preserving_mask_residual="class_only",
    )
    output = conditioner(
        torch.randn(1, 3, 8),
        torch.randn(1, 3, 3),
        torch.randn(1, 8, 4, 4),
        query_presence=torch.ones(1, 3),
    )
    assert torch.count_nonzero(output["mask_residual"]) == 0
    assert torch.count_nonzero(output["class_residual"]) > 0


def test_graph_bypass_is_exact_identity_for_query_classes_and_masks() -> None:
    conditioner = RoomGraphConditioner(
        8,
        8,
        3,
        area_preserving_mask_residual="bypass",
    )
    queries = torch.randn(1, 3, 8)
    output = conditioner(
        queries,
        torch.randn(1, 3, 3),
        torch.randn(1, 8, 4, 4),
        query_presence=torch.ones(1, 3),
    )
    assert torch.equal(output["query_features"], queries)
    assert torch.count_nonzero(output["mask_residual"]) == 0
    assert torch.count_nonzero(output["class_residual"]) == 0
