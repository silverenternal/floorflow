import torch

from floorflow.heads.room_program_set import (
    RoomProgramSetHead,
    decode_room_program,
    load_room_program_set_state,
    match_room_program,
    room_program_cooccurrence_loss,
    room_program_set_loss,
)


def test_room_program_set_loss_is_finite_and_matches_once_per_target() -> None:
    torch.manual_seed(4)
    head = RoomProgramSetHead(
        12,
        4,
        hidden_dim=24,
        tokens=2,
        layers=1,
        heads=4,
        num_queries=6,
    )
    predictions = head(torch.randn(2, 12, 8, 8))
    target_classes = torch.tensor([[0, 0, 2, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
    target_valid = torch.tensor(
        [[True, True, True, False, False, False], [True, False, False, False, False, False]]
    )
    loss, components = room_program_set_loss(
        predictions,
        target_classes,
        target_valid,
    )
    assert torch.isfinite(loss)
    assert components["room_program_set_loss"] > 0
    assert components["room_program_cooccurrence_loss"] > 0
    loss.backward()
    assert head.class_head.weight.grad is not None
    rows, columns = match_room_program(predictions["class_logits"], target_classes, target_valid)
    assert rows[0].numel() == 3
    assert len(set(rows[0].tolist())) == 3
    assert len(set(columns[0].tolist())) == 3


def test_cardinality_decoder_uses_no_fixed_presence_threshold() -> None:
    predictions = {
        "class_logits": torch.tensor(
            [[[8.0, 0.0, -8.0], [0.0, 8.0, -8.0], [-8.0, 0.0, 8.0]]]
        ),
        "presence_logits": torch.tensor([[5.0, 4.0, -5.0]]),
        "cardinality_logits": torch.tensor([[0.0, 0.0, 8.0, 0.0]]),
    }
    selected = decode_room_program(predictions)
    assert selected.tolist() == [[True, True, False]]
    expected = decode_room_program(predictions, cardinality_method="expected")
    assert expected.tolist() == [[True, True, False]]
    by_query = decode_room_program(predictions, cardinality_method="query_count")
    assert by_query.tolist() == [[True, True, False]]


def test_legacy_program_set_state_uses_zero_cardinality_residual() -> None:
    head = RoomProgramSetHead(8, 3, hidden_dim=16, tokens=2, layers=1, num_queries=4)
    state = {
        name: value
        for name, value in head.state_dict().items()
        if not name.startswith("query_cardinality_residual.")
        and not name.startswith(
            (
                "presence_queries",
                "presence_layers.",
                "presence_norm.",
                "presence_class_head.",
                "class_presence_enabled",
                "program_prior",
                "program_ranking_strength",
            )
        )
    }
    load_room_program_set_state(head, state)
    assert head.query_cardinality_residual.weight.eq(0).all()
    assert not bool(head.class_presence_enabled)


def test_cooccurrence_loss_penalizes_independent_extra_classes() -> None:
    target = torch.tensor([[1.0, 1.0, 0.0]])
    coherent = torch.tensor([[4.0, 4.0, -4.0]])
    incoherent = torch.tensor([[4.0, -4.0, 4.0]])
    coherent_loss = room_program_cooccurrence_loss(coherent, target)
    incoherent_loss = room_program_cooccurrence_loss(incoherent, target)
    assert coherent_loss < incoherent_loss


def test_decode_uses_prior_adjusted_ranking_logits() -> None:
    predictions = {
        "class_logits": torch.zeros(1, 2, 4),
        "presence_logits": torch.tensor([[5.0, 4.0, 3.0]]),
        "presence_rank_logits": torch.tensor([[0.0, 6.0, 1.0]]),
        "cardinality_logits": torch.tensor([[0.0, 8.0, 0.0, 0.0]]),
    }
    selected = decode_room_program(predictions)
    assert selected.tolist() == [[False, True, False]]
