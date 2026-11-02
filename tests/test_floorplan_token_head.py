import torch

from floorflow.heads.floorplan_token_head import FloorplanTokenHead
from floorflow.objectives.token_grammar import (
    FloorplanTokenVocabulary,
    allowed_next_kinds,
    constrained_argmax,
    grammar_allowed_mask,
    grammar_prefix_violations,
    token_sequence_loss,
)
from floorflow.representation.unified_floorplan_tokens import FloorplanToken


def test_token_head_and_loss_are_differentiable() -> None:
    head = FloorplanTokenHead(
        8,
        12,
        max_length=6,
        hidden_dim=16,
        heads=4,
    )
    head.eval()
    context = torch.randn(2, 8)
    token_ids = torch.randint(0, 12, (2, 5))
    logits = head(context, token_ids)
    targets = torch.randint(0, 12, (2, 5))
    loss = token_sequence_loss(logits, targets)
    assert logits.shape == (2, 5, 12)
    loss.backward()
    assert head.output.weight.grad is not None


def test_token_head_is_causal_and_masks_padding() -> None:
    torch.manual_seed(7)
    head = FloorplanTokenHead(
        8,
        12,
        max_length=6,
        hidden_dim=16,
        heads=4,
    )
    head.eval()
    context = torch.randn(1, 8)
    prefix = torch.tensor([[1, 2, 3, 4]])
    with_future = torch.tensor([[1, 2, 3, 11]])
    prefix_logits = head(context, prefix)
    future_logits = head(context, with_future)
    assert torch.equal(prefix_logits[:, :3], future_logits[:, :3])

    mask = torch.tensor([[True, True, False, False]])
    masked_logits = head(context, torch.tensor([[1, 2, 12, -1]]), mask)
    changed_padding = head(context, torch.tensor([[1, 2, 99, -7]]), mask)
    assert torch.equal(masked_logits[:, :3], changed_padding[:, :3])


def test_grammar_constraints_and_prefix_diagnostics() -> None:
    prefix = (
        FloorplanToken("SCHEMA", ("v",)),
        FloorplanToken("CASE", ("case",)),
        FloorplanToken("TRACK", ("cubicasa_reconstruction",)),
        FloorplanToken("SIZE", ("32", "32")),
        FloorplanToken("SOURCE_HASH", ("hash",)),
    )
    assert allowed_next_kinds(prefix) == frozenset(
        {"QUANTIZER", "FOOTPRINT_BEGIN"}
    )
    vocabulary = FloorplanTokenVocabulary(
        (
            "QUANTIZER|8",
            "FOOTPRINT_BEGIN",
            "END",
        )
    )
    mask = grammar_allowed_mask(prefix, vocabulary)
    assert mask[:2].all()
    assert not mask[2]
    selected = constrained_argmax(
        torch.tensor([0.0, 1.0, 8.0]),
        prefix,
        vocabulary,
    )
    assert selected == 1
    violations = grammar_prefix_violations(
        (*prefix, FloorplanToken("END"))
    )
    assert violations and violations[0]["index"] == 5
