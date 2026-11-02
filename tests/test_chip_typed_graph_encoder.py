from __future__ import annotations

import torch

from floorflow.chip_flow.graph_conditioning import graph_conditioned_features
from floorflow.chip_flow.sparse_layers import masked_token_mean
from floorflow.chip_flow.typed_graph_encoder import TypedGraphEncoder


def _batch() -> dict:
    count = 7
    b2b = torch.tensor(
        [
            [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5], [2.0, 3.0, 0.7]],
            [[0.0, 1.0, 1.0], [1.0, 2.0, 1.0], [3.0, 4.0, 0.8]],
        ]
    )
    constraints = torch.zeros(2, count, 5)
    constraints[:, :, 4] = torch.tensor(
        [[1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0]] * 2
    )
    constraints[0, :2, 3] = 1
    constraints[1, 2:5, 3] = 2
    constraints[:, :2, 2] = 1
    boxes = torch.stack(
        [
            torch.stack(
                [
                    torch.arange(count),
                    torch.zeros(count),
                    torch.ones(count),
                    torch.ones(count),
                ],
                dim=-1,
            )
        ]
        * 2
    )
    return {
        "areas": torch.ones(2, count),
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "b2b": b2b,
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.5]],
                [[0.0, 2.0, 1.0], [1.0, 4.0, 0.5]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.2, 0.4], [0.8, 0.6], [0.0, 0.0]],
                [[0.1, 0.5], [0.7, 0.9], [0.0, 0.0]],
            ]
        ),
        "n_b2b": torch.tensor([3, 3]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
    }


def _model() -> TypedGraphEncoder:
    return TypedGraphEncoder(
        hidden=32,
        heads=4,
        graph_layers=2,
        cluster_slots=4,
        mib_slots=1,
    )


def _forward(model: TypedGraphEncoder, batch: dict):
    features, _, _ = graph_conditioned_features(batch)
    return model(
        features,
        batch["constraints"],
        areas=batch["areas"],
        b2b=batch["b2b"],
        p2b=batch["p2b"],
        pins=batch["pins"],
        n_b2b=batch["n_b2b"],
        n_p2b=batch["n_p2b"],
        n_pins=batch["n_pins"],
    )


def test_typed_graph_encoder_shapes_masks_and_gradients() -> None:
    torch.manual_seed(3)
    model = _model()
    model.cross_attention.gate.data.fill_(1.0)
    output = _forward(model, _batch())
    assert output.block_tokens.shape == (2, 7, 32)
    assert output.global_token.shape == (2, 1, 32)
    assert output.structural_tokens.shape == (2, 9, 32)
    assert output.constraint_tokens.shape == (2, 3, 32)
    assert output.port_tokens.shape == (2, 3, 32)
    assert 0.0 < float(output.diagnostics["typed_edge_fraction"]) < 1.0
    output.block_tokens.square().mean().backward()
    assert model.block_input[0].weight.grad is not None
    assert model.cross_attention.gate.grad is not None
    assert model.port_input.weight.grad is not None


def test_typed_graph_encoder_is_permutation_equivariant() -> None:
    torch.manual_seed(5)
    model = _model().eval()
    batch = _batch()
    original = _forward(model, batch).block_tokens
    permutation = torch.tensor([3, 0, 5, 1, 6, 2, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(7)
    permuted = {key: value.clone() for key, value in batch.items()}
    permuted["b2b"] = torch.cat(
        [
            inverse[batch["b2b"][..., :2].long()].float(),
            batch["b2b"][..., 2:],
        ],
        dim=-1,
    )
    permuted["p2b"] = batch["p2b"].clone()
    permuted["p2b"][..., 1] = inverse[batch["p2b"][..., 1].long()].float()
    for key in ("areas", "boxes", "constraints", "features"):
        permuted[key] = batch[key][:, permutation]
    reordered = _forward(model, permuted).block_tokens
    assert torch.allclose(original[:, permutation], reordered, atol=2.0e-5)


def test_typed_graph_encoder_handles_empty_pins() -> None:
    model = _model().eval()
    batch = _batch()
    batch["p2b"].zero_()
    batch["pins"].zero_()
    batch["n_p2b"].zero_()
    batch["n_pins"].zero_()
    with torch.no_grad():
        output = _forward(model, batch)
    assert torch.isfinite(output.block_tokens).all()
    assert torch.isfinite(output.port_tokens).all()


def test_full_typed_graph_encoder_parameter_budget() -> None:
    model = TypedGraphEncoder()
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000


def test_netlist_edge_gate_is_identity_initialized_and_trainable() -> None:
    torch.manual_seed(11)
    base = TypedGraphEncoder(hidden=32, heads=4, graph_layers=1)
    gated = TypedGraphEncoder(
        hidden=32,
        heads=4,
        graph_layers=1,
        edge_gate=True,
    )
    gated.load_state_dict(base.state_dict(), strict=False)
    base.cross_attention.gate.data.fill_(1.0)
    gated.cross_attention.gate.data.fill_(1.0)
    batch = _batch()
    with torch.no_grad():
        expected = _forward(base.eval(), batch).block_tokens
        actual = _forward(gated.eval(), batch).block_tokens
    assert torch.allclose(actual, expected, atol=1.0e-6)
    gate = gated.graph_layers[0].b2b_gate
    assert gate is not None
    assert torch.equal(gate[-1].weight, torch.zeros_like(gate[-1].weight))
    gated.train()
    _forward(gated, batch).block_tokens.square().mean().backward()
    assert gate[-1].weight.grad is not None
    assert torch.isfinite(gate[-1].weight.grad).all()


def test_typed_attention_bias_is_zero_initialized_and_trainable() -> None:
    torch.manual_seed(7)
    model = TypedGraphEncoder(
        hidden=32,
        heads=4,
        graph_layers=2,
        typed_mode="typed_bias",
    )
    model.cross_attention.gate.data.fill_(1.0)
    assert torch.equal(
        model.cross_attention.type_bias,
        torch.zeros_like(model.cross_attention.type_bias),
    )

    output = _forward(model, _batch())
    assert torch.isfinite(output.block_tokens).all()
    output.block_tokens.square().mean().backward()
    assert model.cross_attention.type_bias.grad is not None
    assert torch.isfinite(model.cross_attention.type_bias.grad).all()


def test_dense_and_masked_controls_preserve_parameter_budget() -> None:
    batch = _batch()
    full = _model()
    dense = TypedGraphEncoder(
        hidden=32,
        heads=4,
        graph_layers=2,
        typed_mode="dense",
    )
    no_structure = TypedGraphEncoder(
        hidden=32,
        heads=4,
        graph_layers=2,
        typed_mode="no_structural",
    )
    outputs = [
        _forward(model.eval(), batch)
        for model in (full, dense, no_structure)
    ]
    assert float(outputs[1].diagnostics["typed_edge_fraction"]) == 1.0
    assert torch.equal(
        outputs[2].structural_mask,
        torch.zeros_like(outputs[2].structural_mask),
    )
    counts = [
        sum(parameter.numel() for parameter in model.parameters())
        for model in (full, dense, no_structure)
    ]
    assert len(set(counts)) == 1


def test_masked_token_mean_matches_explicit_reduction() -> None:
    tokens = torch.randn(3, 7, 8, requires_grad=True)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    projected = masked_token_mean(tokens, mask)
    weights = mask.to(tokens.dtype)
    expected = torch.einsum("bn,bnd->bd", weights, tokens)
    expected = expected / weights.sum(dim=1, keepdim=True)

    assert torch.equal(projected, expected)
    projected.square().mean().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
