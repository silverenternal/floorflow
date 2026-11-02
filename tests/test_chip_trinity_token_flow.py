from __future__ import annotations

import io
from types import SimpleNamespace

import torch

from floorflow.chip_flow.flow_adapter import (
    decode_for_batch,
    encode_target_and_condition,
    sample_for_batch,
)
from floorflow.chip_flow.slot_model import ChipSlotFlowModel
from floorflow.chip_flow.train_objective import supervised_loss
from floorflow.chip_flow.trinity_token_flow import ChipTrinityTokenFlowModel


def _batch() -> dict:
    count = 7
    b2b = torch.tensor(
        [
            [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 3.0, 1.0]],
        ]
    )
    constraints = torch.zeros(2, count, 5)
    constraints[:, :, 4] = torch.tensor(
        [[1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0]] * 2
    )
    constraints[0, 0, 0] = 1.0
    constraints[1, 1, 1] = 1.0
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
    source = torch.randn(2, count, 3) * 0.2
    return {
        "areas": torch.ones(2, count),
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "source_latent": source,
        "target_latent": torch.randn(2, count, 3),
        "b2b": b2b,
        "p2b": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 6.0, 1.0]]]),
        "pins": torch.tensor([[[0.2, 0.4], [0.0, 0.0]]] * 2),
        "n_b2b": torch.tensor([2, 3]),
        "n_p2b": torch.tensor([1, 1]),
        "n_pins": torch.tensor([1, 1]),
    }


def _model() -> ChipTrinityTokenFlowModel:
    return ChipTrinityTokenFlowModel(
        hidden=32,
        depth=2,
        heads=2,
        head_dim=16,
        graph_pe_k=2,
    )


def test_trinity_token_bridge_forward_backward_decode_and_sample() -> None:
    torch.manual_seed(3)
    model = _model()
    batch = _batch()
    target, condition, context = encode_target_and_condition(model, batch)
    source = torch.randn_like(target)
    time = torch.full((2,), 0.5)
    prediction = model(source, time, condition)
    decoded, diagnostics = decode_for_batch(
        model,
        prediction,
        condition,
        context,
        batch,
    )
    sampled, _ = sample_for_batch(
        model,
        batch,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(7),
    )
    decoded.square().mean().backward()

    assert prediction.shape == (2, 7, 3)
    assert diagnostics == {}
    assert sampled.shape == (2, 7, 3)
    assert model.backbone.final.linear.weight.grad is not None
    assert torch.isfinite(decoded).all()


def test_trinity_token_bridge_is_permutation_equivariant() -> None:
    torch.manual_seed(5)
    model = _model().eval()
    batch = _batch()
    permutation = torch.tensor([3, 0, 5, 1, 6, 2, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(7)
    target, condition, _ = encode_target_and_condition(model, batch)
    time = torch.full((2,), 0.4)
    original = model(target, time, condition)

    permuted = {key: value for key, value in batch.items() if key != "_trinity_token_context"}
    permuted["b2b"] = torch.cat(
        [
            inverse[batch["b2b"][..., :2].long()].float(),
            batch["b2b"][..., 2:],
        ],
        dim=-1,
    )
    permuted["p2b"] = batch["p2b"].clone()
    permuted["p2b"][..., 1] = inverse[batch["p2b"][..., 1].long()].float()
    for key in ("areas", "boxes", "constraints", "features", "target_latent"):
        permuted[key] = batch[key][:, permutation]
    permuted_target, permuted_condition, _ = encode_target_and_condition(
        model,
        permuted,
    )
    reordered = model(permuted_target, time, permuted_condition)
    assert torch.allclose(original, reordered[:, inverse], atol=2.0e-5)


def test_trinity_token_bridge_handles_empty_graph() -> None:
    torch.manual_seed(9)
    model = _model().eval()
    batch = _batch()
    batch["b2b"].zero_()
    batch["n_b2b"].zero_()
    batch["p2b"].zero_()
    batch["n_p2b"].zero_()
    batch["n_pins"].zero_()
    target, condition, _ = encode_target_and_condition(model, batch)
    output = model(target, torch.ones(2), condition)
    assert torch.isfinite(output).all()


def test_trinity_token_bridge_checkpoint_round_trip() -> None:
    model = _model()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _model()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key


def test_trinity_token_bridge_uses_existing_supervised_objective() -> None:
    torch.manual_seed(13)
    model = _model()
    args = SimpleNamespace(
        time_std=1.0,
        flow_huber_beta=0.1,
        sparsity_weight=0.01,
        route_balance_weight=0.01,
        path_steps=0,
        path_weight=0.0,
        physics_weight=1.0,
        overlap_weight_scale=1.0,
        decode_weight=1.0,
        rho_weight=0.0,
        clean_weight=0.1,
    )
    loss, terms = supervised_loss(model, _batch(), args, torch.device("cpu"))
    loss.backward()
    assert torch.isfinite(loss)
    assert {"flow", "physics", "decode", "clean"} <= terms.keys()
    assert model.backbone.final.linear.weight.grad is not None
    assert model.backbone.final.linear.weight.grad.abs().sum() > 0.0


def test_trinity_token_bridge_is_parameter_matched_to_graph_slot() -> None:
    trinity = ChipTrinityTokenFlowModel()
    graph = ChipSlotFlowModel(
        feature_dim=19,
        hidden=192,
        global_dim=192,
        target_layers=4,
        condition_layers=2,
        decoder_layers=4,
        heads=6,
        graph_conditioning=True,
    )
    trinity_parameters = sum(p.numel() for p in trinity.parameters())
    graph_parameters = sum(p.numel() for p in graph.parameters())
    assert abs(trinity_parameters - graph_parameters) / graph_parameters < 0.01
