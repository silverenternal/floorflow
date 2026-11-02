from __future__ import annotations

import torch

from floorflow.chip_flow.flow_adapter import (
    decode_for_batch,
    encode_target_and_condition,
    sample_for_batch,
)
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.sparse_slot_flow import ChipSlotTransportFlowModel


def _batch() -> dict:
    count = 7
    features = torch.randn(2, count, 8)
    features[0, :, :5] = 0.0
    features[1, :, :5] = 1.0
    b2b = torch.tensor(
        [
            [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 3.0, 1.0]],
        ]
    )
    return {
        "n": count,
        "features": features,
        "constraints": features[..., :5],
        "target_latent": torch.randn(2, count, 3),
        "source_latent": torch.randn(2, count, 3),
        "areas": torch.ones(2, count),
        "boxes": torch.ones(2, count, 4),
        "b2b": b2b,
        "p2b": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 6.0, 1.0]]]),
        "pins": torch.tensor([[[0.2, 0.4], [0.0, 0.0]]] * 2),
        "n_b2b": torch.tensor([2, 3]),
        "n_p2b": torch.tensor([1, 1]),
        "n_pins": torch.tensor([1, 1]),
    }


def _model() -> ChipSlotTransportFlowModel:
    return ChipSlotTransportFlowModel(
        feature_dim=8,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        max_slots=4,
        min_slots=2,
        assignment_top_k=2,
        decoder_top_k=2,
        transport_layers=1,
    )


def test_slot_transport_flow_shapes_and_adapter_round_trip() -> None:
    torch.manual_seed(3)
    model = _model().eval()
    batch = _batch()
    target, condition, context = encode_target_and_condition(model, batch)
    source = torch.randn_like(target)
    time = torch.full((2,), 0.5)
    velocity = model(source, time, condition)
    decoded, diagnostics = decode_for_batch(
        model,
        target,
        condition,
        context,
        batch,
    )
    sampled, sampled_diagnostics = sample_for_batch(
        model,
        batch,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(7),
    )

    assert target.shape == (2, 5, 32)
    assert condition.shape == (2, 5 * 32)
    assert velocity.shape == target.shape
    assert decoded.shape == (2, 7, 3)
    assert sampled.shape == decoded.shape
    assert diagnostics["active_slot_count"].shape == (2,)
    assert sampled_diagnostics["active_slot_count"].shape == (2,)
    assert torch.isfinite(decoded).all()
    boxes = decode_boxes_batch(decoded, batch["areas"])
    assert torch.allclose(
        boxes[..., 2] * boxes[..., 3],
        batch["areas"],
        atol=1.0e-5,
    )


def test_slot_transport_is_permutation_equivariant() -> None:
    torch.manual_seed(5)
    model = _model().eval()
    batch = _batch()
    permutation = torch.tensor([3, 0, 5, 1, 6, 2, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    permuted_edges = inverse[batch["b2b"][..., :2].long()]
    permuted_edges = torch.cat(
        [permuted_edges.float(), batch["b2b"][..., 2:]],
        dim=-1,
    )
    permuted_p2b = batch["p2b"].clone()
    permuted_p2b[..., 1] = inverse[batch["p2b"][..., 1].long()].float()
    original = model.encode_target(
        batch["target_latent"],
        batch["features"],
        **{
            key: batch[key]
            for key in (
                "b2b",
                "p2b",
                "pins",
                "n_b2b",
                "n_p2b",
                "n_pins",
            )
        },
    )
    batch["b2b"] = permuted_edges
    batch["p2b"] = permuted_p2b
    batch["features"] = batch["features"][:, permutation]
    batch["constraints"] = batch["constraints"][:, permutation]
    batch["target_latent"] = batch["target_latent"][:, permutation]
    permuted = model.encode_target(
        batch["target_latent"],
        batch["features"],
        **{
            key: batch[key]
            for key in (
                "b2b",
                "p2b",
                "pins",
                "n_b2b",
                "n_p2b",
                "n_pins",
            )
        },
    )
    assert torch.allclose(original, permuted, atol=1.0e-5)


def test_residual_slot_targets_are_condition_relative() -> None:
    torch.manual_seed(9)
    model = ChipSlotTransportFlowModel(
        feature_dim=8,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        max_slots=4,
        min_slots=2,
        assignment_top_k=2,
        decoder_top_k=2,
        transport_layers=1,
        residual_slots=True,
    ).eval()
    batch = _batch()
    target, condition, context = encode_target_and_condition(model, batch)
    source = torch.randn_like(target)
    source[:, 1:] = 0.0
    velocity = model(source, torch.full((2,), 0.5), condition)
    sampled, _ = sample_for_batch(
        model,
        batch,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(11),
    )

    assert context is not None
    active = context.hierarchy.active_slot_mask
    assert torch.all(target[:, 1:][~active].abs() < 1.0e-7)
    assert velocity.shape == target.shape
    assert sampled.shape == (2, 7, 3)
