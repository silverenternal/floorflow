from __future__ import annotations

import torch

from floorflow.chip_flow.flow_adapter import (
    decode_for_batch,
    encode_target_and_condition,
    graph_adapter_inputs,
    sample_for_batch,
)
from floorflow.chip_flow.graph_conditioning import graph_conditioned_features
from floorflow.chip_flow.slot_model import ChipSlotFlowModel


def _batch() -> dict:
    count = 7
    b2b = torch.tensor(
        [
            [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 3.0, 1.0]],
        ]
    )
    constraints = torch.zeros(2, count, 5)
    constraints[:, :, 4] = torch.tensor([[1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0]] * 2)
    constraints[0, 0, 0] = 1.0
    constraints[1, 1, 1] = 1.0
    areas = torch.ones(2, count)
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
        "areas": areas,
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "target_latent": torch.randn(2, count, 3),
        "b2b": b2b,
        "p2b": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 6.0, 1.0]]]),
        "pins": torch.tensor([[[0.2, 0.4], [0.0, 0.0]]] * 2),
        "n_b2b": torch.tensor([2, 3]),
        "n_p2b": torch.tensor([1, 1]),
        "n_pins": torch.tensor([1, 1]),
    }


def _model() -> ChipSlotFlowModel:
    return ChipSlotFlowModel(
        feature_dim=19,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        graph_conditioning=True,
    )


def test_graph_features_and_adjacency_contract() -> None:
    batch = _batch()
    features, adjacency, key_mask = graph_conditioned_features(batch)
    assert features.shape == (2, 7, 19)
    assert adjacency.shape == (2, 7, 7)
    assert key_mask.shape == (2, 7)
    assert key_mask.all()
    assert torch.allclose(adjacency, adjacency.transpose(1, 2))
    assert adjacency.max() <= 1.0
    assert features[..., 10:13].abs().sum() > 0.0
    assert features[..., 16:19].abs().sum() > 0.0


def test_free_block_target_geometry_cannot_leak_into_conditioning() -> None:
    batch = _batch()
    baseline, adjacency, mask = graph_conditioned_features(batch)
    changed = {
        **batch,
        "boxes": batch["boxes"].clone(),
    }
    free = batch["constraints"][..., :2].eq(0).all(dim=-1)
    changed["boxes"][free] += 100.0
    perturbed, changed_adjacency, changed_mask = graph_conditioned_features(
        changed
    )

    torch.testing.assert_close(perturbed, baseline)
    torch.testing.assert_close(changed_adjacency, adjacency)
    torch.testing.assert_close(changed_mask, mask)


def test_graph_conditioned_global_slot_forward_backward_and_sampling() -> None:
    torch.manual_seed(3)
    model = _model()
    batch = _batch()
    target, condition, context = encode_target_and_condition(model, batch)
    assert context is None
    source = torch.randn_like(target)
    time = torch.full((2,), 0.5)
    velocity = model(source, time, condition)
    endpoint = source + 0.5 * velocity
    decoded, _ = decode_for_batch(model, endpoint, condition, context, batch)
    sampled, _ = sample_for_batch(
        model,
        batch,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(7),
    )
    loss = decoded.square().mean()
    loss.backward()

    assert velocity.shape == target.shape
    assert sampled.shape == (2, 7, 3)
    assert model.graph_bias is not None
    assert model.graph_bias.gate.grad is not None
    assert float(model.graph_bias.gate.grad.abs()) > 0.0
    assert model.condition_encoder.graph_layers[0].gate.grad is not None


def test_graph_conditioning_is_permutation_equivariant() -> None:
    torch.manual_seed(5)
    model = _model().eval()
    batch = _batch()
    permutation = torch.tensor([3, 0, 5, 1, 6, 2, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(7)
    permuted_edges = inverse[batch["b2b"][..., :2].long()]
    permuted_edges = torch.cat(
        [permuted_edges.float(), batch["b2b"][..., 2:]],
        dim=-1,
    )
    permuted_p2b = batch["p2b"].clone()
    permuted_p2b[..., 1] = inverse[batch["p2b"][..., 1].long()].float()
    features, adjacency, mask = graph_conditioned_features(batch)
    original = model.encode_target(
        batch["target_latent"],
        features,
        adjacency,
        mask,
    )
    batch["b2b"] = permuted_edges
    batch["p2b"] = permuted_p2b
    for key in ("areas", "boxes", "constraints", "features", "target_latent"):
        batch[key] = batch[key][:, permutation]
    features, adjacency, mask = graph_conditioned_features(batch)
    permuted = model.encode_target(
        batch["target_latent"],
        features,
        adjacency,
        mask,
    )
    assert torch.allclose(original, permuted, atol=1.0e-5)


def test_graph_feature_and_topology_mechanisms_can_be_ablated() -> None:
    batch = _batch()
    feature_only = ChipSlotFlowModel(
        feature_dim=19,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        graph_conditioning=False,
        graph_features=True,
        graph_bias=False,
        graph_messages=False,
    )
    target, condition, _ = encode_target_and_condition(feature_only, batch)
    assert target.shape == (2, 32)
    assert condition.shape == (2, 32)
    assert feature_only.graph_bias is None
    assert not feature_only.condition_encoder.graph_layers
    assert not feature_only.decoder.graph_layers

    topology_only = ChipSlotFlowModel(
        feature_dim=10,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        graph_conditioning=False,
        graph_features=False,
        graph_bias=True,
        graph_messages=True,
    )
    target, condition, _ = encode_target_and_condition(topology_only, batch)
    assert target.shape == (2, 32)
    assert condition.shape == (2, 32)
    assert topology_only.graph_bias is not None
    assert topology_only.condition_encoder.graph_layers
    assert topology_only.decoder.graph_layers


def test_x0_output_mode_samples_with_velocity_conversion() -> None:
    torch.manual_seed(11)
    model = _model()
    model.output_mode = "x0"
    sampled, _ = sample_for_batch(
        model,
        _batch(),
        sampling_steps=2,
        generator=torch.Generator().manual_seed(13),
    )
    assert sampled.shape == (2, 7, 3)
    assert torch.isfinite(sampled).all()


def test_graph_conditioning_dropout_is_training_only() -> None:
    model = _model()
    model.wire_dropout = 1.0
    model.constraint_dropout = 1.0
    batch = _batch()
    model.train()
    features, adjacency, _ = graph_adapter_inputs(model, batch)
    assert adjacency.abs().sum() == 0.0
    assert features[..., 3:9].abs().sum() == 0.0
    assert features[..., 16:19].abs().sum() == 0.0

    batch.pop("_graph_adapter_inputs")
    model.eval()
    features, adjacency, _ = graph_adapter_inputs(model, batch)
    assert adjacency.abs().sum() > 0.0
    assert features[..., 16:19].abs().sum() > 0.0


def test_global_pool_stats_is_a_zero_initialized_skip() -> None:
    batch = _batch()
    torch.manual_seed(23)
    base = _model().eval()
    torch.manual_seed(23)
    stats = ChipSlotFlowModel(
        feature_dim=19,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        graph_conditioning=True,
        global_pool_stats=True,
    ).eval()
    assert torch.equal(
        base.decoder.head[2].weight,
        stats.decoder.head[2].weight,
    )
    state = {
        key: value
        for key, value in stats.state_dict().items()
        if "raw_pool" not in key
    }
    base.load_state_dict(state)
    features, adjacency, mask = graph_conditioned_features(batch)
    expected = base.encode_target(
        batch["target_latent"],
        features,
        adjacency,
        mask,
    )
    observed = stats.encode_target(
        batch["target_latent"],
        features,
        adjacency,
        mask,
    )
    assert torch.equal(observed, expected)

    with torch.no_grad():
        stats.target_encoder.raw_pool.normal_(std=0.01)
        stats.condition_encoder.raw_pool.normal_(std=0.01)
    changed = stats.encode_target(
        batch["target_latent"],
        features,
        adjacency,
        mask,
    )
    assert not torch.equal(changed, expected)
    assert stats.target_encoder.raw_pool.grad is None
    loss = changed.square().mean()
    loss.backward()
    assert stats.target_encoder.raw_pool.grad is not None


def test_global_film_is_zero_initialized_and_receives_gradient() -> None:
    batch = _batch()
    features, adjacency, mask = graph_conditioned_features(batch)
    torch.manual_seed(29)
    base = _model().eval()
    torch.manual_seed(29)
    film = ChipSlotFlowModel(
        feature_dim=19,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        graph_conditioning=True,
        global_film=True,
    ).eval()
    state = {
        key: value
        for key, value in film.state_dict().items()
        if "global_film" not in key
    }
    base.load_state_dict(state)
    global_state = torch.randn(2, 32)
    condition = torch.randn(2, 32)
    expected = base.decode(
        global_state,
        condition,
        features,
        adjacency,
        mask,
    )
    observed = film.decode(
        global_state,
        condition,
        features,
        adjacency,
        mask,
    )
    assert torch.equal(observed, expected)

    with torch.no_grad():
        film.decoder.global_film.normal_(std=0.01)
    observed = film.decode(
        global_state,
        condition,
        features,
        adjacency,
        mask,
    )
    assert not torch.equal(observed, expected)
    observed.square().mean().backward()
    assert film.decoder.global_film.grad is not None
