import torch

from floorflow.chip_flow.hierarchical_model import (
    ChipHierarchicalSlotFlowModel,
)
from floorflow.chip_flow.slot_model import ChipSlotFlowModel


def _model() -> ChipSlotFlowModel:
    return ChipSlotFlowModel(
        feature_dim=10,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
    )


def test_global_encoding_and_slot_decoding_are_permutation_equivariant() -> None:
    torch.manual_seed(7)
    model = _model().eval()
    features = torch.randn(2, 9, 10)
    target = torch.randn(2, 9, 3)
    permutation = torch.tensor([4, 0, 8, 2, 7, 1, 6, 3, 5])

    global_state = model.encode_target(target, features)
    permuted_global = model.encode_target(
        target[:, permutation],
        features[:, permutation],
    )
    condition = model.encode_condition(features)
    decoded = model.decode(global_state, condition, features)
    permuted_decoded = model.decode(
        permuted_global,
        model.encode_condition(features[:, permutation]),
        features[:, permutation],
    )

    assert torch.allclose(global_state, permuted_global, atol=1e-5)
    assert torch.allclose(decoded[:, permutation], permuted_decoded, atol=1e-5)


def test_global_flow_has_expected_shapes() -> None:
    model = _model()
    state = torch.randn(3, 32)
    condition = torch.randn(3, 32)
    time = torch.rand(3)

    velocity = model(state, time, condition)
    sampled = model.sample(
        torch.randn(3, 7, 10),
        sampling_steps=2,
        generator=torch.Generator().manual_seed(3),
    )

    assert velocity.shape == (3, 32)
    assert sampled.shape == (3, 7, 3)


def test_hierarchical_flow_is_permutation_equivariant() -> None:
    torch.manual_seed(11)
    model = ChipHierarchicalSlotFlowModel(
        feature_dim=10,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        num_slots=4,
    ).eval()
    features = torch.randn(2, 9, 10)
    target = torch.randn(2, 9, 3)
    permutation = torch.tensor([4, 0, 8, 2, 7, 1, 6, 3, 5])

    state = model.encode_target(target, features)
    condition = model.encode_condition(features)
    decoded = model.decode(state, condition, features)
    permuted = model.decode(
        model.encode_target(target[:, permutation], features[:, permutation]),
        model.encode_condition(features[:, permutation]),
        features[:, permutation],
    )

    assert torch.allclose(decoded[:, permutation], permuted, atol=1e-5)
    sampled = model.sample(
        features,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(5),
    )
    assert sampled.shape == (2, 9, 3)
    counts = model.active_slot_counts(features)
    assert counts.shape == (2,)
    assert bool(((counts >= 1) & (counts <= 4)).all())
