from collections import Counter

import pytest
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.raster_control_bank import RasterInstanceControl
from floorflow.heads.room_slot_state import (
    RoomSlotStateHead,
    poisson_binomial_log_probability,
)
from floorflow.training.raster_instance_control import (
    configure_joint_slot_state_recovery,
)


def test_poisson_binomial_count_distribution_uses_slot_presence() -> None:
    presence = torch.tensor([[4.0, -4.0, 0.0], [0.0, 0.0, 0.0]])
    log_probability = poisson_binomial_log_probability(presence)
    probability = log_probability.exp()
    assert torch.allclose(
        probability.sum(dim=-1),
        torch.ones(presence.shape[0]),
        atol=1.0e-6,
    )
    counts = torch.arange(presence.shape[1] + 1, dtype=torch.float32)
    assert torch.allclose(
        (probability * counts).sum(dim=-1),
        presence.sigmoid().sum(dim=-1),
        atol=1.0e-5,
    )


def test_room_slot_state_is_identity_at_initialization() -> None:
    torch.manual_seed(9)
    features = torch.randn(2, 8, 16, 16)
    query_features = torch.randn(2, 4, 12)
    mask_logits = torch.randn(2, 4, 16, 16)
    class_logits = torch.randn(2, 4, 6)
    head = RoomSlotStateHead(12, 8, 5, 4, hidden_dim=16, heads=4)
    output = head(
        features,
        query_features,
        mask_logits,
        class_logits,
    )
    assert torch.equal(output["class_logits"], class_logits)
    assert torch.equal(output["mask_logits"], mask_logits)
    assert torch.allclose(
        output["query_existence_logits"],
        torch.logsumexp(class_logits[..., :-1], dim=-1)
        - class_logits[..., -1],
    )
    assert output["room_count_logits"].shape == (2, 5)
    assert output["slot_adjacency"].shape == (2, 4, 4)
    assert output["slot_owner_probability"].shape == mask_logits.shape
    assert torch.all(output["slot_owner_probability"].sum(dim=1) <= 1.0)


def test_room_slot_state_has_live_class_and_presence_gradients() -> None:
    torch.manual_seed(11)
    head = RoomSlotStateHead(12, 8, 5, 4, hidden_dim=16, heads=4)
    output = head(
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 4, 12),
        torch.randn(1, 4, 16, 16),
        torch.randn(1, 4, 6),
    )
    loss = output["class_logits"].square().mean()
    loss = loss + output["query_existence_logits"].square().mean()
    loss = loss + output["mask_logits"].square().mean()
    loss.backward()
    assert head.class_residual.weight.grad is not None
    assert head.presence_residual.weight.grad is not None
    assert head.mask_state_projection.weight.grad is not None
    assert head.class_residual.weight.grad.abs().sum() > 0
    assert head.presence_residual.weight.grad.abs().sum() > 0
    assert head.mask_state_projection.weight.grad.abs().sum() > 0


def test_raster_control_wires_joint_room_slot_state() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=16,
        hidden_dim=32,
        num_queries=4,
        layers=1,
        heads=4,
        joint_room_slot_state=True,
        joint_slot_owner_fusion=True,
    )
    footprint = torch.zeros(1, 64, 64)
    footprint[:, 8:56, 8:56] = 1.0
    output = model(footprint, control_image=torch.zeros(1, 3, 64, 64))
    assert model.joint_room_slot_state is not None
    assert output["room_count_logits"].shape == (1, 5)
    assert output["slot_state_features"].shape == (1, 4, 64)
    assert output["fused_logits"].shape == output["dense_logits"].shape
    assert "slot_state_diagnostics" in output


def test_joint_room_slot_state_rejects_legacy_query_set_head() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 1}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        RasterInstanceControl(
            taxonomy,
            num_queries=4,
            query_set_head=True,
            joint_room_slot_state=True,
        )


def test_joint_slot_recovery_freezes_every_parent_parameter() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 1}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    model = RasterInstanceControl(
        taxonomy,
        decoder_channels=16,
        hidden_dim=32,
        num_queries=4,
        layers=1,
        heads=4,
        joint_room_slot_state=True,
    )
    names = configure_joint_slot_state_recovery(model)
    assert names
    assert all(name.startswith("joint_room_slot_state.") for name in names)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("joint_room_slot_state.")
    )
