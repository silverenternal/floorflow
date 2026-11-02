from __future__ import annotations

from collections import Counter

import pytest
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.masked_channel_distribution import (
    MaskedChannelDistributionHead,
)
from floorflow.model.predicted_channel_model import PredictedChannelModel
from floorflow.model.predicted_optional_stages import (
    apply_opening_host_gate,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Bath": 10}),
            "furniture": Counter({"Bed": 6}),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_preserved_room_stage_keeps_typed_channel_residual() -> None:
    taxonomy = _taxonomy()
    model = PredictedChannelModel(
        taxonomy,
        control_channels=(2, 6, 2, 2),
        program_channels=2,
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=True,
        typed_channel_links=True,
    )
    controls = {
        "room_logits": torch.randn(2, 2, 16, 16),
        "structure": torch.randn(2, 6, 16, 16),
        "furniture": torch.randn(2, 2, 16, 16),
        "support": torch.randn(2, 2, 16, 16),
        "program": torch.tensor([[1, 0], [0, 1]], dtype=torch.bool),
    }
    output = model(controls, torch.ones(2, 16, 16))
    room_ids = torch.tensor(taxonomy.room_ids)
    assert not torch.allclose(
        output["stage_rooms_logits"],
        output["prior_logits"][:, room_ids],
    )
    assert torch.allclose(
        output["stage_rooms_logits"],
        output["room_logits"],
    )
    output["logits"].square().mean().backward()
    assert model.head.typed_channel_residual.grad is not None


def test_background_query_owns_spatial_logits_and_gradient() -> None:
    taxonomy = _taxonomy()
    head = MaskedChannelDistributionHead(
        8,
        taxonomy.num_classes,
        hidden_channels=16,
        heads=4,
        layers=1,
        grid=4,
        full_resolution_output=True,
    )
    features = torch.randn(1, 8, 8, 8)
    output = head(features)
    assert output["logits"].shape[1] == taxonomy.num_classes
    assert torch.count_nonzero(output["logits"][:, 0]) > 1
    output["logits"][:, 0].square().mean().backward()
    assert head.channel_queries.grad is not None
    assert torch.count_nonzero(head.channel_queries.grad[0]) > 0


def test_opening_host_gate_is_identity_initialized_and_host_sensitive() -> None:
    opening_logits = torch.zeros(1, 2, 8, 8)
    wall = torch.zeros(1, 1, 8, 8)
    wall[:, :, :, :4] = 0.9
    wall.requires_grad_()
    rooms = torch.zeros(1, 2, 8, 8)
    rooms[:, 0, :, :4] = 0.9
    rooms[:, 1, :, 4:] = 0.9
    junction = torch.zeros(1, 8, 8)
    junction[:, :, 3] = 0.8
    identity, support = apply_opening_host_gate(
        opening_logits,
        torch.zeros(()),
        wall,
        rooms,
        junction,
    )
    assert torch.equal(identity, opening_logits)
    gated, _support = apply_opening_host_gate(
        opening_logits,
        torch.ones(()),
        wall,
        rooms,
        junction,
    )
    assert support[..., :4].mean() > support[..., 4:].mean()
    assert gated[..., :4].mean() > gated[..., 4:].mean()
    gated.sum().backward()
    assert wall.grad is not None
    assert torch.count_nonzero(wall.grad) > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "use_action_graph": False,
                "enable_instance_room_decoder": True,
            },
            "instance room decoding",
        ),
        (
            {
                "use_action_graph": False,
                "refresh_action_tokens": True,
            },
            "action-token refresh",
        ),
    ],
)
def test_predicted_channel_rejects_silent_consumer_disconnects(
    kwargs: dict[str, bool],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PredictedChannelModel(_taxonomy(), **kwargs)


def test_preserved_room_stage_context_reaches_room_instances() -> None:
    taxonomy = _taxonomy()
    model = PredictedChannelModel(
        taxonomy,
        control_channels=(2, 6, 2, 2),
        program_channels=2,
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=True,
        typed_channel_links=True,
        use_action_graph=True,
        preserve_room_stage=True,
        dynamic_stage_heads=True,
        stage_head_hidden=16,
        enable_instance_room_decoder=True,
        room_instance_hidden=16,
        room_instance_queries=4,
        room_instance_layers=1,
    )
    with torch.no_grad():
        model.stage_refiners["rooms"].output.weight.normal_(std=0.02)
    controls = {
        "room_logits": torch.randn(1, 2, 16, 16),
        "structure": torch.randn(1, 6, 2, 2)
        .repeat_interleave(8, dim=-2)
        .repeat_interleave(8, dim=-1),
        "furniture": torch.randn(1, 2, 16, 16),
        "support": torch.randn(1, 2, 16, 16),
        "program": torch.tensor([[1, 0]], dtype=torch.bool),
    }
    output = model(controls, torch.ones(1, 16, 16))
    assert not torch.allclose(
        output["stage_rooms_candidate_logits"],
        output["stage_rooms_logits"],
    )
    assert output["mask_class_evidence"].shape == (1, 4, 2)
    output["room_instance_logits"].square().mean().backward()
    grad = model.stage_refiners["rooms"].output.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0.0


def test_dynamic_query_and_contact_graph_are_optional_live_paths() -> None:
    taxonomy = _taxonomy()
    model = PredictedChannelModel(
        taxonomy,
        control_channels=(2, 6, 2, 2),
        program_channels=2,
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=True,
        typed_channel_links=True,
        use_action_graph=True,
        enable_instance_room_decoder=True,
        room_instance_hidden=16,
        room_instance_queries=4,
        room_instance_layers=1,
        enable_room_adjacency=True,
        room_instance_adaptive_query=True,
        room_instance_adaptive_query_residual_scale=0.05,
        room_instance_mask_aware_adjacency=True,
        room_instance_count_aware_graph=True,
        typed_opening_host_gate=True,
        refresh_action_tokens=True,
        refresh_validation_tokens=True,
    )
    controls = {
        "room_logits": torch.randn(1, 2, 16, 16),
        "structure": torch.randn(1, 6, 16, 16),
        "furniture": torch.randn(1, 2, 16, 16),
        "support": torch.randn(1, 2, 16, 16),
        "junction_probability": torch.randn(1, 16, 16),
        "program": torch.tensor([[1, 0]], dtype=torch.bool),
    }
    output = model(controls, torch.ones(1, 16, 16))
    assert "adaptive_query_diagnostics" in output
    assert output["room_adjacency_mask_contact_logits"].shape == (1, 4, 4)
    assert output["room_graph_query_presence"].shape == (1, 4)
    assert "room_graph_first_class_residual" in output
    assert "room_graph_first_mask_residual" in output
    assert output["opening_host_support"].shape == (
        1,
        1,
        *output["stage_openings_logits"].shape[-2:],
    )
    assert float(output["action_refresh_count"]) == 4.0
    assert float(output["action_validation_refreshed"]) == 1.0
    (
        output["room_adjacency_logits"].sum()
        + output["stage_openings_logits"].sum()
    ).backward()
    adjacency_head = model.room_instance_stage.room_adjacency_head
    assert adjacency_head is not None
    assert adjacency_head.contact_scale is not None
    assert adjacency_head.contact_scale.grad is not None
    assert model.opening_host_gate_scale is not None
    assert model.opening_host_gate_scale.grad is not None
    assert model.opening_host_gate_scale.grad.abs() > 0.0
