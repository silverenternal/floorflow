from __future__ import annotations

from collections import Counter

import pytest
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
    load_predicted_channel_state_dict,
    predicted_channel_model_kwargs,
)


def _config() -> dict:
    return {
        "control_channels": (2, 6, 2, 2),
        "program_channels": 2,
        "grid": 8,
        "field_channels": 8,
        "head_channels": 16,
        "heads": 4,
        "layers": 1,
        "attention_grid": 4,
        "full_resolution_output": False,
        "typed_channel_links": True,
        "use_action_graph": True,
        "control_evidence_context": True,
        "dynamic_stage_heads": True,
        "stage_head_hidden": 16,
        "dynamic_stage_links": True,
        "action_route_attention": True,
        "action_route_router_count": 4,
        "action_route_top_k": 2,
        "enable_instance_room_decoder": True,
        "room_instance_hidden": 8,
        "room_instance_queries": 2,
        "room_instance_layers": 1,
    }


def test_old_typed_checkpoint_shapes_migrate_with_identity_initialization() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter({"Sink": 2}),
            "door": Counter({"Swing": 2}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    source_model = build_predicted_channel_model(taxonomy, _config())
    state = {
        name: value.detach().clone()
        for name, value in source_model.state_dict().items()
    }
    evidence = "field.evidence_fusion.0.weight"
    state[evidence] = state[evidence][:, : 8 * 9]
    condition = "stage_refiners.openings.condition_projection.weight"
    state[condition] = state[condition][:, :1]
    state["room_instance_stage.residual_scale"] = state[
        "room_instance_stage.residual_scale"
    ].reshape(1)
    state.pop("action_graph.update_scale")
    state.pop("room_instance_stage.class_count_prior_scale")

    target = build_predicted_channel_model(taxonomy, _config())
    report = load_predicted_channel_state_dict(target, state)

    assert evidence in report["migrations"]
    assert condition in report["migrations"]
    assert "room_instance_stage.residual_scale" in report["migrations"]
    assert not report["missing_keys"]
    assert not report["unexpected_keys"]
    assert torch.allclose(
        target.action_graph.update_scale,
        torch.tensor(10.0),
    )
    assert torch.count_nonzero(
        target.state_dict()[evidence][:, 8 * 9 :]
    ) == 0
    assert torch.count_nonzero(
        target.state_dict()[condition][:, 1:]
    ) == 0


def test_unknown_shape_mismatch_fails_closed() -> None:
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
    target = build_predicted_channel_model(taxonomy, _config())
    state = dict(target.state_dict())
    name = "head.input_projection.weight"
    if name not in state:
        name = next(
            key
            for key, value in state.items()
            if value.ndim >= 2 and "weight" in key and value.numel() > 1
        )
    state[name] = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="unsupported checkpoint shape"):
        load_predicted_channel_state_dict(target, state)


def test_checkpoint_config_preserves_relative_owner_mode() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    config = _config()
    config["owner_preserving_room_readout"] = "relative"
    kwargs = predicted_channel_model_kwargs(config)
    assert kwargs["owner_preserving_room_readout"] == "relative"
    model = build_predicted_channel_model(taxonomy, config)
    assert model.room_instance_stage.owner_readout_mode == "relative"
