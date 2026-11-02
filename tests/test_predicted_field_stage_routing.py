from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.model.predicted_channel_model import PredictedChannelModel
from floorflow.model.predicted_control_prior import predicted_control_prior


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Kitchen": 4}),
            "furniture": Counter({"Sink": 4}),
            "door": Counter({"Swing": 4}),
            "window": Counter({"Regular": 4}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _controls(taxonomy, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "room_logits": torch.randn(
            2,
            len(taxonomy.room_ids),
            16,
            16,
            device=device,
        ),
        "structure": torch.randn(
            2,
            len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
            16,
            16,
            device=device,
        ),
        "furniture": torch.randn(
            2,
            len(taxonomy.furniture_ids) + 1,
            16,
            16,
            device=device,
        ),
        "support": torch.randn(2, 2, 16, 16, device=device),
        "program": torch.zeros(
            2,
            2,
            device=device,
            dtype=torch.bool,
        ),
    }


def _model(taxonomy, device: torch.device) -> PredictedChannelModel:
    return PredictedChannelModel(
        taxonomy,
        control_channels=(
            len(taxonomy.room_ids),
            len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
            len(taxonomy.furniture_ids) + 1,
            2,
        ),
        program_channels=2,
        grid=8,
        field_channels=8,
        head_channels=16,
        heads=4,
        layers=1,
        attention_grid=4,
        full_resolution_output=False,
        typed_channel_links=True,
        use_action_graph=True,
        preserve_room_stage=False,
        control_evidence_context=True,
        action_context_conditioning=True,
        typed_source_spatial_router=True,
        refresh_action_tokens=True,
    ).to(device)


def test_typed_source_router_is_spatial_sample_conditioned_and_trainable() -> None:
    taxonomy = _taxonomy()
    device = _device()
    model = _model(taxonomy, device)
    with torch.no_grad():
        model.action_graph.update_scale.fill_(0.1)
    controls = _controls(taxonomy, device)
    footprint = torch.ones(2, 16, 16, device=device)
    first = model(controls, footprint)
    second_controls = {
        key: value.clone()
        for key, value in controls.items()
    }
    second_controls["structure"] = torch.randn_like(
        second_controls["structure"]
    )
    second = model(second_controls, footprint)
    first_weights = first["field_source_routing"]["source_weights"]
    second_weights = second["field_source_routing"]["source_weights"]
    assert first_weights.shape == (2, 4, 4, 8, 8)
    assert torch.allclose(
        first_weights.sum(dim=2),
        torch.ones_like(first_weights[:, :, 0]),
    )
    assert not torch.allclose(first_weights, second_weights)
    first["logits"].square().mean().backward()
    gradient = model.field.source_router.family_logits["walls"].weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0
    fusion_gradient = model.field.source_router.family_fusion.weight.grad
    assert fusion_gradient is not None
    assert torch.count_nonzero(fusion_gradient) > 0
    action_gradient = model.action_context_projection.weight.grad
    assert action_gradient is not None
    assert torch.count_nonzero(action_gradient) > 0
    with torch.no_grad():
        model.action_context_projection.weight.add_(0.01)
    third = model(controls, footprint)
    assert not torch.allclose(first["logits"], third["logits"])


def test_refreshed_action_tokens_make_wall_stage_room_causal() -> None:
    taxonomy = _taxonomy()
    device = _device()
    model = _model(taxonomy, device)
    with torch.no_grad():
        model.action_graph.update_scale.fill_(0.1)
    controls = _controls(taxonomy, device)
    footprint = torch.ones(2, 16, 16, device=device)
    with torch.no_grad():
        first = model(controls, footprint)
        model.stage_heads["rooms"].weight.fill_(0.2)
        second = model(controls, footprint)
    assert not torch.allclose(
        first["stage_rooms_logits"],
        second["stage_rooms_logits"],
    )
    assert not torch.allclose(
        first["action_tokens"]["walls"],
        second["action_tokens"]["walls"],
    )
    assert float(first["action_refresh_count"]) == 3.0


def test_half_precision_controls_produce_finite_float_prior() -> None:
    taxonomy = _taxonomy()
    device = _device()
    controls = {
        "room_logits": torch.rand(
            1,
            len(taxonomy.room_ids),
            16,
            16,
            device=device,
            dtype=torch.float16,
        ),
        "structure": torch.rand(
            1,
            len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
            8,
            8,
            device=device,
            dtype=torch.float16,
        ),
        "furniture": torch.rand(
            1,
            len(taxonomy.room_ids) + len(taxonomy.furniture_ids) + 1,
            8,
            8,
            device=device,
            dtype=torch.float16,
        ),
    }
    prior = predicted_control_prior(taxonomy, controls, 8)
    assert prior.dtype == torch.float32
    assert prior.shape == (1, taxonomy.num_classes, 8, 8)
    assert torch.isfinite(prior).all()
