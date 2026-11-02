from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from floorflow.chip_flow.flow_adapter import build_chip_model
from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.training_scopes import (
    ATOM_SCOPE_PREFIXES,
    ChannelStageController,
    apply_training_scope,
    parse_scope_schedule,
    parse_training_scope,
    scope_at_step,
    scope_owners,
    typed_channel_encoding_requires_grad,
    validate_atom_isolation_schedule,
)

ROOT = Path(__file__).resolve().parents[1]


def _model() -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=16,
        heads=2,
        depth=1,
        head_dim=16,
        typed_layers=1,
        global_spec_conditioning=True,
        node_semantic_dim=2,
        port_semantic_dim=2,
        complexity_conditioning=True,
        degree_expert_conditioning=True,
    )


def _batch() -> dict:
    constraints = torch.zeros(2, 4, 5)
    constraints[:, 0, 0] = 1.0
    constraints[:, :2, 2] = 1.0
    constraints[:, :, 4] = torch.tensor([1.0, 2.0, 0.0, 0.0])
    return {
        "areas": torch.ones(2, 4),
        "boxes": torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0],
             [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        )[None].expand(2, -1, -1),
        "constraints": constraints,
        "features": torch.zeros(2, 4, 19),
        "b2b": torch.tensor([[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]]).expand(2, -1, -1),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0]]]).expand(2, -1, -1),
        "pins": torch.tensor([[[0.5, 0.5]]]).expand(2, -1, -1),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([1, 1]),
        "n_pins": torch.tensor([1, 1]),
        "global_spec": torch.zeros(2, 7),
        "node_features": torch.zeros(2, 4, 2),
        "port_features": torch.zeros(2, 1, 2),
    }


def test_fusion_scope_separates_cross_attention_from_channel_encoders() -> None:
    model = _model()
    stats = apply_training_scope(model, parse_training_scope("fusion"))
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith(("typed.", "time."))
        for name in trainable
    )
    assert all(
        name.startswith(
            (
                "typed.cross_attention.",
                "typed.norm.",
                "time.",
            )
        )
        for name in trainable
    )
    assert not typed_channel_encoding_requires_grad(model)
    assert stats["scope"] == "fusion"


def test_fusion_scope_runs_channel_encoding_without_autograd() -> None:
    model = _model()
    apply_training_scope(model, parse_training_scope("fusion"))
    context = model.prepare_context(_batch())
    context.typed.block_tokens.square().mean().backward()
    assert model.typed.cross_attention.q.weight.grad is not None
    assert model.typed.block_input[0].weight.grad is None


def test_channel_scope_can_be_combined_with_downstream_training() -> None:
    model = _model()
    scope = parse_training_scope("channel:local+downstream")
    apply_training_scope(model, scope)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert any(name.startswith("typed.local_geometry_encoder.") for name in trainable)
    assert any(not name.startswith("typed.") for name in trainable)
    assert not any(name.startswith("typed.cross_attention.") for name in trainable)
    assert typed_channel_encoding_requires_grad(model)


def test_scope_schedule_is_monotonic_and_step_resolved() -> None:
    schedule = parse_scope_schedule(
        "0:channel:local+downstream,100:fusion+downstream,200:all"
    )
    assert scope_at_step(schedule, 0).name == "local+downstream"
    assert scope_at_step(schedule, 99).name == "local+downstream"
    assert scope_at_step(schedule, 100).name == "fusion+downstream"
    assert scope_at_step(schedule, 250).name == "all"
    with pytest.raises(ValueError, match="increasing"):
        parse_scope_schedule("0:all,0:fusion")


def test_stage_controller_reopens_exact_scope_at_transition() -> None:
    model = _model()
    args = SimpleNamespace(training_scope="all", training_scope_schedule="0:fusion,2:all")
    controller = ChannelStageController.from_args(args)
    controller.apply_initial(model)
    assert controller.current.name == "fusion"
    assert not controller.update(model, step=1)
    assert controller.update(model, step=2)
    assert controller.current.name == "all"
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert [row["step"] for row in controller.history] == [0, 2]


def test_default_all_scope_preserves_explicit_freeze_flags() -> None:
    model = _model()
    model.typed.block_input[0].weight.requires_grad_(False)
    controller = ChannelStageController.from_args(
        SimpleNamespace(training_scope="all", training_scope_schedule=None)
    )
    controller.apply_initial(model)
    assert not model.typed.block_input[0].weight.requires_grad
    assert controller.stats["mode"] == "preserve_existing_flags"


def test_semantic_area_scope_selects_owned_atoms_only() -> None:
    model = _model()
    apply_training_scope(model, parse_training_scope("area+downstream"))
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert any(name.startswith("typed.global_spec_conditioner.") for name in trainable)
    assert any(name.startswith("typed.complexity.") for name in trainable)
    assert not any(name.startswith("typed.cross_attention.") for name in trainable)
    assert any(not name.startswith("typed.") for name in trainable)


def test_b2b_atom_scope_owns_dynamic_graph_state_path() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=16,
        heads=2,
        depth=1,
        head_dim=16,
        typed_layers=1,
        dynamic_graph_input=True,
    )

    apply_training_scope(model, parse_training_scope("b2b"))

    assert model.dynamic_relay.gate.requires_grad
    assert model.dynamic_to_latent.weight.requires_grad
    assert "dynamic_relay." in ATOM_SCOPE_PREFIXES["b2b"]
    assert "dynamic_to_latent." in ATOM_SCOPE_PREFIXES["b2b"]


def test_atom_isolation_requires_each_atom_once_then_fusion() -> None:
    schedule = parse_scope_schedule(
        "0:area,100:b2b,200:port_route,300:boundary,400:tail,"
        "500:fusion"
    )

    report = validate_atom_isolation_schedule(schedule)

    assert report["stages"] == 6
    assert report["final_scope"] == "fusion"
    assert set(report["atoms"].values()) == {1}


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("0:area+b2b,100:fusion", "one atom"),
        ("0:area,100:b2b,200:fusion", "each atom"),
        ("0:all", "shared scope"),
        ("0:fusion+downstream", "each atom"),
        (
            (
                "0:area,100:b2b,200:port_route,300:boundary,400:tail,"
                "500:fusion+downstream"
            ),
            "fusion only",
        ),
    ),
)
def test_atom_isolation_rejects_coupled_or_incomplete_schedules(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_atom_isolation_schedule(parse_scope_schedule(value))


def test_every_mainline_non_backbone_parameter_has_an_owner() -> None:
    config = json.loads(
        (ROOT / "configs/chip_architecture_mainline_v4.json").read_text()
    )
    model = build_chip_model(
        {
            **config["architecture"],
            "architecture": "typed_hierarchy",
            "hidden": 16,
            "heads": 2,
            "depth": 1,
            "head_dim": 16,
            **config["branches"],
        }
    )

    ownership = {
        name: scope_owners(name)
        for name, _parameter in model.named_parameters()
        if not name.startswith("backbone.")
    }
    unowned = [name for name, owners in ownership.items() if not owners]
    duplicated = [
        (name, owners)
        for name, owners in ownership.items()
        if len(owners) > 1
    ]

    assert unowned == []
    assert duplicated == []


def test_scope_parser_rejects_unknown_and_empty_groups() -> None:
    with pytest.raises(ValueError, match="unknown"):
        parse_training_scope("channel:imaginary")
    with pytest.raises(ValueError, match="empty"):
        parse_training_scope(" + ")
