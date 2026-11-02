from __future__ import annotations

import json

import pytest
import torch
from pathlib import Path

from floorflow.model.budget import (
    BudgetExceededError,
    derive_kernel_candidates,
    estimate_output_head_params,
    select_output_kernels,
)
from floorflow.model.builder import (
    ModelMacro,
    load_state_into_model,
    map_legacy_state_dict,
)
from floorflow.model.registries import ModelRegistry
from floorflow.model.specs import ArchSpec, BudgetSpec


TINY_BACKBONE = {
    "semantic_channel_attention": False,
    "slot_mixing": False,
    "floor_conditioning": False,
    "topology_aux": False,
    "semantic_endpoint_head": False,
    "fourier_conditioning": False,
}


def _tiny_spec(**overrides) -> ArchSpec:
    payload = {
        "name": "tiny",
        "target_channels": 4,
        "base_channels": 16,
        "levels": 2,
        "resolution": (32, 32),
        "backbone_kwargs": TINY_BACKBONE,
    }
    payload.update(overrides)
    return ArchSpec(**payload)


def test_adaptive_kernels_scale_with_resolution_and_budget() -> None:
    low = derive_kernel_candidates((256, 256), target_feature_px=2)
    high = derive_kernel_candidates((1024, 1024), target_feature_px=2)
    assert low == (3, 5, 7, 9)
    assert high == (9, 11, 13, 15)

    two_scale_cost = estimate_output_head_params(64, 12, low[:2])
    selected = select_output_kernels(
        (256, 256),
        2,
        64,
        12,
        max_params=two_scale_cost,
        min_scales=2,
    )
    assert selected == low[:2]


def test_budget_validation_and_unknown_family() -> None:
    with pytest.raises(ValueError):
        BudgetSpec(max_params=0)
    with pytest.raises(KeyError):
        ModelMacro(_tiny_spec(family="missing")).build()


def test_builder_is_reproducible_and_fits_parameter_budget() -> None:
    spec = _tiny_spec(
        base_channels=32,
        budget=BudgetSpec(max_params=200_000, max_base_channels=32),
    )
    first = ModelMacro(spec).build_bundle()
    second = ModelMacro(spec).build_bundle()
    assert first.structure_signature == second.structure_signature
    assert first.resolved.base_channels < 32
    assert sum(first.parameter_table.values()) <= 200_000


def test_builder_refuses_impossible_budget() -> None:
    spec = _tiny_spec(budget=BudgetSpec(max_params=1_000))
    with pytest.raises(BudgetExceededError):
        ModelMacro(spec).resolve()


def test_compute_budget_is_measured_from_forward_shapes() -> None:
    spec = _tiny_spec(budget=BudgetSpec(max_macs=100_000_000))
    resolved = ModelMacro(spec).resolve()
    assert resolved.estimated_macs is not None
    assert 0 < resolved.estimated_macs <= 100_000_000


def test_legacy_state_dict_prefix_is_mapped() -> None:
    model = ModelMacro(_tiny_spec()).build()
    state = {f"module.{key}": value for key, value in model.state_dict().items()}
    mapped, renamed = map_legacy_state_dict(state, set(model.state_dict()))
    assert len(renamed) == len(state)
    assert set(mapped) == set(model.state_dict())
    report = load_state_into_model(model, state)
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []


def test_registry_rejects_duplicate_family() -> None:
    registry = ModelRegistry()
    registry.add("family", lambda spec, resolved: torch.nn.Identity())
    with pytest.raises(ValueError):
        registry.add("family", lambda spec, resolved: torch.nn.Identity())


def test_arch_spec_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "arch.json"
    path.write_text(json.dumps(_tiny_spec().to_dict()))
    loaded = ArchSpec.from_json(path)
    assert loaded.fingerprint() == _tiny_spec().fingerprint()


def test_shipped_macro_profiles_are_valid() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "model_macros"
    balanced = ArchSpec.from_json(root / "floorflow_256_balanced.json")
    high_budget = ArchSpec.from_json(root / "floorflow_512_high_budget.json")
    assert balanced.resolution == (256, 256)
    assert high_budget.resolution == (512, 512)
    assert high_budget.budget.max_params > balanced.budget.max_params
