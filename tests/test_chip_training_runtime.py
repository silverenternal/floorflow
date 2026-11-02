import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from floorflow.chip_flow.chip_checkpoint_io import save_chip_checkpoint
from floorflow.chip_flow.hierarchy_coordinates import HierarchyBlockCoordinateGate
from floorflow.chip_flow.reproducibility import stable_batch_seed
from floorflow.chip_flow.training_runtime import (
    clip_optimizer_gradients_fail_closed,
    ema_decay_at_step,
    ema_gap_by_prefix,
    gradient_norm_by_prefix,
    initial_ema_state,
    make_optimizer,
    parameter_delta_by_prefix,
    parameter_probe_by_prefix,
    save_torch_atomic,
    set_optimizer_group_lrs,
    update_ema_state_,
    write_json_atomic,
)


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.LayerNorm(8),
        torch.nn.Linear(8, 3),
    )


def test_chip_optimizer_keeps_requested_hyperparameters() -> None:
    model = _tiny_model()
    optimizer = make_optimizer(
        model,
        lr=2.0e-4,
        weight_decay=1.0e-3,
        betas=(0.9, 0.95),
        fused=False,
    )
    assert optimizer.defaults["lr"] == 2.0e-4
    assert optimizer.defaults["weight_decay"] == 1.0e-3
    assert optimizer.defaults["betas"] == (0.9, 0.95)


def test_chip_optimizer_scales_only_backbone_parameters() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.head = torch.nn.Linear(4, 3)

    model = Model()
    optimizer = make_optimizer(
        model,
        lr=2.0e-4,
        weight_decay=0.0,
        fused=False,
        backbone_lr_scale=0.1,
    )

    groups = {
        id(parameter): float(group["lr_scale"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert groups[id(model.backbone.weight)] == 0.1
    assert groups[id(model.head.weight)] == 1.0


def test_foreach_ema_matches_reference_update() -> None:
    model = _tiny_model()
    reference = copy.deepcopy(model).eval()
    ema = initial_ema_state(reference)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.25)
    expected = {
        name: value.detach().clone()
        for name, value in reference.state_dict().items()
    }
    with torch.no_grad():
        model_state = model.state_dict()
        for name, value in expected.items():
            source = model_state[name]
            if torch.is_floating_point(value):
                value.mul_(0.7).add_(source, alpha=0.3)
            else:
                value.copy_(source)
    update_ema_state_(ema, model.state_dict(), 0.7)
    for name, value in expected.items():
        assert torch.allclose(ema[name], value, atol=1.0e-7), name


def test_initial_ema_state_is_independent_of_source() -> None:
    model = _tiny_model()
    ema = initial_ema_state(model)
    source = model.state_dict()
    for name, value in ema.items():
        assert value.data_ptr() != source[name].data_ptr(), name
    with torch.no_grad():
        for value in ema.values():
            value.add_(1.0)
    for name, value in ema.items():
        if torch.is_floating_point(value):
            assert not torch.equal(value, model.state_dict()[name]), name


def test_training_diagnostics_group_norms_by_top_level_module() -> None:
    model = _tiny_model()
    model(torch.ones(2, 4)).sum().backward()
    ema = initial_ema_state(model)
    probe = parameter_probe_by_prefix(model, limit=8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01)

    gradient_norms = gradient_norm_by_prefix(model)
    updates = parameter_delta_by_prefix(model, probe, limit=8)
    ema_gaps = ema_gap_by_prefix(model, ema)

    assert set(gradient_norms) == {"0", "1", "2"}
    assert all(value >= 0.0 for value in gradient_norms.values())
    assert all(value > 0.0 for value in updates.values())
    assert set(ema_gaps) == set(updates)


def test_atomic_json_and_torch_writes_replace_targets(tmp_path) -> None:
    json_path = tmp_path / "nested" / "report.json"
    checkpoint_path = tmp_path / "nested" / "checkpoint.pt"

    write_json_atomic(json_path, {"status": "complete"})
    save_torch_atomic(checkpoint_path, {"step": 3})

    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "status": "complete"
    }
    assert torch.load(checkpoint_path, weights_only=True) == {"step": 3}
    assert not list((tmp_path / "nested").glob(".*.tmp"))


def test_evaluation_batch_seed_is_stable_and_batch_specific() -> None:
    assert stable_batch_seed(19, [4, 5]) == stable_batch_seed(19, [4, 9])
    assert stable_batch_seed(19, [4, 5]) != stable_batch_seed(19, [5, 6])


def test_optimizer_groups_decay_heads_and_warmups() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.typed = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
            self.structured_heads = nn.Module()
            self.structured_heads.factorized_block = nn.Linear(4, 3)
            self.structured_heads.legalizer_response = nn.Sequential(
                nn.Linear(3, 2),
                nn.Linear(2, 2),
            )
            self.coordinate_gate = HierarchyBlockCoordinateGate(4)

    model = Model()
    optimizer = make_optimizer(
        model,
        lr=1.0e-4,
        weight_decay=1.0e-3,
        fused=False,
        backbone_lr_scale=0.1,
        structured_head_lr_scale=0.5,
        legalizer_head_lr_scale=0.25,
        structured_head_warmup_steps=10,
        legalizer_head_warmup_steps=20,
    )
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["backbone_decay"]["lr_scale"] == pytest.approx(0.1)
    assert groups["structured_decay"]["lr_scale"] == pytest.approx(0.5)
    assert groups["legalizer_decay"]["lr_scale"] == pytest.approx(0.25)
    assert groups["core_no_decay"]["weight_decay"] == 0.0
    assert groups["structured_no_decay"]["weight_decay"] == 0.0
    assert groups["legalizer_no_decay"]["weight_decay"] == 0.0
    no_decay_ids = {
        id(parameter) for parameter in groups["core_no_decay"]["params"]
    }
    assert all(
        id(parameter) in no_decay_ids
        for parameter in model.coordinate_gate.parameters()
    )
    applied = set_optimizer_group_lrs(optimizer, 1.0e-4, step=0)
    assert applied["structured_decay"] == pytest.approx(5.0e-6)
    assert applied["legalizer_decay"] == pytest.approx(1.25e-6)


def test_grouped_gradient_clipping_reports_pre_and_post_norms() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2))
    optimizer = make_optimizer(
        model,
        lr=1.0e-3,
        weight_decay=0.0,
        fused=False,
    )
    model(torch.randn(3, 4)).square().mean().backward()
    stats = clip_optimizer_gradients_fail_closed(
        model,
        optimizer,
        max_norm=0.5,
        per_group=True,
    )
    assert stats["global_pre_clip_norm"] > 0
    assert stats["global_post_clip_norm"] > 0
    assert stats["groups"]
    assert all(
        value["post_clip_norm"] <= 0.5 + 1.0e-6
        for value in stats["groups"].values()
        if value["pre_clip_norm"] > 0
    )


def test_grouped_gradient_clipping_fails_on_nonfinite_values() -> None:
    model = nn.Linear(2, 2)
    optimizer = make_optimizer(
        model,
        lr=1.0e-3,
        weight_decay=0.0,
        fused=False,
    )
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        clip_optimizer_gradients_fail_closed(
            model,
            optimizer,
            max_norm=1.0,
        )


def test_ema_modes_keep_fixed_control_and_warm_up() -> None:
    args = type(
        "Args",
        (),
        {"ema_decay": 0.9995, "ema_mode": "fixed", "ema_warmup_steps": 100},
    )()
    assert ema_decay_at_step(args, 10) == pytest.approx(0.9995)
    args.ema_mode = "warmup"
    assert ema_decay_at_step(args, 0) < ema_decay_at_step(args, 50)
    assert ema_decay_at_step(args, 100) == pytest.approx(0.9995)


def test_chip_checkpoint_persists_exact_resume_state(tmp_path, monkeypatch) -> None:
    class _Sampler:
        def state_dict(self) -> dict:
            return {"cursor": 7, "count": 21}

    checkpoint_module = __import__(
        "floorflow.chip_flow.chip_checkpoint_io",
        fromlist=["model_config"],
    )
    monkeypatch.setattr(
        checkpoint_module,
        "model_config",
        lambda _: {"architecture": "typed_hierarchy"},
    )
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    batch_rng = np.random.default_rng(321)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    ema = initial_ema_state(model)
    path = tmp_path / "checkpoint.pt"

    expected_torch = torch.get_rng_state()
    expected_numpy = np.random.get_state()
    expected_python = random.getstate()
    expected_batch = copy.deepcopy(batch_rng.bit_generator.state)
    save_chip_checkpoint(
        path,
        step=9,
        model=model,
        ema=ema,
        optimizer=optimizer,
        batch_rng=batch_rng,
        balanced_sampler=_Sampler(),
        data_order_hash="abc123",
        args=SimpleNamespace(device="cpu", grad_clip_per_group=True),
    )
    state = torch.load(path, map_location="cpu", weights_only=False)

    assert state["step"] == 9
    assert state["data_order_hash"] == "abc123"
    assert torch.equal(state["torch_rng"], expected_torch)
    np.testing.assert_array_equal(state["numpy_rng"][1], expected_numpy[1])
    assert state["numpy_rng"][2:] == expected_numpy[2:]
    assert state["python_rng"] == expected_python
    assert state["batch_rng_state"] == expected_batch
    assert state["balanced_sampler_state"] == {"cursor": 7, "count": 21}
    assert state["cuda_rng_state"] is None
    assert state["config"]["architecture"] == "typed_hierarchy"
    assert state["runtime"]["gradient_clipping"] is True

    expected_torch_value = torch.rand(3)
    expected_numpy_value = np.random.rand(3)
    expected_python_value = random.random()
    expected_batch_value = batch_rng.random(3)
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    restored_rng = np.random.default_rng(999)
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    restored_rng.bit_generator.state = state["batch_rng_state"]
    assert torch.equal(torch.rand(3), expected_torch_value)
    np.testing.assert_array_equal(np.random.rand(3), expected_numpy_value)
    assert random.random() == expected_python_value
    np.testing.assert_array_equal(restored_rng.random(3), expected_batch_value)
