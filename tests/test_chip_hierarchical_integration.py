from __future__ import annotations

from types import SimpleNamespace

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import build_chip_model, sample_for_batch
from floorflow.chip_flow.guided_sampling import sample_hierarchy_guided_flow
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.train_objective import (
    hierarchy_bbox_schedule,
    supervised_loss,
)


def _batch() -> dict:
    count = 6
    boxes = torch.zeros(3, count, 4)
    boxes[..., 0] = torch.arange(count) * 1.1
    boxes[..., 1] = torch.arange(count).remainder(2) * 0.8
    boxes[..., 2:] = 1.0
    constraints = torch.zeros(3, count, 5)
    constraints[:, :, 3] = torch.tensor(
        [[1, 1, 2, 2, 3, 3]] * 3
    )
    constraints[:, :1, 4] = 1
    constraints[:, -1:, 4] = 2
    return {
        "n": count,
        "areas": torch.ones(3, count),
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(3, count, 10),
        "source_latent": boxes[..., :3].clone(),
        "target_latent": boxes[..., [0, 1, 2]].clone(),
        "b2b": torch.tensor(
            [
                [[0, 1, 1], [1, 2, 0.5]],
                [[1, 2, 1], [2, 3, 0.5]],
                [[2, 3, 1], [3, 4, 0.5]],
            ],
            dtype=torch.float32,
        ),
        "p2b": torch.tensor(
            [
                [[0, 0, 1], [1, 1, 0.5]],
                [[0, 1, 1], [1, 2, 0.5]],
                [[0, 2, 1], [1, 3, 0.5]],
            ],
            dtype=torch.float32,
        ),
        "pins": torch.tensor(
            [
                [[0.2, 0.4], [0.8, 0.6]],
                [[0.1, 0.5], [0.7, 0.9]],
                [[0.3, 0.2], [0.6, 0.8]],
            ]
        ),
        "n_b2b": torch.tensor([2, 2, 2]),
        "n_p2b": torch.tensor([2, 2, 2]),
        "n_pins": torch.tensor([2, 2, 2]),
    }


def _small_model() -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=24,
        heads=3,
        depth=1,
        head_dim=8,
        typed_layers=1,
    )


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        path_steps=0,
        time_std=1.0,
        flow_huber_beta=0.1,
        hierarchy_global_weight=0.1,
        hierarchy_structural_weight=0.1,
        hierarchy_port_weight=0.1,
        hierarchy_bbox_weight=0.5,
        hierarchy_bbox_decay_steps=0,
        physics_weight=1.0,
        overlap_weight_scale=1.0,
        decode_weight=1.0,
        clean_weight=0.1,
        rho_weight=0.0,
        path_weight=0.0,
        sparsity_weight=0.0,
        route_balance_weight=0.0,
    )


def test_matched_configuration_is_within_one_percent() -> None:
    model = build_chip_model(
        {
            "architecture": "typed_hierarchy",
            "hidden": 192,
            "heads": 3,
            "depth": 10,
            "head_dim": 64,
            "graph_pe_k": 2,
            "typed_layers": 1,
        }
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert abs(parameters / 5_856_400 - 1.0) <= 0.01


def test_one_step_loss_backward_is_finite() -> None:
    torch.manual_seed(4)
    model = _small_model()
    loss, terms = supervised_loss(
        model,
        _batch(),
        _args(),
        torch.device("cpu"),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert {
        "flow_block",
        "flow_global",
        "flow_structural",
        "flow_port",
        "hierarchy_bbox",
    } <= set(terms)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainable
    )


def test_bbox_alignment_schedule_decays_to_zero() -> None:
    assert hierarchy_bbox_schedule(0.5, 0, 5000) == 0.5
    assert hierarchy_bbox_schedule(0.5, 10000, 0) == 0.5
    assert hierarchy_bbox_schedule(0.5, 10000, 5000) == 0.25
    assert hierarchy_bbox_schedule(0.5, 10000, 10000) == 0.0


def test_sampling_does_not_read_free_block_targets() -> None:
    torch.manual_seed(9)
    model = _small_model().eval()
    batch = _batch()
    changed = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    changed["target_latent"].add_(8.0)
    generator_a = torch.Generator().manual_seed(19)
    generator_b = torch.Generator().manual_seed(19)
    first, _ = sample_for_batch(
        model,
        batch,
        sampling_steps=2,
        generator=generator_a,
    )
    second, _ = sample_for_batch(
        model,
        changed,
        sampling_steps=2,
        generator=generator_b,
    )
    assert torch.allclose(first, second)


def test_checkpoint_loader_restores_typed_hierarchy() -> None:
    model = _small_model()
    config = {
        "architecture": "typed_hierarchy",
        "hidden": 24,
        "heads": 3,
        "depth": 1,
        "head_dim": 8,
        "graph_pe_k": 2,
        "typed_layers": 1,
        "projector": False,
        "projector_iterations": 4,
    }
    checkpoint = {"config": config, "model": model.state_dict()}
    restored = load_chip_checkpoint(
        checkpoint,
        "cpu",
        use_ema=False,
    )
    assert isinstance(restored, HierarchicalTokenFlowModel)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_checkpoint_loader_accepts_legacy_typed_bias_default() -> None:
    model = _small_model()
    config = {
        "architecture": "typed_hierarchy",
        "hidden": 24,
        "heads": 3,
        "depth": 1,
        "head_dim": 8,
        "graph_pe_k": 2,
        "typed_layers": 1,
        "typed_mode": "typed",
        "projector": False,
        "projector_iterations": 4,
    }
    state = model.state_dict()
    state.pop("typed.cross_attention.type_bias")
    restored = load_chip_checkpoint(
        {"config": config, "model": state},
        "cpu",
        use_ema=False,
    )
    assert torch.equal(
        restored.typed.cross_attention.type_bias,
        torch.zeros(4),
    )


def test_no_netlist_context_ignores_b2b_edges() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=24,
        heads=3,
        depth=1,
        head_dim=8,
        typed_layers=1,
        use_netlist=False,
    ).eval()
    batch = _batch()
    changed = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    changed["b2b"].add_(5.0)
    with torch.no_grad():
        first = model.prepare_context(batch)
        second = model.prepare_context(changed)
    assert torch.equal(first.denoiser_cond.adjacency, second.denoiser_cond.adjacency)
    assert torch.equal(first.typed.block_tokens, second.typed.block_tokens)


def test_guided_sampling_produces_finite_proxy_correction() -> None:
    model = _small_model().eval()
    latent, diagnostics = sample_hierarchy_guided_flow(
        model,
        _batch(),
        sampling_steps=2,
        generator=torch.Generator().manual_seed(19),
        guidance_scale=0.02,
    )
    assert torch.isfinite(latent).all()
    assert torch.isfinite(diagnostics["guidance_energy_mean"])
    assert diagnostics["guidance_gradient_rms_mean"] > 0.0
    assert diagnostics["guidance_update_rms_mean"] > 0.0


def test_no_hierarchy_removes_cross_level_block_residual() -> None:
    model = HierarchicalTokenFlowModel(
        hidden=24,
        heads=3,
        depth=1,
        head_dim=8,
        typed_layers=1,
        use_hierarchy=False,
    ).eval()
    model.typed.cross_attention.gate.data.fill_(1.0)
    batch = _batch()
    context = model.prepare_context(batch)
    hierarchy = build_hierarchy_targets(batch, context)
    with torch.no_grad():
        prediction = model(hierarchy, torch.ones(3), context)
        baseline = model.backbone(
            hierarchy.block_state,
            torch.ones(3),
            context.denoiser_cond,
        )
    assert torch.equal(prediction.block_state, baseline)
