from __future__ import annotations

from types import SimpleNamespace

import torch

import floorflow.chip_flow.shape_mode_objectives as shape_mode_module
from floorflow.chip_flow.branch_health import branch_health
from floorflow.chip_flow.branch_registry import enabled_branches
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import build_chip_model
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.hierarchy_sampling import sample_hierarchy_flow
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.shape_mode_head import (
    ShapeModeHead,
    apply_shape_modes,
    attach_shape_mode_head,
    mode_diversity_loss,
    mode_usage_loss,
    shape_mode_auxiliary_loss,
    shape_mode_latents_from_prediction,
    shape_mode_training_loss,
    winner_take_all_loss,
)
from floorflow.chip_flow.training_manifest import (
    RESUME_CRITICAL_KEYS,
    RESUME_NEW_PARAM_PREFIXES,
)
from floorflow.chip_flow.training_scopes import ATOM_SCOPE_PREFIXES


def _batch() -> dict:
    count = 5
    constraints = torch.zeros(2, count, 5)
    constraints[:, :, 3] = torch.tensor([[1, 1, 2, 2, 0]] * 2)
    constraints[:, :2, 2] = 1
    constraints[:, :, 4] = torch.tensor([[1, 0, 2, 0, 0]] * 2)
    boxes = torch.stack(
        [
            torch.stack(
                [
                    torch.arange(count) * 1.2,
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
        "areas": torch.ones(2, count),
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor([[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]] * 2),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 0.5]]] * 2),
        "pins": torch.tensor([[[0.2, 0.4], [0.8, 0.6]]] * 2),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
    }


def _model() -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=24,
        heads=3,
        depth=1,
        head_dim=8,
        typed_layers=1,
    )


def test_shape_head_is_identity_at_init_and_head_gradient_is_reachable() -> None:
    torch.manual_seed(11)
    head = ShapeModeHead(24, modes=4, width=8)
    residuals, mode_logits = head(
        torch.randn(2, 7, 24),
        torch.randn(2, 7, 3),
    )
    assert residuals.shape == (2, 4, 7)
    assert mode_logits.shape == (2, 4)
    assert torch.equal(residuals, torch.zeros_like(residuals))
    residuals.sum().backward()
    assert head.output.weight.grad is not None
    assert head.output.weight.grad.abs().sum() > 0.0


def test_shape_modes_preserve_centers_and_exact_area() -> None:
    areas = torch.tensor([[2.0, 3.0, 5.0]])
    source = torch.tensor([[[0.2, 0.3, -0.4], [0.5, 0.1, 0.2], [-0.3, 0.7, 0.5]]])
    residuals = torch.tensor([[[0.1, -0.2, 0.3]]])
    latents = apply_shape_modes(source, residuals)
    assert latents.shape == (1, 1, 3, 3)
    torch.testing.assert_close(latents[..., :2], source[:, None, :, :2])
    boxes = decode_boxes_batch(latents[0], areas)
    torch.testing.assert_close(boxes[0, :, 2] * boxes[0, :, 3], areas[0])


def test_endpoint_modes_can_move_soft_centers_without_moving_preplaced_blocks() -> None:
    torch.manual_seed(14)
    head = ShapeModeHead(
        24,
        modes=2,
        width=8,
        max_delta=0.2,
        center_scale=0.1,
    )
    with torch.no_grad():
        head.output.weight.normal_(std=0.02)
        head.center_output.weight.normal_(std=0.2)
    prediction = torch.zeros(1, 3, 3)
    constraints = torch.zeros(1, 3, 5)
    constraints[0, 1, 1] = 1

    output = head.forward_with_latents(
        prediction,
        torch.randn(1, 3, 24),
        prediction,
        constraints=constraints,
    )

    assert not torch.allclose(output.mode_latents[0, :, 0, :2], prediction[0, 0, :2])
    torch.testing.assert_close(
        output.mode_latents[0, :, 1, :2],
        prediction[0, 1, :2].expand(2, 2),
    )
    assert output.center_residuals.shape == (1, 2, 3, 2)


def test_shape_modes_preserve_fixed_rho_and_share_mib_rho() -> None:
    prediction = torch.tensor([[[0.0, 0.0, -0.2], [1.0, 0.0, 0.1], [2.0, 0.0, 0.3]]])
    residuals = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    constraints = torch.zeros(1, 3, 5)
    constraints[0, 0, 0] = 1
    constraints[0, 1:, 2] = 1
    modes = apply_shape_modes(
        prediction,
        residuals,
        constraints=constraints,
    )
    torch.testing.assert_close(
        modes[0, :, 0, 2],
        prediction[0, 0, 2].expand(2),
    )
    torch.testing.assert_close(
        modes[0, :, 1, 2],
        modes[0, :, 2, 2],
    )


def test_mib_sharing_never_rewrites_anchored_rho() -> None:
    prediction = torch.tensor(
        [[[0.0, 0.0, -0.2], [1.0, 0.0, 0.3], [2.0, 0.0, 0.0]]]
    )
    residuals = torch.tensor([[[0.1, 0.2, 0.4]]])
    constraints = torch.zeros(1, 3, 5)
    constraints[0, :2, 0] = 1
    constraints[0, :, 2] = 1

    modes = apply_shape_modes(
        prediction,
        residuals,
        constraints=constraints,
    )

    torch.testing.assert_close(
        modes[0, :, :2, 2],
        prediction[0, :2, 2][None],
    )
    torch.testing.assert_close(
        modes[0, :, 2, 2],
        torch.full((1,), 0.05),
    )


def test_shape_modes_can_use_mode_specific_block_support() -> None:
    prediction = torch.zeros(1, 5, 3)
    residuals = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.1, 0.2, 0.3, 0.4, 0.5],
            ]
        ]
    )

    modes = apply_shape_modes(
        prediction,
        residuals,
        top_fractions=(0.2, 1.0),
    )

    assert torch.count_nonzero(modes[0, 0, :, 2]) == 1
    assert torch.count_nonzero(modes[0, 1, :, 2]) == 5


def test_support_budget_expands_to_keep_mib_members_shared() -> None:
    prediction = torch.zeros(1, 5, 3)
    residuals = torch.full((1, 1, 5), 0.2)
    constraints = torch.zeros(1, 5, 5)
    constraints[0, 1:3, 2] = 1

    modes = apply_shape_modes(
        prediction,
        residuals,
        constraints=constraints,
        top_fractions=(0.2,),
    )

    torch.testing.assert_close(
        modes[0, 0, 1:3, 2],
        torch.full((2,), 0.2),
    )


def test_shape_head_is_block_permutation_equivariant() -> None:
    torch.manual_seed(13)
    head = ShapeModeHead(24, modes=3, width=8)
    tokens = torch.randn(1, 6, 24)
    state = torch.randn(1, 6, 3)
    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    expected, expected_logits = head(tokens, state)
    actual, actual_logits = head(
        tokens[:, permutation],
        state[:, permutation],
    )
    torch.testing.assert_close(actual, expected[:, :, permutation])
    torch.testing.assert_close(actual_logits, expected_logits)


def test_shape_mode_features_affect_selection_but_not_rho_residuals() -> None:
    torch.manual_seed(15)
    head = ShapeModeHead(24, modes=3, width=8)
    tokens = torch.randn(1, 5, 24)
    state = torch.randn(1, 5, 3)
    features = torch.randn(1, 3, 8)

    plain_residuals, plain_logits = head(tokens, state)
    feature_residuals, feature_logits = head(
        tokens,
        state,
        mode_features=features,
    )

    torch.testing.assert_close(feature_residuals, plain_residuals)
    assert not torch.allclose(feature_logits, plain_logits)
    torch.nn.functional.cross_entropy(
        feature_logits,
        torch.tensor([1]),
    ).backward()
    assert head.mode_feature_encoder.weight.grad is not None
    assert head.mode_feature_encoder.weight.grad.abs().sum() > 0.0


def test_wta_diversity_and_usage_losses() -> None:
    target = torch.tensor([[[0.0, 0.0, 0.1], [0.2, 0.1, -0.1]]])
    modes = torch.stack(
        [
            target,
            target + 1.0,
            target - 1.0,
        ],
        dim=1,
    )
    loss, winner = winner_take_all_loss(modes, target)
    assert loss.item() == 0.0
    assert winner.tolist() == [0]
    collapsed = mode_diversity_loss(torch.zeros(1, 3, 4), margin=0.1)
    separated = mode_diversity_loss(
        torch.stack(
            [
                torch.zeros(1, 4),
                torch.full((1, 4), 0.2),
                torch.full((1, 4), -0.2),
            ],
            dim=1,
        ),
        margin=0.1,
    )
    assert collapsed > separated
    assert mode_usage_loss(torch.zeros(8, 3)).item() < 1.0e-5
    assert mode_usage_loss(torch.tensor([[0.0, 10.0, 10.0]] * 8)).item() > 1.0
    terms = shape_mode_auxiliary_loss(modes, target)
    assertions = (
        "total",
        "wta",
        "generator",
        "diversity",
        "usage",
        "selection",
        "winner",
    )
    assert all(name in terms for name in assertions)


def test_adapter_preserves_m27_identity_and_checkpoint_contract() -> None:
    torch.manual_seed(17)
    base = _model().eval()
    candidate = _model().eval()
    candidate.load_state_dict(base.state_dict(), strict=False)
    attach_shape_mode_head(candidate, modes=3, width=8)
    batch = _batch()
    base_context = base.prepare_context(batch)
    candidate_context = candidate.prepare_context(batch)
    target = build_hierarchy_targets(batch, candidate_context)
    time = torch.full((2,), 0.5)
    with torch.no_grad():
        expected = base(target, time, base_context).block_state
        actual = candidate(target, time, candidate_context)
    torch.testing.assert_close(actual.block_state, expected)
    mode_latents = shape_mode_latents_from_prediction(
        candidate,
        actual,
        target,
        candidate_context,
    )
    assert mode_latents is not None
    assert torch.equal(
        mode_latents[..., 2],
        actual.block_state[:, None, :, 2].expand_as(
            mode_latents[..., 2]
        ),
    )
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 24,
            "heads": 3,
            "depth": 1,
            "head_dim": 8,
            "typed_layers": 1,
            "shape_modes": 3,
            "shape_mode_width": 8,
            "shape_mode_max_delta": 0.5,
        },
        "model": candidate.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=False)
    assert hasattr(restored, "shape_mode_head")
    assert "shape_modes" in RESUME_CRITICAL_KEYS
    assert "shape_mode_head." in RESUME_NEW_PARAM_PREFIXES


def test_old_shape_checkpoint_accepts_new_selector_parameters() -> None:
    model = _model()
    attach_shape_mode_head(model, modes=3, width=8)
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if key != "shape_mode_head.mode_bias"
        and key != "shape_mode_head.baseline_bias"
        and not key.startswith("shape_mode_head.mode_score.")
        and not key.startswith("shape_mode_head.mode_feature_encoder.")
        and not key.startswith("shape_mode_head.baseline_feature_score.")
    }
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 24,
            "heads": 3,
            "depth": 1,
            "head_dim": 8,
            "typed_layers": 1,
            "shape_modes": 3,
            "shape_mode_width": 8,
        },
        "model": legacy_state,
    }

    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=False)

    assert restored.shape_mode_head.mode_bias[0] == 10.0


def test_old_shape_checkpoint_adds_zero_center_output_when_requested() -> None:
    model = _model()
    attach_shape_mode_head(model, modes=3, width=8, center_scale=0.2)
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("shape_mode_head.center_output.")
    }
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 24,
            "heads": 3,
            "depth": 1,
            "head_dim": 8,
            "typed_layers": 1,
            "shape_modes": 3,
            "shape_mode_width": 8,
            "shape_mode_center_scale": 0.2,
        },
        "model": legacy_state,
    }

    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=False)

    center_output = restored.shape_mode_head.center_output
    assert center_output is not None
    assert torch.count_nonzero(center_output.weight) == 0


def test_build_chip_model_enables_head_only_by_config() -> None:
    base_config = {
        "architecture": "typed_hierarchy",
        "hidden": 24,
        "heads": 3,
        "depth": 1,
        "head_dim": 8,
        "typed_layers": 1,
    }
    assert not hasattr(build_chip_model(base_config), "shape_mode_head")
    opt_in = build_chip_model(
        {
            **base_config,
            "shape_modes": 2,
            "shape_mode_width": 8,
            "shape_mode_center_scale": 0.2,
        }
    )
    assert hasattr(opt_in, "shape_mode_head")
    assert opt_in.shape_mode_head.center_output is not None
    assert "shape_modes" in enabled_branches({"shape_modes": 2})
    assert "shape_mode_head." in ATOM_SCOPE_PREFIXES["area"]


def test_shape_mode_training_loss_reaches_head_parameters(monkeypatch) -> None:
    torch.manual_seed(23)
    model = _model()
    attach_shape_mode_head(model, modes=3, width=8)
    batch = _batch()
    batch["constraints"][:, 0, 1] = 1.0
    batch["constraints"][:, 1, 0] = 1.0
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    prediction = model(target, torch.full((2,), 0.5), context)
    args = SimpleNamespace(
        shape_mode_loss_weight=0.1,
        shape_mode_generator_weight=0.1,
        shape_mode_diversity_weight=0.01,
        shape_mode_usage_weight=0.01,
        shape_mode_selection_weight=0.1,
        shape_mode_diversity_margin=0.05,
        shape_mode_usage_temperature=0.1,
    )
    seen_baselines = []
    original_features = shape_mode_module.shape_mode_geometry_features

    def record_features(mode_latents, baseline, batch, **kwargs):
        seen_baselines.append(baseline.detach().clone())
        return original_features(mode_latents, baseline, batch, **kwargs)

    monkeypatch.setattr(
        shape_mode_module,
        "shape_mode_geometry_features",
        record_features,
    )
    loss, terms = shape_mode_training_loss(
        prediction,
        target,
        args,
        (model, target, context, batch),
    )
    assert torch.isfinite(loss)
    assert terms["shape_mode"].item() > 0.0
    loss.backward()
    assert model.shape_mode_head.output.weight.grad is not None
    assert model.shape_mode_head.output.weight.grad.abs().sum() > 0.0
    assert model.shape_mode_head.mode_score[-1].weight.grad is not None
    assert model.shape_mode_head.mode_feature_encoder.weight.grad is not None
    assert model.shape_mode_head.baseline_bias.grad is not None
    assert branch_health(model)["shape_mode"]["parameter_count"] > 0
    expected_baseline = project_conditioned_latent(
        prediction.block_state,
        batch["target_latent"],
        batch["constraints"],
    )
    torch.testing.assert_close(seen_baselines[0], expected_baseline.detach())
    assert not torch.equal(seen_baselines[0], prediction.block_state.detach())


def test_hierarchy_sampling_exports_identity_shape_candidates() -> None:
    torch.manual_seed(29)
    model = _model()
    attach_shape_mode_head(model, modes=2, width=8)
    model.hierarchy_source_levels = "random"
    batch = _batch()
    batch["source_latent"] = torch.randn(2, 5, 3)
    generated, diagnostics = sample_hierarchy_flow(
        model,
        batch,
        sampling_steps=1,
        generator=torch.Generator().manual_seed(2),
        projector=False,
        projector_iterations=4,
    )
    modes = diagnostics["shape_mode_latents"]
    assert modes.shape == (2, 2, 5, 3)
    torch.testing.assert_close(
        modes[..., 2],
        generated[:, None, :, 2].expand_as(modes[..., 2]),
    )
    assert diagnostics["shape_mode_logits"].shape == (2, 2)
    assert diagnostics["shape_mode_candidate_logits"].shape == (2, 3)
    assert diagnostics["shape_mode_selected"].tolist() == [-1, -1]
    assert diagnostics["shape_mode_selected_candidate"].tolist() == [0, 0]


def test_hierarchy_sampling_applies_the_selected_shape_mode() -> None:
    torch.manual_seed(31)
    model = _model()
    attach_shape_mode_head(model, modes=2, width=8, apply=True)
    with torch.no_grad():
        model.shape_mode_head.output.weight.normal_(0.0, 0.1)
        model.shape_mode_head.mode_bias[1].fill_(10.0)
        model.shape_mode_head.baseline_bias.fill_(-20.0)
    model.hierarchy_source_levels = "random"
    batch = _batch()
    batch["source_latent"] = torch.randn(2, 5, 3)

    generated, diagnostics = sample_hierarchy_flow(
        model,
        batch,
        sampling_steps=1,
        generator=torch.Generator().manual_seed(4),
        projector=False,
        projector_iterations=4,
    )
    modes = diagnostics["shape_mode_latents"]

    assert diagnostics["shape_mode_selected"].tolist() == [1, 1]
    torch.testing.assert_close(generated, modes[:, 1])
