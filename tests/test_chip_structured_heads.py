from __future__ import annotations

from types import SimpleNamespace

import torch

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.hierarchy_adapters import apply_hierarchy_input_adapters
from floorflow.chip_flow.model_config import model_config
from floorflow.chip_flow.structured_heads import (
    DualSemanticPortHead,
    FactorizedBlockHead,
    MembershipHierarchyCoupler,
)
from floorflow.chip_flow.train_cli import parse_args


def _batch() -> dict:
    count = 6
    constraints = torch.zeros(2, count, 5)
    constraints[..., 3] = torch.tensor([1, 1, 2, 2, 3, 3]).expand(2, -1)
    constraints[:, :2, 4] = 1
    return {
        "areas": torch.ones(2, count),
        "boxes": torch.stack(
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
        ),
        "constraints": constraints,
        "features": torch.zeros(2, count, 19),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]],
                [[0.0, 2.0, 1.0], [2.0, 4.0, 0.5]],
            ]
        ),
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.5]],
                [[0.0, 2.0, 1.0], [1.0, 4.0, 0.5]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.2, 0.4], [0.8, 0.6], [0.5, 0.5]],
                [[0.1, 0.5], [0.7, 0.9], [0.4, 0.3]],
            ]
        ),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([3, 3]),
    }


def _model(*, structured: bool) -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=1,
        typed_layers=1,
        head_dim=32,
        structured_heads=structured,
        projector_strategy="conflict_aware" if structured else "legacy",
        projector_sparse_pairs=True,
        projector_grid_size=8,
    )


def test_structured_heads_are_zero_initialized_and_checkpointed() -> None:
    torch.manual_seed(7)
    base = _model(structured=False)
    structured = _model(structured=True)
    structured.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    base_context = base.prepare_context(batch)
    structured_context = structured.prepare_context(batch)
    target = build_hierarchy_targets(batch, base_context)
    time = torch.full((2,), 0.5)
    expected = base(target, time, base_context)
    actual = structured(target, time, structured_context)
    torch.testing.assert_close(actual.block_state, expected.block_state)
    torch.testing.assert_close(actual.port_state, expected.port_state)
    assert actual.legalizer_displacement is not None
    assert actual.legalizer_rung_logits is not None
    assert torch.equal(
        actual.legalizer_displacement,
        torch.zeros_like(actual.legalizer_displacement),
    )
    assert torch.equal(
        actual.legalizer_rung_logits,
        torch.zeros_like(actual.legalizer_rung_logits),
    )

    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 1,
            "head_dim": 32,
            "typed_layers": 1,
            "structured_heads": True,
            "structured_legalizer_response": True,
            "projector_strategy": "conflict_aware",
            "projector_sparse_pairs": True,
            "projector_grid_size": 8,
        },
        "model": structured.state_dict(),
        "ema": structured.state_dict(),
    }
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=True)
    assert isinstance(restored, HierarchicalTokenFlowModel)
    assert restored.structured_heads is not None
    assert restored.projector_strategy == "conflict_aware"
    assert restored.projector_grid_size == 8


def test_old_structured_checkpoint_adds_atom_aggregator() -> None:
    model = _model(structured=True)
    legacy = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("structured_heads.atom_aggregator.")
    }
    checkpoint = {
        "config": {
            "architecture": "typed_hierarchy",
            "hidden": 64,
            "heads": 4,
            "depth": 1,
            "head_dim": 32,
            "typed_layers": 1,
            "structured_heads": True,
            "structured_legalizer_response": True,
        },
        "model": legacy,
    }

    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=False)

    aggregator = restored.structured_heads.atom_aggregator
    assert aggregator.mix_gate.item() == 0.0
    assert aggregator.atom_count == 7


def test_full_model_routes_gradient_through_all_structured_heads() -> None:
    torch.manual_seed(9)
    model = _model(structured=True)
    suite = model.structured_heads
    assert suite is not None
    assert suite.hierarchy_coupling is not None
    assert suite.factorized_block is not None
    assert suite.dual_semantic_ports is not None
    suite.hierarchy_coupling.global_gate.data.fill_(1.0)
    suite.hierarchy_coupling.structure_gate.data.fill_(1.0)
    torch.nn.init.normal_(
        suite.hierarchy_coupling.global_reduce[-1].weight,
        std=0.02,
    )
    torch.nn.init.normal_(
        suite.hierarchy_coupling.structure_reduce[-1].weight,
        std=0.02,
    )
    torch.nn.init.normal_(suite.factorized_block.center_head[-1].weight, std=0.02)
    torch.nn.init.normal_(suite.factorized_block.rho_head[-1].weight, std=0.02)
    torch.nn.init.normal_(
        suite.dual_semantic_ports.connected_head[-1].weight,
        std=0.02,
    )
    torch.nn.init.normal_(
        suite.dual_semantic_ports.unconnected_head[-1].weight,
        std=0.02,
    )
    batch = _batch()
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    prediction = model(target, torch.full((2,), 0.5), context)
    assert prediction.atom_diagnostics is not None
    assert "atom_aggregator_active_fraction" in prediction.atom_diagnostics
    loss = prediction.block_state.square().mean()
    loss = loss + prediction.port_state.square().mean()
    loss.backward()
    assert suite.hierarchy_coupling.global_reduce[-1].weight.grad is not None
    assert suite.factorized_block.center_head[-1].weight.grad is not None
    assert suite.dual_semantic_ports.connected_head[-1].weight.grad is not None


def test_port_consensus_consumes_refined_dual_semantic_ports() -> None:
    class PortRefiner(torch.nn.Module):
        def forward(
            self,
            prediction,
            _state,
            _tokens,
            _connected,
            _mask,
            _time,
        ):
            return prediction + prediction.new_tensor([1.0, 2.0])

    class PortRecorder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen = None

        def forward(
            self,
            _tokens,
            _state,
            prediction,
            _incidence,
            _mask,
            _time,
            *,
            pin_positions=None,
            pin_mask=None,
        ):
            del pin_positions, pin_mask
            self.seen = prediction.clone()
            return torch.zeros_like(_state), {}

    structured = SimpleNamespace(
        hierarchy_coupling=None,
        dual_semantic_ports=PortRefiner(),
    )
    recorder = PortRecorder()
    prediction = torch.zeros(1, 2, 2)
    _latent, _output, refined, _diagnostics = apply_hierarchy_input_adapters(
        latent_residual=torch.zeros(1, 3, 3),
        block_state=torch.zeros(1, 3, 3),
        block_tokens=torch.zeros(1, 3, 8),
        structural_tokens=torch.zeros(1, 9, 8),
        structural_mask=torch.zeros(1, 9, dtype=torch.bool),
        port_state=torch.zeros(1, 2, 2),
        port_prediction=prediction,
        port_tokens=torch.zeros(1, 2, 8),
        port_block_incidence=torch.ones(1, 3, 2, dtype=torch.bool),
        port_mask=torch.ones(1, 2, dtype=torch.bool),
        global_prediction=torch.zeros(1, 4),
        structural_prediction=torch.zeros(1, 9, 3),
        time=torch.tensor([0.5]),
        time_token=torch.zeros(1, 8),
        context=SimpleNamespace(
            pin_positions=torch.zeros(1, 2, 2),
            pin_mask=torch.ones(1, 2, dtype=torch.bool),
        ),
        order_staging=None,
        coarse_to_fine=None,
        port_consensus=recorder,
        port_consensus_output=False,
        spatial_occupancy=None,
        complexity_slot=None,
        structured_heads=structured,
    )

    expected = prediction + prediction.new_tensor([1.0, 2.0])
    torch.testing.assert_close(recorder.seen, expected)
    torch.testing.assert_close(refined, expected)


def test_same_call_membership_coupler_uses_predicted_l0_l1() -> None:
    torch.manual_seed(11)
    module = MembershipHierarchyCoupler(32, bottleneck=8)
    module.global_gate.data.fill_(1.0)
    module.structure_gate.data.fill_(1.0)
    torch.nn.init.normal_(module.global_reduce[-1].weight, std=0.02)
    torch.nn.init.normal_(module.structure_reduce[-1].weight, std=0.02)
    block_tokens = torch.randn(1, 4, 32)
    block_state = torch.randn(1, 4, 3)
    membership = torch.zeros(1, 4, 9, dtype=torch.bool)
    membership[0, :2, 0] = True
    membership[0, 2:, 1] = True
    mask = torch.zeros(1, 9, dtype=torch.bool)
    mask[0, :2] = True
    structural_tokens = torch.randn(1, 9, 32)
    structural = torch.randn(1, 9, 3)
    global_prediction = torch.randn(1, 4)
    time = torch.tensor([0.5])
    base = module(
        block_tokens,
        block_state,
        membership,
        mask,
        global_prediction,
        structural,
        structural_tokens,
        time,
    )
    changed_global = module(
        block_tokens,
        block_state,
        membership,
        mask,
        global_prediction + 2.0,
        structural,
        structural_tokens,
        time,
    )
    changed_structure = module(
        block_tokens,
        block_state,
        membership,
        mask,
        global_prediction,
        structural + 2.0,
        structural_tokens,
        time,
    )
    assert not torch.equal(base, changed_global)
    assert not torch.equal(base, changed_structure)
    (base.square().mean() + changed_structure.square().mean()).backward()
    assert module.global_reduce[-1].weight.grad is not None
    assert module.structure_reduce[-1].weight.grad is not None


def test_membership_coupler_zero_output_does_not_deadlock_head_gradients() -> None:
    module = MembershipHierarchyCoupler(16, bottleneck=8)
    block_tokens = torch.randn(1, 3, 16)
    block_state = torch.randn(1, 3, 3)
    membership = torch.zeros(1, 3, 9, dtype=torch.bool)
    membership[0, :, 0] = True
    mask = torch.zeros(1, 9, dtype=torch.bool)
    mask[0, 0] = True
    output = module(
        block_tokens,
        block_state,
        membership,
        mask,
        torch.randn(1, 4),
        torch.randn(1, 9, 3),
        torch.randn(1, 9, 16),
        torch.tensor([0.5]),
    )
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert module.global_reduce[-1].weight.grad is not None
    assert module.global_reduce[-1].weight.grad.abs().sum() > 0.0
    assert module.structure_reduce[-1].weight.grad is not None
    assert module.structure_reduce[-1].weight.grad.abs().sum() > 0.0


def test_membership_coupler_is_block_permutation_equivariant() -> None:
    torch.manual_seed(13)
    module = MembershipHierarchyCoupler(32, bottleneck=8)
    module.global_gate.data.fill_(1.0)
    module.structure_gate.data.fill_(1.0)
    torch.nn.init.normal_(module.global_reduce[-1].weight, std=0.02)
    torch.nn.init.normal_(module.structure_reduce[-1].weight, std=0.02)
    tokens = torch.randn(1, 6, 32)
    state = torch.randn(1, 6, 3)
    membership = torch.zeros(1, 6, 9, dtype=torch.bool)
    membership[0, :3, 0] = True
    membership[0, 3:, 1] = True
    mask = torch.tensor([[True, True] + [False] * 7])
    structural = torch.randn(1, 9, 3)
    structural_tokens = torch.randn(1, 9, 32)
    global_prediction = torch.randn(1, 4)
    time = torch.tensor([0.4])
    original = module(
        tokens,
        state,
        membership,
        mask,
        global_prediction,
        structural,
        structural_tokens,
        time,
    )
    permutation = torch.tensor([2, 5, 0, 4, 1, 3])
    reordered = module(
        tokens[:, permutation],
        state[:, permutation],
        membership[:, permutation],
        mask,
        global_prediction,
        structural,
        structural_tokens,
        time,
    )
    torch.testing.assert_close(original[:, permutation], reordered)


def test_factorized_block_head_separates_center_and_rho() -> None:
    torch.manual_seed(17)
    module = FactorizedBlockHead(32, bottleneck=8)
    block_tokens = torch.randn(2, 6, 32)
    state = torch.randn(2, 6, 3)
    base = torch.randn(2, 6, 3)
    time_token = torch.randn(2, 32)
    torch.nn.init.normal_(module.center_head[-1].weight, std=0.02)
    center_only = module(base, block_tokens, state, time_token)
    assert not torch.equal(center_only[..., :2], torch.zeros_like(center_only[..., :2]))
    assert torch.equal(center_only[..., 2], torch.zeros_like(center_only[..., 2]))
    module.center_head[-1].weight.data.zero_()
    torch.nn.init.normal_(module.rho_head[-1].weight, std=0.02)
    rho_only = module(base, block_tokens, state, time_token)
    assert torch.equal(rho_only[..., :2], torch.zeros_like(rho_only[..., :2]))
    assert not torch.equal(rho_only[..., 2], torch.zeros_like(rho_only[..., 2]))


def test_factorized_shared_trunk_routes_both_heads_and_bounds_rho() -> None:
    torch.manual_seed(18)
    module = FactorizedBlockHead(
        32,
        bottleneck=8,
        shared_trunk=True,
        rho_trust_radius=0.02,
    )
    module.center_head[-1].bias.data.fill_(0.1)
    module.rho_head[-1].bias.data.fill_(0.5)
    output = module(
        torch.randn(2, 6, 3),
        torch.randn(2, 6, 32),
        torch.randn(2, 6, 3),
        torch.randn(2, 32),
    )
    assert module.shared is not None
    assert output[..., 2].abs().max() <= 0.02
    (output[..., :2].square().mean() + output[..., 2].square().mean()).backward()
    assert module.shared[0].weight.grad is not None
    assert torch.isfinite(module.shared[0].weight.grad).all()


def test_factorized_head_preserves_locked_block_degrees_of_freedom() -> None:
    module = FactorizedBlockHead(32, bottleneck=8)
    module.center_head[-1].bias.data.fill_(0.2)
    module.rho_head[-1].bias.data.fill_(0.3)
    constraints = torch.zeros(1, 3, 5)
    constraints[0, 0, 1] = 1.0
    constraints[0, 1, 0] = 1.0
    output = module(
        torch.zeros(1, 3, 3),
        torch.randn(1, 3, 32),
        torch.zeros(1, 3, 3),
        torch.randn(1, 32),
        constraints=constraints,
    )

    torch.testing.assert_close(output[0, 0], torch.zeros(3))
    assert not torch.equal(output[0, 1, :2], torch.zeros(2))
    torch.testing.assert_close(output[0, 1, 2], torch.tensor(0.0))
    assert not torch.equal(output[0, 2], torch.zeros(3))


def test_dual_port_head_uses_connected_and_unconnected_paths() -> None:
    module = DualSemanticPortHead(32, bottleneck=8)
    module.connected_head[-1].bias.data.copy_(torch.tensor([1.0, 2.0]))
    module.unconnected_head[-1].bias.data.copy_(torch.tensor([3.0, 4.0]))
    base = torch.zeros(1, 4, 2)
    state = torch.zeros(1, 4, 2)
    tokens = torch.zeros(1, 4, 32)
    connected = torch.tensor([[True, False, True, False]])
    mask = torch.tensor([[True, True, True, False]])
    output = module(
        base,
        state,
        tokens,
        connected,
        mask,
        torch.zeros(1, 32),
    )
    torch.testing.assert_close(output[0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(output[0, 1], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(output[0, 2], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(output[0, 3], torch.zeros(2))


def test_typed_hierarchy_config_records_effective_x0_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_chip_slot",
            "--architecture",
            "typed_hierarchy",
            "--out-dir",
            "outputs/smoke/head_config",
            "--train-indices",
            "configs/chip_train_screen_5k_v1.json",
        ],
    )
    args = parse_args()
    config = model_config(args)
    assert config["output_mode"] == "x0"


def test_grouping_proxy_final_weight_is_available_from_training_cli(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_chip_slot",
            "--architecture",
            "typed_hierarchy",
            "--out-dir",
            "outputs/smoke/head_config",
            "--train-indices",
            "configs/chip_train_screen_5k_v1.json",
            "--grouping-proxy-final-weight",
            "0.01",
        ],
    )
    args = parse_args()
    assert args.grouping_proxy_final_weight == 0.01
