from argparse import Namespace

import pytest
import torch

from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.objective_schedules import scheduled_proxy_weight
from floorflow.chip_flow.slot_model import ChipSlotFlowModel
from floorflow.chip_flow.train_objective import (
    annealed_weight,
    delayed_ramp_weight,
    supervised_loss,
    top_fraction_mean,
)


def _batch() -> dict:
    batch, count = 2, 4
    constraints = torch.zeros(batch, count, 5)
    constraints[:, 0, 0] = 1.0
    constraints[:, :2, 2] = 1.0
    constraints[:, :, 4] = torch.tensor([1.0, 2.0, 0.0, 0.0])
    return {
        "areas": torch.ones(batch, count),
        "boxes": torch.tensor(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            ]
        ).expand(batch, -1, -1),
        "constraints": constraints,
        "features": torch.zeros(batch, count, 10),
        "target_latent": torch.randn(batch, count, 3),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 1.0], [1.0, 2.0, 0.5], [0.0, 0.0, 0.0]],
            ]
        ),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 1.0]]]),
        "pins": torch.tensor([[[0.5, 0.5]], [[0.5, 0.5]]]),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([1, 1]),
        "n_pins": torch.tensor([1, 1]),
    }


def _args() -> Namespace:
    return Namespace(
        time_std=1.0,
        flow_huber_beta=0.1,
        path_steps=2,
        path_weight=0.2,
        physics_weight=1.0,
        overlap_weight_scale=10.0,
        decode_weight=1.0,
        rho_weight=0.0,
        clean_weight=0.1,
        sparsity_weight=0.01,
        route_balance_weight=0.01,
    )


def test_path_consistency_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(37)
    model = ChipSlotFlowModel(
        feature_dim=19,
        hidden=16,
        global_dim=16,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        graph_conditioning=True,
    )
    batch = _batch()
    loss, parts = supervised_loss(model, batch, _args(), torch.device("cpu"))
    assert torch.isfinite(loss)
    assert float(parts["path"].detach()) > 0.0
    loss.backward()
    assert model.target_encoder.graph_layers[0].gate.grad is not None
    assert model.decoder.global_film is None


def test_normalized_loss_balance_is_an_opt_in_integrated_objective() -> None:
    torch.manual_seed(39)
    model = ChipSlotFlowModel(
        feature_dim=19,
        hidden=16,
        global_dim=16,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        graph_conditioning=True,
    )
    args = Namespace(
        **_args().__dict__,
        loss_balance_mode="normalized",
        loss_balance_floor=1.0e-6,
    )
    loss, terms = supervised_loss(model, _batch(), args, torch.device("cpu"))
    assert torch.isfinite(loss)
    for name in ("flow", "physics", "decode", "clean"):
        assert f"loss_balance_{name}" in terms
    loss.backward()
    assert model.target_encoder.graph_layers[0].gate.grad is not None


def test_objective_weight_schedules_hit_exact_endpoints() -> None:
    assert annealed_weight(1.0, 0.2, 0, 100, "linear") == 1.0
    assert annealed_weight(1.0, 0.2, 50, 100, "linear") == pytest.approx(0.6)
    assert annealed_weight(1.0, 0.2, 100, 100, "linear") == 0.2
    assert annealed_weight(1.0, 0.2, 50, 100, "cosine") == pytest.approx(0.6)
    assert annealed_weight(1.0, 0.2, 50, 0, "linear") == 1.0


def test_delayed_ramp_holds_proxy_off_then_reaches_target() -> None:
    assert delayed_ramp_weight(0.05, 1000, 2000, 999) == 0.0
    assert delayed_ramp_weight(0.05, 1000, 2000, 1000) == 0.0
    assert delayed_ramp_weight(0.05, 1000, 2000, 2000) == pytest.approx(0.025)
    assert delayed_ramp_weight(0.05, 1000, 2000, 3000) == pytest.approx(0.05)


def test_scheduled_proxy_uses_final_weight_after_delay() -> None:
    args = Namespace(
        grouping_proxy_weight=0.0,
        grouping_proxy_final_weight=0.01,
        grouping_proxy_delay_steps=25,
        grouping_proxy_anneal_steps=100,
    )

    assert scheduled_proxy_weight(args, "grouping_proxy", 24) == 0.0
    assert scheduled_proxy_weight(
        args,
        "grouping_proxy",
        75,
    ) == pytest.approx(0.005)
    assert scheduled_proxy_weight(
        args,
        "grouping_proxy",
        125,
    ) == pytest.approx(0.01)


def test_top_fraction_mean_tracks_worst_cases() -> None:
    values = torch.tensor([1.0, 4.0, 2.0, 3.0])
    assert top_fraction_mean(values, 0.25) == pytest.approx(4.0)
    assert top_fraction_mean(values, 0.5) == pytest.approx(3.5)
    assert top_fraction_mean(values, 1.0) == pytest.approx(2.5)


def test_identity_hierarchy_decoder_skips_constant_clean_terms() -> None:
    torch.manual_seed(43)
    model = HierarchicalTokenFlowModel(hidden=16, heads=2, depth=1)
    batch = _batch()
    batch["source_latent"] = torch.randn(2, 4, 3)
    args = _args()
    args.path_steps = 0
    args.hierarchy_global_weight = 0.1
    args.hierarchy_structural_weight = 0.1
    args.hierarchy_port_weight = 0.1
    args.hierarchy_source_mode = "random"
    args.hierarchy_source_noise = 0.0
    args.hierarchy_source_levels = "random"
    _, terms = supervised_loss(model, batch, args, torch.device("cpu"))
    assert float(terms["clean"]) == 0.0
    assert float(terms["clean_physics"]) == 0.0
    values = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        top_fraction_mean(values, 0.0)


def test_tail_weight_changes_objective_and_reports_gradient() -> None:
    torch.manual_seed(41)
    model = ChipSlotFlowModel(
        feature_dim=19,
        hidden=16,
        global_dim=16,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        graph_conditioning=True,
    )
    base_args = _args()
    tail_args = {**vars(base_args)}
    tail_args.update(
        {
            "physics_tail_weight": 0.5,
            "physics_tail_fraction": 0.5,
            "physics_tail_delay_steps": 0,
            "physics_tail_anneal_steps": 0,
        }
    )
    torch.manual_seed(43)
    base_loss, _ = supervised_loss(
        model,
        _batch(),
        base_args,
        torch.device("cpu"),
    )
    torch.manual_seed(43)
    tail_loss, terms = supervised_loss(
        model,
        _batch(),
        Namespace(**tail_args),
        torch.device("cpu"),
    )
    assert tail_loss > base_loss
    assert terms["physics_tail"] > 0.0
    tail_loss.backward()
    assert model.target_encoder.graph_layers[0].gate.grad is not None


def test_structured_head_contract_suppresses_duplicate_decode_loss() -> None:
    torch.manual_seed(47)
    model = HierarchicalTokenFlowModel(hidden=16, heads=2, depth=1)
    batch = _batch()
    batch["source_latent"] = torch.randn(2, 4, 3)
    args = _args()
    args.path_steps = 0
    args.hierarchy_global_weight = 0.1
    args.hierarchy_structural_weight = 0.1
    args.hierarchy_port_weight = 0.1
    args.hierarchy_source_mode = "random"
    args.hierarchy_source_noise = 0.0
    args.hierarchy_source_levels = "random"
    args.structured_head_contract = True
    args.hierarchy_forced_head_weight = 0.0
    loss, terms = supervised_loss(model, batch, args, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert float(terms["decode"].detach()) == 0.0
    assert "flow_block_center" in terms
    assert "flow_block_rho" in terms
    assert "flow_port_connected" in terms
    assert "flow_port_unconnected" in terms


def test_endpoint_compatibility_term_is_integrated_when_enabled() -> None:
    torch.manual_seed(53)
    model = HierarchicalTokenFlowModel(hidden=16, heads=2, depth=1)
    batch = _batch()
    batch["source_latent"] = torch.randn(2, 4, 3)
    args = _args()
    args.path_steps = 0
    args.hierarchy_global_weight = 0.1
    args.hierarchy_structural_weight = 0.1
    args.hierarchy_port_weight = 0.1
    args.hierarchy_source_mode = "random"
    args.hierarchy_source_noise = 0.0
    args.hierarchy_source_levels = "random"
    args.endpoint_compatibility_weight = 0.1
    args.endpoint_compatibility_iterations = 1
    args.endpoint_compatibility_delay_steps = 0
    args.endpoint_compatibility_anneal_steps = 0
    loss, terms = supervised_loss(model, batch, args, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert float(terms["endpoint_compatibility_current_weight"]) == pytest.approx(0.1)
    assert "endpoint_compatibility" in terms
    assert "endpoint_projection_displacement" in terms


def test_full_structured_head_objective_trains_end_to_end_on_cpu() -> None:
    torch.manual_seed(59)
    model = HierarchicalTokenFlowModel(
        hidden=16,
        heads=2,
        depth=1,
        structured_heads=True,
    )
    batch = _batch()
    batch["source_latent"] = torch.randn(2, 4, 3)
    batch["legalizer_displacement"] = torch.randn(2, 4, 2)
    batch["legalizer_rung"] = torch.tensor([1, 3])
    args = _args()
    args.path_steps = 0
    args.hierarchy_global_weight = 0.1
    args.hierarchy_structural_weight = 0.1
    args.hierarchy_port_weight = 0.1
    args.hierarchy_source_mode = "random"
    args.hierarchy_source_noise = 0.0
    args.hierarchy_source_levels = "random"
    args.structured_head_contract = True
    args.hierarchy_forced_head_weight = 0.0
    args.endpoint_compatibility_weight = 0.05
    args.endpoint_compatibility_iterations = 1
    args.endpoint_compatibility_delay_steps = 0
    args.endpoint_compatibility_anneal_steps = 0
    args.endpoint_compatibility_displacement_weight = 1.0
    args.endpoint_compatibility_gain_weight = 0.25
    args.endpoint_refiner_energy_weight = 1.0
    args.legalizer_response_weight = 0.1
    args.legalizer_displacement_weight = 1.0
    args.legalizer_rung_weight = 0.1
    args.legalizer_response_delay_steps = 0
    args.legalizer_response_anneal_steps = 0
    loss, terms = supervised_loss(model, batch, args, torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.structured_heads is not None
    assert model.structured_heads.factorized_block is not None
    gradient = model.structured_heads.factorized_block.center_head[-1].weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0
    assert terms["flow_block_center"] >= 0
    assert terms["flow_port_connected"] >= 0
    assert "legalizer_response" in terms
    assert terms["legalizer_response_displacement"] >= 0
    assert terms["atom_aggregator_active_fraction"] > 0
    assert torch.isfinite(terms["atom_aggregator_gate_min_positive"])


def test_training_projector_is_differentiable_through_main_endpoint() -> None:
    torch.manual_seed(61)
    model = HierarchicalTokenFlowModel(hidden=16, heads=2, depth=1)
    batch = _batch()
    batch["source_latent"] = torch.randn(2, 4, 3)
    args = _args()
    args.path_steps = 0
    args.hierarchy_global_weight = 0.1
    args.hierarchy_structural_weight = 0.1
    args.hierarchy_port_weight = 0.1
    args.hierarchy_source_mode = "random"
    args.hierarchy_source_noise = 0.0
    args.hierarchy_source_levels = "random"
    args.training_projector = True
    args.training_projector_iterations = 2
    args.training_projector_strength = 1.0
    args.training_projector_displacement_weight = 0.1
    args.projector_sparse_pairs = True
    args.projector_strategy = "conflict_aware"
    args.projector_grid_size = 16
    loss, terms = supervised_loss(model, batch, args, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert "training_projector_displacement" in terms
    loss.backward()
    assert model.backbone.final.linear.weight.grad is not None
    assert model.backbone.final.linear.weight.grad.abs().sum() > 0
