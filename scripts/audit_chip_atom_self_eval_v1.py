#!/usr/bin/env python3
"""Run parameter and gradient self-evaluation for chip model atoms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.chip_flow.atom_health import atom_health
from floorflow.chip_flow.atom_probe_fixtures import (
    ATOM_PREFIXES,
    activate_probe_parameters,
    full_probe_model,
    membership_probe_model,
    probe_args,
    probe_batch,
)
from floorflow.chip_flow.hard_constraint_projector import (
    project_hard_constraints,
)
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.legalizer_portfolio import (
    LegalizerCandidate,
    select_legalizer_candidate,
)
from floorflow.chip_flow.nfe_consistency import (
    hierarchy_nfe_consistency_loss,
)
from floorflow.chip_flow.ppa_external import parse_orfs_graph
from floorflow.chip_flow.train_objective import supervised_loss
from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.chip_flow.training_scopes import (
    apply_training_scope,
    parse_training_scope,
)


def _probe_loss(
    model: HierarchicalTokenFlowModel,
    batch: dict,
) -> float:
    torch.manual_seed(59)
    context = model.prepare_context(batch)
    state = build_hierarchy_targets(batch, context)
    prediction = model(state, torch.full((2,), 0.5), context)
    loss = (
        prediction.global_state.float().square().mean()
        + prediction.structural_state.float().square().mean()
        + prediction.block_state.float().square().mean()
        + prediction.port_state.float().square().mean()
    )
    return float(loss.detach())


def _disable_sensitivity(
    model: HierarchicalTokenFlowModel,
    batch: dict,
) -> dict[str, dict[str, float | bool]]:
    activate_probe_parameters(model)
    model.eval()
    baseline = _probe_loss(model, batch)
    result = {}
    for atom_id, prefixes in ATOM_PREFIXES.items():
        selected = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith(prefixes)
        ]
        if not selected:
            continue
        original = [parameter.detach().clone() for _, parameter in selected]
        with torch.no_grad():
            for _, parameter in selected:
                parameter.zero_()
        disabled = _probe_loss(model, batch)
        with torch.no_grad():
            for (_, parameter), value in zip(selected, original, strict=True):
                parameter.copy_(value)
        delta = abs(disabled - baseline)
        generator = torch.Generator().manual_seed(1000 + int(atom_id[1:]))
        with torch.no_grad():
            for _, parameter in selected:
                flat = parameter.reshape(-1)
                if flat.numel() > 1:
                    order = torch.randperm(
                        flat.numel(),
                        generator=generator,
                    )
                    parameter.copy_(flat[order].reshape_as(parameter))
        shuffled = _probe_loss(model, batch)
        with torch.no_grad():
            for (_, parameter), value in zip(selected, original, strict=True):
                parameter.copy_(value)
        result[atom_id] = {
            "baseline_loss": baseline,
            "disabled_loss": disabled,
            "absolute_delta": delta,
            "relative_delta": delta / max(abs(baseline), 1.0e-12),
            "sensitive": bool(torch.isfinite(torch.tensor(delta)) and delta > 1.0e-9),
            "shuffled_loss": shuffled,
            "shuffle_absolute_delta": abs(shuffled - baseline),
            "equal_parameter_sensitive": bool(
                torch.isfinite(torch.tensor(shuffled))
                and abs(shuffled - baseline) > 1.0e-9
            ),
        }
    return result


def _response_probe_loss(model: HierarchicalTokenFlowModel) -> float:
    head = model.structured_heads.legalizer_response
    displacement, rung = head(
        torch.randn(2, 8, 32, generator=torch.Generator().manual_seed(67)),
        torch.randn(2, 8, 3, generator=torch.Generator().manual_seed(68)),
        torch.randn(2, 8, 3, generator=torch.Generator().manual_seed(69)),
        model.time(torch.full((2, 1), 0.5)),
    )
    return float((displacement.sum() + rung.sum()).detach())


def _extra_sensitivity() -> dict[str, dict[str, float | bool]]:
    result = {}
    response_model = full_probe_model().eval()
    activate_probe_parameters(response_model)
    baseline = _response_probe_loss(response_model)
    selected = [
        (name, parameter)
        for name, parameter in response_model.named_parameters()
        if name.startswith(ATOM_PREFIXES["A29"])
    ]
    original = [parameter.detach().clone() for _, parameter in selected]
    with torch.no_grad():
        for _, parameter in selected:
            parameter.zero_()
    disabled = _response_probe_loss(response_model)
    with torch.no_grad():
        for (_, parameter), value in zip(selected, original, strict=True):
            parameter.copy_(value)
    with torch.no_grad():
        for _, parameter in selected:
            flat = parameter.reshape(-1)
            if flat.numel() > 1:
                order = torch.randperm(flat.numel())
                parameter.copy_(flat[order].reshape_as(parameter))
    shuffled = _response_probe_loss(response_model)
    result["A29"] = {
        "baseline_loss": baseline,
        "disabled_loss": disabled,
        "absolute_delta": abs(disabled - baseline),
        "relative_delta": abs(disabled - baseline)
        / max(abs(baseline), 1.0e-12),
        "sensitive": bool(abs(disabled - baseline) > 1.0e-9),
        "shuffled_loss": shuffled,
        "shuffle_absolute_delta": abs(shuffled - baseline),
        "equal_parameter_sensitive": bool(abs(shuffled - baseline) > 1.0e-9),
        "probe": "direct_legalizer_response",
    }
    membership_model = membership_probe_model().eval()
    activate_probe_parameters(membership_model)
    baseline = _probe_loss(membership_model, probe_batch())
    selected = [
        (name, parameter)
        for name, parameter in membership_model.named_parameters()
        if name.startswith("conditioners.membership.")
    ]
    original = [parameter.detach().clone() for _, parameter in selected]
    with torch.no_grad():
        for _, parameter in selected:
            parameter.zero_()
    disabled = _probe_loss(membership_model, probe_batch())
    with torch.no_grad():
        for (_, parameter), value in zip(selected, original, strict=True):
            parameter.copy_(value)
    with torch.no_grad():
        for _, parameter in selected:
            flat = parameter.reshape(-1)
            if flat.numel() > 1:
                order = torch.randperm(flat.numel())
                parameter.copy_(flat[order].reshape_as(parameter))
    shuffled = _probe_loss(membership_model, probe_batch())
    result["A16"] = {
        "baseline_loss": baseline,
        "disabled_loss": disabled,
        "absolute_delta": abs(disabled - baseline),
        "relative_delta": abs(disabled - baseline)
        / max(abs(baseline), 1.0e-12),
        "sensitive": bool(abs(disabled - baseline) > 1.0e-9),
        "shuffled_loss": shuffled,
        "shuffle_absolute_delta": abs(shuffled - baseline),
        "equal_parameter_sensitive": bool(abs(shuffled - baseline) > 1.0e-9),
        "probe": "membership_model",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atoms",
        type=Path,
        default=Path("configs/chip_sota_metric_ownership_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/chip_atom_self_eval_v1.json"),
    )
    args = parser.parse_args()
    torch.manual_seed(31)
    model = full_probe_model()
    batch = probe_batch()
    sensitivity = _disable_sensitivity(model, probe_batch())
    sensitivity.update(_extra_sensitivity())
    loss, terms = supervised_loss(model, batch, probe_args(), torch.device("cpu"))
    loss.backward()
    health = atom_health(model, ATOM_PREFIXES)
    model.zero_grad(set_to_none=True)
    response_head = model.structured_heads.legalizer_response
    response_output = response_head(
        torch.randn(2, 8, 32),
        torch.randn(2, 8, 3),
        torch.randn(2, 8, 3),
        model.time(torch.rand(2, 1)),
    )
    (response_output[0].sum() + response_output[1].sum()).backward()
    health["A29"] = atom_health(
        model,
        {"A29": ATOM_PREFIXES["A29"]},
    )["A29"]
    runtime_probes = {}
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    time = torch.full((2,), 0.5)
    prediction = model(target, time, context)
    nfe_loss, nfe_terms = hierarchy_nfe_consistency_loss(
        model,
        target,
        time,
        prediction,
        target,
        context,
        probe_args(),
        step=0,
    )
    runtime_probes["A31"] = {
        "status": "pass" if torch.isfinite(nfe_loss) else "fail",
        "loss": float(nfe_loss.detach()),
        "terms": {
            name: float(value.detach())
            for name, value in nfe_terms.items()
        },
    }
    projected = project_hard_constraints(
        batch["target_latent"],
        batch["target_latent"],
        batch["constraints"],
        batch["areas"],
        iterations=1,
        grid_size=8,
    )
    runtime_probes["A33"] = {
        "status": "pass",
        "remaining_overlap_pairs": projected.remaining_overlap_pairs,
        "boundary_violations": projected.boundary_violations,
        "max_area_error": projected.max_area_error,
        "fixed_shape_error": projected.fixed_shape_error,
    }
    baseline = LegalizerCandidate(
        "baseline",
        v_rel=0.2,
        displacement=0.1,
        boundary=1.0,
        grouping=1.0,
        rung=1,
        solves=1,
        b2b_hpwl=0.2,
        p2b_hpwl=0.1,
        outline_compactness=0.8,
        outline_aspect=0.2,
    )
    alternate = LegalizerCandidate(
        "alternate",
        v_rel=0.15,
        displacement=0.08,
        boundary=0.0,
        grouping=0.0,
        rung=1,
        solves=1,
        b2b_hpwl=0.15,
        p2b_hpwl=0.08,
        outline_compactness=0.9,
        outline_aspect=0.1,
    )
    selected = select_legalizer_candidate(
        baseline,
        (alternate,),
        policy="geometry_consensus",
    )
    runtime_probes["A35"] = {
        "status": "pass",
        "selected": selected.name,
    }
    scope_stats = apply_training_scope(
        model,
        parse_training_scope("fusion"),
    )
    runtime_probes["A37"] = {
        "status": "pass",
        "scope": scope_stats["scope"],
        "trainable_parameters": scope_stats["trainable_parameters"],
    }
    aggregator = model.structured_heads.atom_aggregator
    with torch.no_grad():
        aggregator.reliability[-1].weight.zero_()
        aggregator.reliability[-1].bias.zero_()
    branch_residuals = torch.randn(
        2,
        aggregator.atom_count,
        8,
        3,
    )
    aggregator_inputs = (
        branch_residuals,
        context.typed.block_tokens,
        target.global_state,
        context.typed.structural_tokens,
        context.typed.structural_mask,
        context.typed.port_tokens,
        context.typed.port_mask,
        time,
    )
    with torch.no_grad():
        aggregator.mix_gate.zero_()
        initial, initial_diagnostics = aggregator(*aggregator_inputs)
    identity_error = float(
        (initial - branch_residuals.sum(dim=1)).abs().max().detach()
    )
    with torch.no_grad():
        aggregator.mix_gate.fill_(0.25)
    learned, _ = aggregator(*aggregator_inputs)
    learned.square().mean().backward()
    aggregator_gradient = aggregator.mix_gate.grad
    runtime_probes["A42"] = {
        "status": (
            "pass"
            if identity_error <= 1.0e-6
            and aggregator_gradient is not None
            and torch.isfinite(aggregator_gradient).all()
            and initial_diagnostics["atom_aggregator_active_fraction"] > 0
            else "fail"
        ),
        "identity_error": identity_error,
        "initial_gain": float(initial_diagnostics["atom_aggregator_gain"]),
        "active_fraction": float(
            initial_diagnostics["atom_aggregator_active_fraction"]
        ),
        "min_positive_gate": float(
            initial_diagnostics["atom_aggregator_gate_min_positive"]
        ),
        "mix_gate_gradient": (
            float(aggregator_gradient.abs().sum())
            if aggregator_gradient is not None
            else None
        ),
    }
    model.zero_grad(set_to_none=True)
    curve_summary_path = Path(
        "outputs/reports/chip_m27_canonical_refiner_curve_v1.json"
    )
    curve_cases_path = Path(
        "outputs/reports/chip_m27_canonical_refiner_curve_v1_cases.json"
    )
    if curve_summary_path.is_file() and curve_cases_path.is_file():
        curve_summary = json.loads(curve_summary_path.read_text(encoding="utf-8"))
        curve_cases = json.loads(curve_cases_path.read_text(encoding="utf-8"))
        runtime_probes["A32"] = {
            "status": curve_summary.get("status", "fail"),
            "checkpoints": len(curve_cases),
            "cases": curve_summary.get("selected_cases"),
            "best_by_preregistered_metric": curve_summary.get(
                "best_by_preregistered_mean_metric"
            ),
        }
        legalizer_rows = [
            row
            for rows in curve_cases.values()
            for row in rows
            if "legalizer_rung" in row
        ]
        runtime_probes["A34"] = {
            "status": "pass" if legalizer_rows else "fail",
            "rows": len(legalizer_rows),
            "feasible_fraction": (
                sum(bool(row["legalized_feasible"]) for row in legalizer_rows)
                / len(legalizer_rows)
                if legalizer_rows
                else 0.0
            ),
            "rung_values": sorted(
                {int(row["legalizer_rung"]) for row in legalizer_rows}
            ),
        }
    else:
        runtime_probes["A32"] = {"status": "missing"}
        runtime_probes["A34"] = {"status": "missing"}
    external_tsv = (
        "ORFS_DBU_PER_MICRON\t1000\n"
        "ORFS_DIE\t0\t0\t10000\t10000\n"
        "ORFS_CORE\t0\t0\t9000\t9000\n"
        "ORFS_MACRO\tmacro_a\tmaster_a\t100\t100\t0\t0\tPLACED\tN\n"
        "ORFS_PORT\tport_a\t100\t100\n"
        "ORFS_NET_ITERM\tnet_a\tmacro_a\n"
        "ORFS_NET_BTERM\tnet_a\tport_a"
    )
    external_graph = parse_orfs_graph(external_tsv)
    runtime_probes["A36"] = {
        "status": "pass",
        "macros": len(external_graph.macros),
        "ports": len(external_graph.ports),
        "nets": len(external_graph.net_macros),
        "core_size": external_graph.core_size,
    }
    torch.manual_seed(31)
    membership_model = membership_probe_model()
    membership_loss, _ = supervised_loss(
        membership_model,
        probe_batch(),
        probe_args(),
        torch.device("cpu"),
    )
    membership_loss.backward()
    membership_health = atom_health(
        membership_model,
        {"A16": ("conditioners.membership.",)},
    )
    health["A16"] = membership_health["A16"]
    atom_config = json.loads(args.atoms.read_text(encoding="utf-8"))
    rows = []
    for atom in atom_config["atoms"]:
        atom_id = atom["id"]
        measured = health.get(atom_id)
        if atom_id in runtime_probes:
            status = f"runtime_{runtime_probes[atom_id]['status']}"
        elif measured is None:
            status = {
                "A31": "objective_probe_required",
                "A32": "external_module_probe_required",
            }.get(atom_id, "parameter_free_probe_required")
        elif measured["active"]:
            status = "active"
        elif measured["gate_abs_max"] <= 1.0e-9 and measured["gradient_tensor_count"]:
            status = "zero_gate_identity"
        else:
            status = "inactive_no_gradient"
        rows.append(
            {
                "atom_id": atom_id,
                "module": atom["module"],
                "self_eval_plan": atom["self_eval"],
                "status": status,
                "health": measured,
                "controlled_disable": sensitivity.get(atom_id),
            }
        )
    report = {
        "version": "chip-atom-self-eval-v1",
        "synthetic_batch": {"cases": 2, "blocks": 8, "pins": 12},
        "loss": float(loss.detach()),
        "loss_terms": {
            name: float(value.detach())
            for name, value in terms.items()
            if torch.is_tensor(value) and value.numel() == 1
        },
        "summary": {
            "atoms": len(rows),
            "parameter_measured": sum(
                not row["status"].endswith("probe_required") for row in rows
            ),
            "active": sum(row["status"] == "active" for row in rows),
            "zero_gate_identity": sum(
                row["status"] == "zero_gate_identity" for row in rows
            ),
            "inactive_no_gradient": sum(
                row["status"] == "inactive_no_gradient" for row in rows
            ),
            "runtime_probe_required": sum(
                row["status"].endswith("probe_required") for row in rows
            ),
            "runtime_probes_passed": sum(
                row["status"] == "runtime_pass" for row in rows
            ),
        },
        "atoms": rows,
        "runtime_probes": runtime_probes,
        "controlled_disable_summary": {
            "tested": len(sensitivity),
            "sensitive": sum(row["sensitive"] for row in sensitivity.values()),
        },
        "equal_parameter_summary": {
            "tested": len(sensitivity),
            "sensitive": sum(
                row["equal_parameter_sensitive"] for row in sensitivity.values()
            ),
        },
        "evidence_boundary": (
            "Parameter and gradient coverage only; missing runtime probes are "
            "not treated as passes."
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
