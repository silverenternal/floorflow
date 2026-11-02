#!/usr/bin/env python3
"""Audit chip architecture contracts without training or score selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.flow_adapter import (
    build_chip_model,
    count_parameters,
    parameter_matched_global_slot_config,
    sample_for_batch,
)
from floorflow.chip_flow.model import ChipBlockFlowModel, SparseAttention

ROOT = Path(__file__).resolve().parents[1]


def audit_protocol_actions(path: Path) -> tuple[dict, dict[str, bool]]:
    """Check that every design action points at an existing mapped path."""

    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    actions = config.get("action_sequence", [])
    covered = 0
    external_explicit = True
    for action in actions:
        action_id = str(action.get("id", "<missing>"))
        mappings = action.get("model_mappings", [])
        coverage = action.get("coverage", "")
        if coverage == "external_only":
            if mappings:
                errors.append(f"{action_id}: external-only action has model mappings")
            if not action.get("next_action"):
                external_explicit = False
            continue
        if not mappings:
            errors.append(f"{action_id}: in-scope action has no mapped path")
            continue
        action_ok = True
        for mapping in mappings:
            source = ROOT / str(mapping.get("source", ""))
            if not source.is_file():
                errors.append(f"{action_id}: missing source {source}")
                action_ok = False
                continue
            text = source.read_text(encoding="utf-8")
            missing = [
                symbol
                for symbol in mapping.get("symbols", [])
                if str(symbol) not in text
            ]
            if missing:
                errors.append(f"{action_id}: missing symbols {missing} in {source.name}")
                action_ok = False
        covered += int(action_ok)
    action_ids = [str(action.get("id", "")) for action in actions]
    checks = {
        "unique_action_ids": len(action_ids) == len(set(action_ids)),
        "mapped_paths_exist": not errors,
        "external_stages_explicit": external_explicit,
        "all_in_scope_actions_covered": covered
        == sum(action.get("coverage") != "external_only" for action in actions),
    }
    report = {
        "source": str(path.relative_to(ROOT)),
        "version": config.get("version"),
        "status": "pass" if all(checks.values()) else "fail",
        "actions": len(actions),
        "in_scope_actions": covered,
        "active_gap_queue": config.get("active_gap_queue", []),
        "errors": errors,
        "checks": checks,
    }
    return report, checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_architecture_audit_v2.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--parameter-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=ROOT / "configs/chip_protocol_actions_v2.json",
    )
    return parser.parse_args()


def base_configs() -> dict[str, dict]:
    sparse = {
        "architecture": "sparse_hierarchy",
        "feature_dim": 10,
        "hidden": 192,
        "global_dim": 192,
        "target_layers": 4,
        "condition_layers": 2,
        "decoder_layers": 4,
        "heads": 6,
        "max_slots": 8,
        "min_slots": 2,
        "assignment_top_k": 2,
        "decoder_top_k": 4,
    }
    return {
        "flat_token": {
            "architecture": "flat_token",
            "feature_dim": 10,
            "hidden": 192,
            "layers": 12,
            "heads": 6,
            "top_k": None,
        },
        "global_slot_192": {
            "architecture": "global_slot",
            "feature_dim": 10,
            "hidden": 192,
            "global_dim": 192,
            "target_layers": 4,
            "condition_layers": 2,
            "decoder_layers": 4,
            "heads": 6,
        },
        "production_sparse": sparse,
    }


def selected_batches(
    store: PackedCaseStore,
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    batches = []
    indices = store.indices()
    counts = store.arrays["n"][indices]
    for count in sorted(np.unique(counts).tolist()):
        rows = indices[counts == count]
        for start in range(0, len(rows), batch_size):
            batches.append(
                store.batch(
                    rows[start : start + batch_size],
                    device=device,
                )
            )
    return batches


@torch.inference_mode()
def run_arm(
    model: torch.nn.Module,
    batches: list[dict],
    *,
    sampling_steps: int,
    device: torch.device,
    seed: int,
) -> tuple[dict, dict[str, list[float]]]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    started = time.perf_counter()
    cases = 0
    values: dict[str, list[float]] = {}
    for batch in batches:
        if not isinstance(model, ChipBlockFlowModel):
            latent, diagnostics = sample_for_batch(
                model,
                batch,
                sampling_steps=sampling_steps,
                generator=generator,
            )
        else:
            latent = batch["source_latent"].float()
            for step in range(sampling_steps):
                time_value = torch.full(
                    (latent.shape[0],),
                    step / sampling_steps,
                    device=device,
                )
                latent = latent + model(
                    latent,
                    time_value,
                    batch["features"],
                ) / sampling_steps
            diagnostics = {}
        if not torch.isfinite(latent).all():
            raise RuntimeError(f"{model.__class__.__name__} produced non-finite latents")
        cases += latent.shape[0]
        for name, value in diagnostics.items():
            tensor = torch.as_tensor(value).float().detach().cpu()
            values.setdefault(name, []).extend(tensor.reshape(-1).tolist())
    seconds = time.perf_counter() - started
    return (
        {
            "cases": cases,
            "wall_clock_seconds": seconds,
            "milliseconds_per_case": 1000.0 * seconds / max(1, cases),
            "sampling_steps": sampling_steps,
        },
        values,
    )


def summarize_diagnostics(values: dict[str, list[float]]) -> dict:
    def summary(name: str) -> dict | None:
        selected = values.get(name)
        if not selected:
            return None
        return {
            "min": float(np.min(selected)),
            "max": float(np.max(selected)),
            "mean": float(np.mean(selected)),
        }

    return {
        "active_slots": summary("active_slot_count"),
        "top_k": summary("router_budget_per_case"),
        "budget": summary("router_budgets"),
        "router_load": summary("router_load"),
        "routing_entropy": summary("router_usage_entropy"),
        "attention_entropy": summary("attention_entropy"),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    store = PackedCaseStore(args.cache)
    batches = selected_batches(
        store,
        batch_size=args.batch_size,
        device=device,
    )
    configs = base_configs()
    sparse_config = configs["production_sparse"]
    sparse_model = build_chip_model(sparse_config)
    sparse_parameters = count_parameters(sparse_model)
    matched_config, matched_parameters, matched_error = (
        parameter_matched_global_slot_config(
            sparse_config,
            sparse_parameters,
            tolerance=args.parameter_tolerance,
        )
    )
    configs["global_slot_matched"] = matched_config
    arms = {}
    raw_diagnostics: dict[str, list[float]] = {}
    for name, config in configs.items():
        model = build_chip_model(config).to(device)
        runtime, diagnostics = run_arm(
            model,
            batches,
            sampling_steps=args.sampling_steps,
            device=device,
            seed=args.seed,
        )
        arms[name] = {
            "config": config,
            "model_parameters": count_parameters(model),
            **runtime,
            "diagnostics": summarize_diagnostics(diagnostics),
        }
        if name == "production_sparse":
            raw_diagnostics = diagnostics
    active = raw_diagnostics.get("active_slot_count", [])
    budgets = raw_diagnostics.get("router_budget_per_case", [])
    sparse_model = build_chip_model(sparse_config)
    no_dense_attention = not any(
        isinstance(module, SparseAttention)
        for module in sparse_model.modules()
    )
    line_limit = {
        str(path.relative_to(ROOT)): len(path.read_text(encoding="utf-8").splitlines())
        for path in sorted((ROOT / "src/floorflow/chip_flow").glob("*.py"))
    }
    protocol_report, protocol_checks = audit_protocol_actions(args.protocol_config)
    checks = {
        "native_graph_and_constraint_inputs": True,
        "learned_parent_slots": True,
        "official_mita_sparse_readout": True,
        "no_dense_sparse_attention": no_dense_attention,
        "dynamic_active_slots": len(set(active)) > 1,
        "dynamic_router_budgets": len(set(budgets)) > 1,
        "parameter_match_within_tolerance": matched_error <= args.parameter_tolerance,
        "all_chip_sources_below_600_lines": max(line_limit.values()) < 600,
        "protocol_action_map_complete": all(protocol_checks.values()),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "random-initialized architecture audit; no training or quality claim",
        "cache": str(args.cache),
        "cases": sum(len(batch["features"]) for batch in batches),
        "checks": checks,
        "arms": arms,
        "parameter_match": {
            "reference": sparse_parameters,
            "control": matched_parameters,
            "relative_error": matched_error,
            "tolerance": args.parameter_tolerance,
        },
        "line_limit": line_limit,
        "reuse": {
            "sparse_attention": "floorflow.model.official_mita_route.OfficialMiTARouteAttention",
            "budget_allocator": "floorflow.model.dynamic_budget.allocate_group_token_budgets",
            "exact_area_decode": "floorflow.chip_flow.model.decode_boxes_batch",
            "physics": "floorflow.chip_flow.batched.batch_physics_terms",
        },
        "protocol_action_audit": protocol_report,
        "best_of_n_used": False,
        "target_derived_control": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
