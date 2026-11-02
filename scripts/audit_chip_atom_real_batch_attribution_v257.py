#!/usr/bin/env python3
"""CPU-only real-batch attribution for the v257 chip atom contracts."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from floorflow.chip_flow.atom_health import atom_health
from floorflow.chip_flow.atom_probe_fixtures import (
    ATOM_PREFIXES,
    activate_probe_parameters,
    probe_args,
)
from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.flow_adapter import build_chip_model, count_parameters
from floorflow.chip_flow.hierarchical_flow_runtime import (
    hierarchical_training_state,
    interpolate_hierarchy,
)
from floorflow.chip_flow.hierarchy_adapters import ATOM_BRANCH_NAMES
from floorflow.chip_flow.objective_schedules import scheduled_proxy_weight
from floorflow.chip_flow.source_mixing import source_shelf_probability_at_step
from floorflow.chip_flow.train_objective import supervised_loss
from floorflow.chip_flow.training_runtime import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
STRATA = ("21_40", "41_60", "61_80", "81_100", "101_120")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_architecture_mainline_v4.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=(
            ROOT
            / "outputs/cache/chip_floorset_200k_train_12k_dev_range_v1/train"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_atom_real_batch_attribution_v257.json",
    )
    parser.add_argument("--cases-per-stratum", type=int, default=4)
    parser.add_argument("--step", type=int, default=100)
    return parser.parse_args()


def _training_args(config: dict[str, Any], step: int) -> Namespace:
    args = probe_args()
    for name, value in config["objective"].items():
        setattr(args, name, value)
    args.step = int(step)
    return args


def _stratum_count(
    store: PackedCaseStore,
    stratum: str,
    cases: int,
) -> int | None:
    lower, upper = (int(value) for value in stratum.split("_"))
    available = [
        count
        for count, indices in store.by_block_count.items()
        if lower <= count <= upper and len(indices) >= cases
    ]
    return min(available) if available else None


def _schedule_checks(
    args: Namespace,
    terms: dict[str, torch.Tensor],
    step: int,
) -> dict[str, dict[str, float | bool]]:
    checks = {}
    for name in ("official_cost", "grouping_proxy", "routing_proxy"):
        expected = scheduled_proxy_weight(args, name, step)
        actual = float(terms.get(f"{name}_current_weight", 0.0))
        checks[name] = {
            "expected": expected,
            "actual": actual,
            "aligned": abs(expected - actual) < 1.0e-6,
        }
    return checks


def _branch_diagnostics(
    model: torch.nn.Module,
    batch: dict,
    args: Namespace,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        context = model.prepare_context(batch)
        source, target = hierarchical_training_state(
            batch,
            context,
            source_mode=args.hierarchy_source_mode,
            source_noise=args.hierarchy_source_noise,
            source_levels=args.hierarchy_source_levels,
            source_shelf_probability=source_shelf_probability_at_step(
                args,
                args.step,
            ),
        )
        state = interpolate_hierarchy(
            source,
            target,
            torch.full(
                (batch["areas"].shape[0],),
                0.5,
                device=batch["areas"].device,
            ),
        )
        prediction = model(state, torch.full_like(
            torch.empty(batch["areas"].shape[0]),
            0.5,
        ), context)
    diagnostics = prediction.atom_diagnostics
    residual = diagnostics["atom_aggregator_residual_rms"]
    reliability = diagnostics["atom_aggregator_reliability"]
    return {
        "branch_names": list(ATOM_BRANCH_NAMES),
        "residual_rms": residual.mean(dim=(0, 2)).tolist(),
        "reliability_mean": reliability.mean(dim=(0, 2)).tolist(),
        "gain": float(diagnostics["atom_aggregator_gain"]),
        "active_fraction": float(
            diagnostics["atom_aggregator_active_fraction"]
        ),
    }


def main() -> None:
    cli = parse_args()
    config = json.loads(cli.config.read_text(encoding="utf-8"))
    model_config = {
        **config["architecture"],
        "architecture": "typed_hierarchy",
        **config["branches"],
    }
    model = build_chip_model(model_config)
    activate_probe_parameters(model)
    model.train()
    args = _training_args(config, cli.step)
    store = PackedCaseStore(cli.cache)
    rows = []
    for stratum in STRATA:
        count = _stratum_count(store, stratum, cli.cases_per_stratum)
        if count is None:
            rows.append({"stratum": stratum, "status": "unavailable"})
            continue
        indices = store.indices(count)[: cli.cases_per_stratum]
        batch = store.batch(indices, device="cpu")
        model.train()
        model.zero_grad(set_to_none=True)
        loss, terms = supervised_loss(
            model,
            batch,
            args,
            torch.device("cpu"),
            step=cli.step,
        )
        loss.backward()
        health = atom_health(model, ATOM_PREFIXES)
        scalar_terms = {
            name: float(value.detach())
            for name, value in terms.items()
            if torch.is_tensor(value) and value.numel() == 1
        }
        finite_gradients = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        rows.append(
            {
                "stratum": stratum,
                "status": "pass" if bool(torch.isfinite(loss)) else "fail",
                "block_count": count,
                "cases": len(indices),
                "loss": float(loss.detach()),
                "finite_gradients": finite_gradients,
                "atom_health": health,
                "branches": _branch_diagnostics(model, batch, args),
                "terms": scalar_terms,
                "schedule_checks": _schedule_checks(args, terms, cli.step),
            }
        )
    passed = all(
        row["status"] == "pass"
        and row["finite_gradients"]
        and all(value["finite"] for value in row["atom_health"].values())
        and all(
            value["aligned"] for value in row["schedule_checks"].values()
        )
        for row in rows
        if row["status"] != "unavailable"
    )
    report = {
        "version": "chip-atom-real-batch-attribution-v257",
        "status": "pass_cpu_contract_only" if passed else "fail",
        "device": "cpu",
        "config": str(cli.config.resolve()),
        "cache": str(cli.cache.resolve()),
        "step": cli.step,
        "parameters": count_parameters(model),
        "probe_setup": {
            "zero_gated_parameters_activated": True,
            "seeded": 47,
            "identity_contract": "covered by atom unit tests and atom self-eval",
        },
        "strata": rows,
        "evidence_boundary": (
            "Seeded zero-gate activation probe on frozen real batches. "
            "Gradient, gate, residual and finite-value audit only; no "
            "legalized quality or SOTA claim."
        ),
    }
    write_json_atomic(cli.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "strata": len(rows),
        "parameters": report["parameters"],
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
