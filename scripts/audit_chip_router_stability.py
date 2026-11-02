#!/usr/bin/env python3
"""CPU gradient and routing-stability audit for the sparse chip hierarchy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.flow_adapter import graph_inputs
from floorflow.chip_flow.sparse_flow_model import ChipSparseHierarchyFlowModel

ROOT = Path(__file__).resolve().parents[1]


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
        default=ROOT / "outputs/reports/chip_router_stability_cpu_v1.json",
    )
    parser.add_argument("--cases-per-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def gradient_norm(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().float().norm())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    store = PackedCaseStore(args.cache)
    model = ChipSparseHierarchyFlowModel(
        feature_dim=10,
        hidden=64,
        global_dim=64,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        max_slots=8,
        min_slots=2,
        assignment_top_k=2,
        assignment_temperature=0.7,
        decoder_top_k=4,
    )
    active_counts = []
    budget_means = []
    router_load_max = []
    dead_slot_fractions = []
    assignment_entropy = []
    gradient_rows = []
    count_rows = []
    for count in sorted(store.by_block_count):
        indices = store.indices(count)[: args.cases_per_count]
        if len(indices) == 0:
            continue
        batch = store.batch(indices, device="cpu")
        context = model.encode_condition_context(
            batch["features"],
            **graph_inputs(batch),
        )
        decoded, decode_diagnostics = model.decode_with_diagnostics(
            context.global_state,
            context,
            batch["features"],
        )
        regularization = model.sparse_regularization(
            context,
            decode_diagnostics,
        )
        loss = (
            decoded.float().square().mean()
            + regularization["budget"]
            + regularization["route_balance_total"]
        )
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {
            "slot_queries": gradient_norm(
                model.condition_hierarchy.router.slot_queries.grad
            ),
            "budget_head": gradient_norm(
                model.condition_hierarchy.router.budget_head.weight.grad
            ),
            "route_q_proj": gradient_norm(model.decoder.route.q_proj.weight.grad),
            "route_k_proj": gradient_norm(model.decoder.route.k_proj.weight.grad),
        }
        gradients["all_finite"] = all(
            value is not None and np.isfinite(value)
            for value in gradients.values()
        )
        gradients["all_nonzero"] = all(
            value is not None and value > 0.0
            for key, value in gradients.items()
            if key not in {"all_finite", "all_nonzero"}
        )
        active = decode_diagnostics["active_slot_count"]
        budgets = decode_diagnostics["router_budget_per_case"]
        load = decode_diagnostics["router_load"]
        active_counts.extend(active.tolist())
        budget_means.extend(budgets.tolist())
        router_load_max.extend(load.amax(dim=1).tolist())
        dead_slot_fractions.append(
            float(regularization["dead_slot_fraction"])
        )
        assignment_entropy.append(
            float(
                context.hierarchy.diagnostics["assignment_entropy"]
                .float()
                .detach()
            )
        )
        gradient_rows.append(gradients)
        count_rows.append(
            {
                "blocks": count,
                "cases": len(indices),
                "active_slots": active.tolist(),
                "budget_mean": budgets.tolist(),
            }
        )
    checks = {
        "dynamic_active_slots": len(set(active_counts)) > 1,
        "dynamic_router_budgets": len(set(budget_means)) > 1,
        "gradients_finite": all(row["all_finite"] for row in gradient_rows),
        "active_route_gradients_nonzero": all(
            row["all_nonzero"] for row in gradient_rows
        ),
        "no_all_dead_batch": max(dead_slot_fractions, default=1.0) < 1.0,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "random-initialized CPU gradient audit; no quality claim",
        "checks": checks,
        "cases": sum(row["cases"] for row in count_rows),
        "active_slots": {
            "min": min(active_counts),
            "max": max(active_counts),
            "mean": float(np.mean(active_counts)),
            "unique": sorted(set(active_counts)),
        },
        "router_budget": {
            "min": min(budget_means),
            "max": max(budget_means),
            "mean": float(np.mean(budget_means)),
            "unique": sorted(set(budget_means)),
        },
        "router_load_max": {
            "min": min(router_load_max),
            "max": max(router_load_max),
            "mean": float(np.mean(router_load_max)),
        },
        "dead_slot_fraction": {
            "min": min(dead_slot_fractions),
            "max": max(dead_slot_fractions),
            "mean": float(np.mean(dead_slot_fractions)),
        },
        "assignment_entropy": {
            "min": min(assignment_entropy),
            "max": max(assignment_entropy),
            "mean": float(np.mean(assignment_entropy)),
        },
        "rows": count_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
