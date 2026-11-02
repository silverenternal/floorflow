#!/usr/bin/env python3
"""Intervene on the sampled global state to measure decoder dependence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.chip_flow.batched import batch_physics_terms
from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.flow_adapter import (
    encode_target_and_condition,
    graph_adapter_inputs,
)
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_physics import group_penalty
from scripts.audit_chip_global_representation import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "outputs/chip_final/screen_20k/"
        "graph_velocity_seed19_20k_final/checkpoints/final.pt",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/dev",
    )
    parser.add_argument("--cases-per-count", type=int, default=5)
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_global_causality_probe_v1.json",
    )
    return parser.parse_args()


@torch.no_grad()
def sample_state(
    model,
    condition: torch.Tensor,
    *,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    state = torch.randn(
        condition.shape[0],
        model.global_dim,
        device=condition.device,
        dtype=condition.dtype,
        generator=generator,
    )
    for step in range(steps):
        time = torch.full(
            (state.shape[0],),
            step / steps,
            device=state.device,
            dtype=state.dtype,
        )
        state = state + model(state, time, condition) / steps
    return state


@torch.no_grad()
def geometry_metrics(
    model,
    state: torch.Tensor,
    condition: torch.Tensor,
    batch: dict,
    features: torch.Tensor,
    adjacency: torch.Tensor,
    key_mask: torch.Tensor,
) -> dict[str, float]:
    latent = model.decode(
        state,
        condition,
        features,
        adjacency,
        key_mask,
    )
    latent = project_conditioned_latent(
        latent,
        batch["target_latent"],
        batch["constraints"],
    )
    boxes = decode_boxes_batch(latent, batch["areas"])
    terms = batch_physics_terms(
        boxes,
        batch["areas"],
        batch["constraints"],
        batch["b2b"],
        batch["p2b"],
        batch["pins"],
        batch["n_b2b"],
        batch["n_p2b"],
        batch["n_pins"],
    )
    group = torch.stack(
        [
            group_penalty(
                boxes[index],
                batch["constraints"][index, :, 3],
                batch["areas"][index],
            )
            for index in range(latent.shape[0])
        ]
    )
    return {
        "latent_mse": float((latent - batch["target_latent"]).square().mean()),
        "overlap": float(terms["overlap"].mean()),
        "group": float(group.mean()),
        "boundary": float(terms["boundary"].mean()),
        "total": float(terms["total"].mean() + group.mean()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device)
    store = PackedCaseStore(args.cache)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    totals: dict[str, dict[str, float]] = {}
    cases = 0
    with torch.no_grad():
        for count in sorted(store.by_block_count):
            indices = store.indices(count)[: args.cases_per_count]
            batch = store.batch(indices, device=device)
            features, adjacency, key_mask = graph_adapter_inputs(model, batch)
            target, condition, _ = encode_target_and_condition(model, batch)
            original = sample_state(
                model,
                condition,
                steps=args.sampling_steps,
                generator=generator,
            )
            variants = {
                "original": original,
                "zero": torch.zeros_like(original),
                "shuffled": original.flip(0),
                "condition": condition,
                "target_oracle": target,
            }
            for name, state in variants.items():
                metrics = geometry_metrics(
                    model,
                    state,
                    condition,
                    batch,
                    features,
                    adjacency,
                    key_mask,
                )
                for metric, value in metrics.items():
                    totals.setdefault(name, {})
                    totals[name][metric] = (
                        totals[name].get(metric, 0.0) + value * len(indices)
                    )
            cases += len(indices)
    averages = {
        name: {metric: value / cases for metric, value in metrics.items()}
        for name, metrics in totals.items()
    }
    baseline = averages["original"]["total"]
    effects = {
        name: {
            "total_relative_delta": (
                averages[name]["total"] - baseline
            )
            / baseline,
            "latent_mse_relative_delta": (
                averages[name]["latent_mse"]
                - averages["original"]["latent_mse"]
            )
            / averages["original"]["latent_mse"],
        }
        for name in averages
        if name != "original"
    }
    checks = {
        "case_specific_global_state_used": (
            effects["shuffled"]["total_relative_delta"] > 0.05
        ),
        "zero_global_state_changes_geometry": (
            effects["zero"]["total_relative_delta"] > 0.05
        ),
        "oracle_global_state_improves_geometry": (
            effects["target_oracle"]["total_relative_delta"] < -0.05
        ),
    }
    report = {
        "status": (
            "global_state_causal"
            if checks["case_specific_global_state_used"]
            else "decoder_ignores_global_state"
        ),
        "scope": "dev intervention probe at NFE4; no official quality claim",
        "checkpoint": str(args.checkpoint),
        "cases": cases,
        "sampling_steps": args.sampling_steps,
        "averages": averages,
        "effects_vs_original": effects,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
