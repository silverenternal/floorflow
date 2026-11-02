#!/usr/bin/env python3
"""Fixed-order CPU overfit comparison: global latent versus slot transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from floorflow.chip_flow.batched import batch_physics_terms
from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.flow_adapter import (
    decode_for_batch,
    encode_target_and_condition,
)
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.slot_model import ChipSlotFlowModel
from floorflow.chip_flow.sparse_slot_flow import ChipSlotTransportFlowModel

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--cases-per-count", type=int, default=8)
    parser.add_argument(
        "--block-counts",
        type=int,
        nargs="+",
        default=[21, 26, 31, 36, 41, 46, 51, 60],
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--endpoint-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_slot_flow_overfit_probe_v1.json",
    )
    return parser.parse_args()


def selected_indices(
    store: PackedCaseStore,
    block_counts: list[int],
    per_count: int,
) -> dict[int, list[int]]:
    return {
        count: store.indices(count)[:per_count].tolist()
        for count in block_counts
    }


def model_for(arm: str, args: argparse.Namespace) -> torch.nn.Module:
    common = {
        "feature_dim": 10,
        "hidden": args.hidden,
        "global_dim": args.hidden,
        "target_layers": 1,
        "condition_layers": 1,
        "decoder_layers": 1,
        "heads": 4,
    }
    if arm == "global":
        return ChipSlotFlowModel(**common)
    transport_common = {
        **common,
        "target_layers": 2,
        "condition_layers": 2,
    }
    return ChipSlotTransportFlowModel(
        **transport_common,
        max_slots=8,
        min_slots=2,
        assignment_top_k=2,
        decoder_top_k=4,
        transport_layers=1,
        residual_slots=arm == "slot_residual",
    )


def losses_for_batch(
    model: torch.nn.Module,
    batch: dict,
    generator: torch.Generator,
    *,
    evaluate: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    target, condition, context = encode_target_and_condition(model, batch)
    source = torch.randn(target.shape, generator=generator)
    time = (
        torch.full((target.shape[0],), 0.5, device=target.device)
        if evaluate
        else torch.rand(target.shape[0], generator=generator)
    )
    time_broadcast = time.reshape(
        -1,
        *([1] * (target.ndim - 1)),
    )
    interpolant = (1.0 - time_broadcast) * source + time_broadcast * target
    velocity = model(interpolant, time.reshape(-1), condition)
    flow = velocity.sub(target - source).pow(2).mean()
    estimate = interpolant + (1.0 - time_broadcast) * velocity
    latent, diagnostics = decode_for_batch(
        model,
        estimate,
        condition,
        context,
        batch,
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
    endpoint = F.smooth_l1_loss(
        latent,
        batch["target_latent"],
        beta=0.1,
    )
    return flow + terms["total"].mean(), endpoint, diagnostics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    store: PackedCaseStore,
    indices: dict[int, list[int]],
    generator: torch.Generator,
) -> dict:
    model.eval()
    total = endpoint_sum = 0.0
    cases = 0
    active_slots = []
    for rows in indices.values():
        batch = store.batch(rows, device="cpu")
        loss, endpoint, diagnostics = losses_for_batch(
            model,
            batch,
            generator,
            evaluate=True,
        )
        total += float(loss) * len(rows)
        endpoint_sum += float(endpoint) * len(rows)
        cases += len(rows)
        if "active_slot_count" in diagnostics:
            active_slots.extend(diagnostics["active_slot_count"].tolist())
    result = {
        "fit_loss": total / cases,
        "endpoint_loss": endpoint_sum / cases,
        "cases": cases,
    }
    if active_slots:
        result["active_slots"] = {
            "min": min(active_slots),
            "max": max(active_slots),
            "mean": sum(active_slots) / len(active_slots),
        }
    return result


def train_arm(
    arm: str,
    store: PackedCaseStore,
    indices: dict[int, list[int]],
    args: argparse.Namespace,
) -> dict:
    torch.manual_seed(args.seed)
    model = model_for(arm, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    eval_generator = torch.Generator().manual_seed(args.seed + 1000)
    initial = evaluate(model, store, indices, eval_generator)
    train_generator = torch.Generator().manual_seed(args.seed + 2000)
    history = []
    counts = sorted(indices)
    for step in range(args.steps):
        model.train()
        count = counts[step % len(counts)]
        batch = store.batch(indices[count], device="cpu")
        loss, endpoint, diagnostics = losses_for_batch(
            model,
            batch,
            train_generator,
            evaluate=False,
        )
        optimizer.zero_grad(set_to_none=True)
        (loss + args.endpoint_weight * endpoint).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step == args.steps - 1:
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "endpoint_loss": float(endpoint.detach()),
                    "active_slots": (
                        diagnostics.get(
                            "active_slot_count",
                            torch.zeros(1),
                        ).tolist()
                    ),
                }
            )
    final = evaluate(model, store, indices, eval_generator)
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial": initial,
        "final": final,
        "relative_loss_reduction": (
            (initial["fit_loss"] - final["fit_loss"])
            / initial["fit_loss"]
        ),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    store = PackedCaseStore(args.cache)
    indices = selected_indices(
        store,
        args.block_counts,
        args.cases_per_count,
    )
    results = {
        arm: train_arm(arm, store, indices, args)
        for arm in ("global", "slot_transport", "slot_residual")
    }
    global_final = results["global"]["final"]["fit_loss"]
    transport_final = results["slot_transport"]["final"]["fit_loss"]
    global_endpoint = results["global"]["final"]["endpoint_loss"]
    transport_endpoint = results["slot_transport"]["final"]["endpoint_loss"]
    residual_final = results["slot_residual"]["final"]["fit_loss"]
    residual_endpoint = results["slot_residual"]["final"]["endpoint_loss"]
    dynamic = results["slot_transport"].get("final", {}).get("active_slots")
    checks = {
        "fit_both_arms": all(
            result["relative_loss_reduction"] >= 0.30
            for result in results.values()
        ),
        "transport_not_worse": transport_final <= global_final * 1.02,
        "endpoint_within_five_percent": (
            transport_endpoint <= global_endpoint * 1.05
        ),
        "residual_beats_absolute_fit": residual_final < transport_final,
        "residual_endpoint_within_five_percent": (
            residual_endpoint <= global_endpoint * 1.05
        ),
        "dynamic_slots_retained": bool(
            dynamic and dynamic["min"] < dynamic["max"]
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "fixed-order 64-case CPU overfit probe; not a quality claim",
        "cache": str(args.cache),
        "steps": args.steps,
        "endpoint_weight": args.endpoint_weight,
        "cases": sum(len(rows) for rows in indices.values()),
        "block_counts": sorted(indices),
        "checks": checks,
        "arms": results,
        "promotion": (
            "slot_residual"
            if all(checks.values())
            else "keep_global_or_sparse"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
