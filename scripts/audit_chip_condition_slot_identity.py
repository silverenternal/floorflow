#!/usr/bin/env python3
"""CPU audit of condition-derived slot health and target predictability."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.flow_adapter import graph_inputs
from floorflow.chip_flow.sparse_slot_flow import ChipSlotTransportFlowModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "outputs/chip_production_v2/screen_5k/slot_transport_seed19/checkpoints/final.pt",
    )
    parser.add_argument(
        "--block-counts",
        type=int,
        nargs="+",
        default=[21, 26, 31, 36, 41, 46, 51, 60],
    )
    parser.add_argument("--cases-per-count", type=int, default=8)
    parser.add_argument("--train-cases-per-count", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_condition_slot_identity_probe_v2.json",
    )
    return parser.parse_args()


def load_model(path: Path) -> ChipSlotTransportFlowModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = ChipSlotTransportFlowModel(
        feature_dim=int(config.get("feature_dim", 10)),
        hidden=int(config["hidden"]),
        global_dim=int(config.get("global_dim", config["hidden"])),
        target_layers=int(config.get("target_layers", 4)),
        condition_layers=int(config.get("condition_layers", 2)),
        decoder_layers=int(config.get("decoder_layers", 4)),
        heads=int(config["heads"]),
        max_slots=int(config.get("max_slots", 8)),
        min_slots=int(config.get("min_slots", 2)),
        assignment_top_k=int(config.get("assignment_top_k", 2)),
        decoder_top_k=int(config.get("decoder_top_k", 4)),
        transport_layers=int(config.get("transport_layers", 2)),
        residual_slots=False,
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def ridge_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_x = train_x.mean(dim=0, keepdim=True)
    std_x = train_x.std(dim=0, correction=0, keepdim=True).clamp_min(1.0e-6)
    mean_y = train_y.mean(dim=0, keepdim=True)
    std_y = train_y.std(dim=0, correction=0, keepdim=True).clamp_min(1.0e-6)
    x_train = (train_x - mean_x) / std_x
    x_test = (test_x - mean_x) / std_x
    y_train = (train_y - mean_y) / std_y
    gram = x_train.T @ x_train
    gram = gram + alpha * torch.eye(
        gram.shape[0],
        device=gram.device,
        dtype=gram.dtype,
    )
    weights = torch.linalg.solve(gram, x_train.T @ y_train)
    prediction = (x_test @ weights) * std_y + mean_y
    zero_prediction = mean_y.expand(test_x.shape[0], train_y.shape[1]).clone()
    return prediction, zero_prediction


def normalized_entropy(probabilities: torch.Tensor) -> float:
    probabilities = probabilities[probabilities > 0].float()
    if probabilities.numel() <= 1:
        return 0.0
    entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy / math.log(probabilities.numel()))


def grouped_eta_squared(
    targets: torch.Tensor,
    groups: torch.Tensor,
    group_count: int,
) -> float:
    target_mean = targets.mean(dim=0, keepdim=True)
    total = (targets - target_mean).square().sum()
    between = torch.zeros((), dtype=targets.dtype)
    for group in range(group_count):
        selected = groups == group
        if selected.any():
            between = between + selected.sum() * (
                targets[selected].mean(dim=0) - target_mean[0]
            ).square().sum()
    return float(between / total.clamp_min(1.0e-12))


def finite_routing(
    model: ChipSlotTransportFlowModel,
    batch: dict,
) -> dict[str, bool]:
    graph = graph_inputs(batch)
    outputs = {}
    variants = {
        "original": (batch["features"], graph),
        "empty_constraints": (
            batch["features"].clone(),
            {**graph, "constraints": torch.zeros_like(graph["constraints"])},
        ),
        "sparse_graph": (
            batch["features"].clone(),
            {
                **graph,
                "n_b2b": (graph["n_b2b"] // 2).clamp_min(0),
                "n_p2b": (graph["n_p2b"] // 2).clamp_min(0),
            },
        ),
    }
    for name, (features, variant_graph) in variants.items():
        context = model.encode_condition_context(features, **variant_graph)
        outputs[name] = bool(
            torch.isfinite(context.hierarchy.slot_tokens).all()
            and torch.isfinite(context.hierarchy.global_state).all()
        )
    return outputs


@torch.no_grad()
def main() -> None:
    args = parse_args()
    store = PackedCaseStore(args.cache)
    model = load_model(args.checkpoint)
    slot_tokens = []
    slot_ids = []
    active_counts = []
    dead_slots = []
    usage = torch.zeros(model.max_slots)
    target_latents = []
    raw_features = []
    slot_contexts = []
    primary_slots = []
    stress_case = None
    for count in args.block_counts:
        indices = store.indices(count)[: args.cases_per_count]
        batch = store.batch(indices, device="cpu")
        if stress_case is None:
            stress_case = batch
        context = model.encode_condition_context(
            batch["features"],
            **graph_inputs(batch),
        )
        hierarchy = context.hierarchy
        active_counts.extend(hierarchy.active_slot_mask.sum(dim=1).tolist())
        dead_slots.append(
            float(hierarchy.diagnostics["dead_slot_fraction"])
        )
        current_usage = hierarchy.active_slot_mask.float().mean(dim=0)
        usage += current_usage * len(indices)
        for slot in range(model.max_slots):
            selected = hierarchy.active_slot_mask[:, slot]
            if selected.any():
                slot_tokens.append(hierarchy.slot_tokens[selected, slot])
                slot_ids.append(
                    torch.full(
                        (int(selected.sum()),),
                        slot,
                        dtype=torch.long,
                    )
                )
        target_latents.append(
            batch["target_latent"].float().reshape(-1, 3)
        )
        raw_features.append(
            batch["features"].float().reshape(-1, batch["features"].shape[-1])
        )
        slot_contexts.append(
            torch.bmm(
                hierarchy.token_assignments,
                hierarchy.slot_tokens,
            ).reshape(-1, model.global_dim)
        )
        primary_slots.append(
            hierarchy.token_assignments.argmax(dim=-1).reshape(-1)
        )

    tokens = torch.cat(slot_tokens)
    ids = torch.cat(slot_ids)
    cosine = torch.nn.functional.normalize(tokens, dim=-1) @ torch.nn.functional.normalize(
        tokens,
        dim=-1,
    ).T
    same = ids[:, None].eq(ids[None, :])
    diagonal = torch.eye(len(ids), dtype=torch.bool)
    off_diagonal = ~diagonal
    same_cosine = cosine[same & off_diagonal]
    different_cosine = cosine[~same]
    if not same_cosine.numel() or not different_cosine.numel():
        raise RuntimeError("slot identity audit requires both same- and cross-slot pairs")
    identity_margin = float(same_cosine.mean() - different_cosine.mean())

    targets = torch.cat(target_latents)
    features = torch.cat(raw_features)
    contexts = torch.cat(slot_contexts)
    primary = torch.cat(primary_slots)
    absolute_eta_squared = grouped_eta_squared(
        targets,
        primary,
        model.max_slots,
    )

    train_masks = []
    test_masks = []
    for count in args.block_counts:
        rows = store.indices(count)[: args.cases_per_count]
        mask = torch.zeros(len(rows), count, dtype=torch.bool)
        mask[: args.train_cases_per_count] = True
        train_masks.append(mask.flatten())
        test_masks.append(~mask.flatten())
    train_mask = torch.cat(train_masks)
    test_mask = torch.cat(test_masks)
    raw_prediction, _ = ridge_predict(
        features[train_mask],
        targets[train_mask],
        features[test_mask],
        args.ridge,
    )
    slot_prediction, zero_prediction = ridge_predict(
        contexts[train_mask],
        targets[train_mask],
        contexts[test_mask],
        args.ridge,
    )
    combined_prediction, _ = ridge_predict(
        torch.cat([features, contexts], dim=-1)[train_mask],
        targets[train_mask],
        torch.cat([features, contexts], dim=-1)[test_mask],
        args.ridge,
    )
    train_raw_prediction, _ = ridge_predict(
        features[train_mask],
        targets[train_mask],
        features[train_mask],
        args.ridge,
    )
    train_residual = targets[train_mask] - train_raw_prediction
    test_targets = targets[test_mask]
    test_residual = test_targets - raw_prediction
    residual_slot_prediction, _ = ridge_predict(
        contexts[train_mask],
        train_residual,
        contexts[test_mask],
        args.ridge,
    )
    raw_mse = float((raw_prediction - test_targets).square().mean())
    slot_mse = float((slot_prediction - test_targets).square().mean())
    combined_mse = float((combined_prediction - test_targets).square().mean())
    zero_mse = float((zero_prediction - test_targets).square().mean())
    residual_mse = float(
        (residual_slot_prediction - test_residual).square().mean()
    )
    residual_eta_squared = grouped_eta_squared(
        test_residual,
        primary[test_mask],
        model.max_slots,
    )
    usage = usage / sum(
        len(store.indices(count)[: args.cases_per_count])
        for count in args.block_counts
    )
    routing_stress = finite_routing(model, stress_case)
    checks = {
        "no_dead_slots": max(dead_slots) == 0.0,
        "dynamic_slot_count": min(active_counts) < max(active_counts),
        "identity_margin_above_0_05": identity_margin >= 0.05,
        "residual_slot_eta_squared_above_0_05": residual_eta_squared >= 0.05,
        "slot_predicts_raw_residual": residual_mse < raw_mse,
        "slot_plus_raw_beats_raw_by_1pct": combined_mse <= 0.99 * raw_mse,
        "routing_stress_all_finite": all(routing_stress.values()),
    }
    report = {
        "status": "pass" if all(checks.values()) else "reject_or_redesign",
        "scope": "CPU condition-slot identity and held-out target predictability audit",
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "cases": len(args.block_counts) * args.cases_per_count,
        "block_counts": args.block_counts,
        "cases_per_count": args.cases_per_count,
        "train_cases_per_count": args.train_cases_per_count,
        "metrics": {
            "active_slots_mean": float(
                torch.tensor(active_counts, dtype=torch.float32).mean()
            ),
            "active_slots_min": int(min(active_counts)),
            "active_slots_max": int(max(active_counts)),
            "max_dead_slot_fraction": max(dead_slots),
            "slot_usage": usage.tolist(),
            "slot_usage_entropy": normalized_entropy(usage),
            "same_slot_cosine": float(same_cosine.mean()),
            "different_slot_cosine": float(different_cosine.mean()),
            "identity_margin": identity_margin,
            "absolute_target_slot_eta_squared": absolute_eta_squared,
            "heldout_residual_slot_eta_squared": residual_eta_squared,
            "raw_feature_ridge_mse": raw_mse,
            "slot_context_ridge_mse": slot_mse,
            "slot_plus_raw_ridge_mse": combined_mse,
            "case_mean_mse": zero_mse,
            "slot_raw_residual_mse": residual_mse,
            "slot_raw_residual_relative_improvement": (
                raw_mse - residual_mse
            )
            / raw_mse,
            "slot_context_relative_improvement": (raw_mse - slot_mse) / raw_mse,
            "slot_plus_raw_relative_improvement": (raw_mse - combined_mse)
            / raw_mse,
        },
        "routing_stress": routing_stress,
        "checks": checks,
        "decision": (
            "continue to stable local-context decoder"
            if all(checks.values())
            else "redesign or reject local slots before GPU training"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
