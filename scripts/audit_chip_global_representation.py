#!/usr/bin/env python3
"""Measure cross-case information retained by the chip global latent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.flow_adapter import encode_target_and_condition
from floorflow.chip_flow.slot_model import ChipSlotFlowModel

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument("--cases-per-count", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_global_representation_probe_v1.json",
    )
    return parser.parse_args()


def load_model(path: Path) -> ChipSlotFlowModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = ChipSlotFlowModel(
        feature_dim=int(config.get("feature_dim", 19)),
        hidden=int(config["hidden"]),
        global_dim=int(config.get("global_dim", config["hidden"])),
        target_layers=int(config.get("target_layers", 4)),
        condition_layers=int(config.get("condition_layers", 2)),
        decoder_layers=int(config.get("decoder_layers", 4)),
        heads=int(config["heads"]),
        graph_conditioning=True,
        output_mode=config.get("output_mode", "velocity"),
        global_pool_stats=bool(config.get("global_pool_stats", False)),
        global_film=bool(config.get("global_film", False)),
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def effective_rank(values: torch.Tensor) -> float:
    centered = values - values.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.float())
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1.0e-12)
    entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum()
    return float(entropy.exp())


def main() -> None:
    args = parse_args()
    model = load_model(args.checkpoint)
    store = PackedCaseStore(args.cache)
    targets = []
    conditions = []
    with torch.no_grad():
        for count in sorted(store.by_block_count):
            indices = store.indices(count)[-args.cases_per_count :]
            batch = store.batch(indices, device="cpu")
            target, condition, _ = encode_target_and_condition(model, batch)
            targets.append(target)
            conditions.append(condition)
    target = torch.cat(targets)
    condition = torch.cat(conditions)
    residual = target - condition
    target_std = target.std(dim=0, correction=0)
    condition_std = condition.std(dim=0, correction=0)
    metrics = {
        "cases": int(target.shape[0]),
        "global_dim": int(target.shape[1]),
        "target_mean_norm": float(target.norm(dim=-1).mean()),
        "condition_mean_norm": float(condition.norm(dim=-1).mean()),
        "residual_mean_norm": float(residual.norm(dim=-1).mean()),
        "residual_to_target_norm": float(
            residual.norm(dim=-1).mean() / target.norm(dim=-1).mean()
        ),
        "target_condition_cosine": float(
            torch.nn.functional.cosine_similarity(
                target,
                condition,
                dim=-1,
            ).mean()
        ),
        "target_dim_std_mean": float(target_std.mean()),
        "target_dim_std_max": float(target_std.max()),
        "condition_dim_std_mean": float(condition_std.mean()),
        "target_effective_rank": effective_rank(target),
        "condition_effective_rank": effective_rank(condition),
        "target_top_singular_values": torch.linalg.svdvals(
            target - target.mean(dim=0, keepdim=True)
        )[:8].tolist(),
    }
    checks = {
        "target_cross_case_std_above_1e-4": metrics["target_dim_std_mean"]
        > 1.0e-4,
        "target_effective_rank_above_8": metrics["target_effective_rank"] > 8.0,
        "condition_is_not_target_base": metrics["residual_to_target_norm"] > 1.0,
    }
    report = {
        "status": "informative" if checks["target_cross_case_std_above_1e-4"] else "collapsed",
        "scope": "global-latent cross-case information probe; no quality claim",
        "checkpoint": str(args.checkpoint),
        "metrics": metrics,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
