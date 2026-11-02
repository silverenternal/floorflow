#!/usr/bin/env python3
"""CPU probe for whether slot residuals are more predictable than absolute slots."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_slot_predictability_probe_v1.json",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> ChipSlotTransportFlowModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
        residual_slots=True,
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval()


@torch.no_grad()
def collect_states(
    model: ChipSlotTransportFlowModel,
    store: PackedCaseStore,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    condition_rows = []
    target_rows = []
    residual_rows = []
    mask_rows = []
    slot_target_norms = []
    slot_residual_norms = []
    identity_cosines = []
    for count in args.block_counts:
        indices = store.indices(count)[: args.cases_per_count]
        batch = store.batch(indices, device="cpu")
        graph = graph_inputs(batch)
        context = model.encode_condition_context(batch["features"], **graph)
        condition = model.condition_vector(context).reshape(
            len(indices),
            model.state_tokens,
            model.global_dim,
        )
        target = model.encode_target(
            batch["target_latent"],
            batch["features"],
            **graph,
        )
        mask = model.state_mask(context)
        residual = torch.cat(
            [
                target[:, :1],
                target[:, 1:] - condition[:, 1:],
            ],
            dim=1,
        )
        condition_rows.append(condition.flatten(1))
        target_rows.append(target.flatten(1))
        residual_rows.append(residual.flatten(1))
        mask_rows.append(mask[:, :, None].expand_as(target).flatten(1))

        active_condition = condition[:, 1:][mask[:, 1:]]
        active_target = target[:, 1:][mask[:, 1:]]
        active_residual = residual[:, 1:][mask[:, 1:]]
        slot_target_norms.append(active_target.norm(dim=-1))
        slot_residual_norms.append(active_residual.norm(dim=-1))
        identity_cosines.append(
            torch.nn.functional.cosine_similarity(
                active_condition,
                active_target,
                dim=-1,
            )
        )
    return {
        "condition": torch.cat(condition_rows),
        "target": torch.cat(target_rows),
        "residual": torch.cat(residual_rows),
        "mask": torch.cat(mask_rows).bool(),
        "target_norm": torch.cat(slot_target_norms),
        "residual_norm": torch.cat(slot_residual_norms),
        "identity_cosine": torch.cat(identity_cosines),
    }


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return (prediction - target).square()[mask].mean()


def nearest_neighbours(condition: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(condition, dim=-1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-float("inf"))
    return similarity.argmax(dim=1)


def main() -> None:
    args = parse_args()
    store = PackedCaseStore(args.cache)
    model = load_model(args.checkpoint)
    states = collect_states(model, store, args)
    neighbour = nearest_neighbours(states["condition"])
    zero_prediction = states["condition"]
    absolute_prediction = states["target"][neighbour]
    residual_prediction = zero_prediction + states["residual"][neighbour]
    mask = states["mask"]
    metrics = {
        "condition_relative_mse": float(
            masked_mse(zero_prediction, states["target"], mask)
        ),
        "nearest_absolute_mse": float(
            masked_mse(absolute_prediction, states["target"], mask)
        ),
        "nearest_residual_mse": float(
            masked_mse(residual_prediction, states["target"], mask)
        ),
        "mean_absolute_slot_norm": float(states["target_norm"].mean()),
        "mean_residual_slot_norm": float(states["residual_norm"].mean()),
        "residual_to_absolute_norm": float(
            states["residual_norm"].mean() / states["target_norm"].mean()
        ),
        "mean_identity_cosine": float(states["identity_cosine"].mean()),
        "absolute_minus_residual_mse": float(
            masked_mse(absolute_prediction, states["target"], mask)
            - masked_mse(residual_prediction, states["target"], mask)
        ),
    }
    checks = {
        "residual_scale_is_smaller": (
            metrics["mean_residual_slot_norm"]
            < metrics["mean_absolute_slot_norm"]
        ),
        "residual_beats_absolute_neighbour": (
            metrics["nearest_residual_mse"]
            < metrics["nearest_absolute_mse"]
        ),
        "residual_beats_untransported_condition": (
            metrics["nearest_residual_mse"]
            < metrics["condition_relative_mse"]
        ),
        "slot_identity_has_positive_cosine": (
            metrics["mean_identity_cosine"] > 0.0
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "retire_or_redesign",
        "scope": "CPU slot-identity and residual-scale predictability probe; no quality claim",
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "cases": int(states["condition"].shape[0]),
        "block_counts": args.block_counts,
        "cases_per_count": args.cases_per_count,
        "metrics": metrics,
        "checks": checks,
        "decision": (
            "continue residual transport"
            if all(checks.values())
            else "do not spend GPU time on residual transport until scale or identity is fixed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
