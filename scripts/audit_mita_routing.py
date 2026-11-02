#!/usr/bin/env python3
"""Audit MiTA routing health on a deterministic validation prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_scalar(values: list[torch.Tensor]) -> float:
    return float(torch.stack([value.float().cpu() for value in values]).mean())


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        device,
        taxonomy=None,
    )
    if not getattr(model, "use_family_mita_attention", False):
        raise ValueError("checkpoint does not use MiTA family-stream attention")
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    collected: dict[str, list[torch.Tensor]] = {}
    usage: dict[str, list[torch.Tensor]] = {}
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        zeros = torch.zeros(
            target.shape[0],
            model.target_channels,
            *target.shape[-2:],
            device=device,
        )
        _velocity, aux = model.forward_with_aux(
            zeros,
            torch.zeros(target.shape[0], device=device),
            footprint,
            floor_ids=floor_ids,
        )
        aggregate = aux.get("family_stream") or {}
        for key in (
            "mita_routing_entropy",
            "mita_attention_entropy",
            "mita_effective_expert_fraction",
        ):
            if key in aggregate:
                collected.setdefault(key, []).append(aggregate[key].detach())
        for name, diagnostics in model.family_stream_mixer.last_attention_diagnostics.items():
            usage.setdefault(name, []).append(
                diagnostics["router_usage"].detach().float().cpu()
            )
    if not collected:
        raise RuntimeError("MiTA diagnostics were not emitted")
    aggregates = {key: mean_scalar(values) for key, values in collected.items()}
    stream_usage = {
        name: torch.stack(values).mean(dim=0).tolist()
        for name, values in usage.items()
    }
    for values in stream_usage.values():
        tensor = torch.tensor(values)
        aggregates.setdefault(
            "maximum_stream_expert_share",
            float(tensor.max()),
        )
        aggregates["maximum_stream_expert_share"] = max(
            aggregates["maximum_stream_expert_share"],
            float(tensor.max()),
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "cases": min(args.limit, len(dataset)),
        "taxonomy_channels": taxonomy.num_target_channels,
        "health_thresholds": {
            "routing_entropy_min": 0.25,
            "effective_expert_fraction_min": 0.25,
            "maximum_stream_expert_share_max": 0.90,
        },
        "metrics": aggregates,
        "router_usage_by_stream": stream_usage,
    }
    report["passed"] = bool(
        aggregates["mita_routing_entropy"] >= 0.25
        and aggregates["mita_effective_expert_fraction"] >= 0.25
        and aggregates["maximum_stream_expert_share"] <= 0.90
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
