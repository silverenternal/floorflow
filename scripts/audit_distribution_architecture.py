#!/usr/bin/env python3
"""Audit real-checkpoint shape, gradient, and memory contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.cfm_model import DistributionCFMModel
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.joint_loss import joint_distribution_cfm_loss
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
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--pixel-grid", type=int, default=16)
    parser.add_argument("--element-grid", type=int, default=16)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    base, config, taxonomy, _ = load_checkpoint_model(
        args.checkpoint,
        device,
        taxonomy=None,
    )
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    counts = {
        "wall": args.queries,
        "opening": args.queries,
        "room": args.queries,
        "furniture": args.queries,
        "support": max(1, args.queries // 2),
    }
    model = DistributionCFMModel(
        base,
        hierarchy,
        hidden_channels=args.hidden_channels,
        pixel_grid=args.pixel_grid,
        element_grid=args.element_grid,
        queries_per_group=counts,
    ).to(device)
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.batch_size,
        augment=False,
    )
    target, footprint, floor_ids, _ = next(
        iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False))
    )
    target = target.to(device)
    footprint = footprint[:, None].to(device)
    floor_ids = floor_ids.to(device)
    positive = torch.from_numpy(
        load_positive_weights(
            args.cache_root,
            taxonomy.num_target_channels,
            20.0,
        )
    ).to(device)
    weights = torch.cat([torch.ones(1, device=device), positive])
    initial = model.contour_initial_state(
        footprint,
        floor_ids=floor_ids,
    )
    loss, components = joint_distribution_cfm_loss(
        model,
        target,
        initial["cfm"]["state"],
        footprint,
        taxonomy,
        hierarchy,
        floor_ids=floor_ids,
        queries_per_group=counts,
        class_weights=weights,
        contour_output=initial,
    )
    loss.backward()
    report = {
        "checkpoint": str(args.checkpoint),
        "condition_mode": "contour_only",
        "batch_size": int(target.shape[0]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "feature_channels": base.output.in_channels,
        "target_channels": model.target_channels,
        "pixel_shape": list(initial["pixel"]["logits"].shape),
        "source_shape": list(initial["cfm"]["state"].shape),
        "element_shapes": {
            name: list(value["mask_logits"].shape)
            for name, value in initial["elements"].items()
        },
        "loss": float(loss.detach()),
        "finite_loss": bool(torch.isfinite(loss)),
        "gate_gradients": {
            "cfm_element_blend": model.cfm_bridge.element_blend.grad is not None,
            "element_to_pixel": model.hub.element_feedback_scale.grad is not None,
            "pixel_to_element": any(
                value.grad is not None
                for value in model.hub.element.group_queries.values()
            ),
        },
        "peak_cuda_memory_gb": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "component_keys": sorted(components),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
