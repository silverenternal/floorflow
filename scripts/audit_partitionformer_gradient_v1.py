"""Audit partition-loss gradient balance on one real RPLAN batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    collate_rplan_items,
    load_rplan_rows,
)
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.unified_floorplan_generator import (
    UnifiedFloorplanGenerator,
    UnifiedGeneratorConfig,
)
from floorflow.training.unified_plan_generator_training import (
    unified_generator_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    groups = {
        "identity_decoder": model.identity_decoder,
        "spatial_encoder": model.spatial_encoder,
        "all": model,
    }
    result = {}
    for name, module in groups.items():
        squared = torch.zeros((), device=next(model.parameters()).device)
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared = squared + parameter.grad.float().pow(2).sum()
        result[name] = float(squared.sqrt())
    return result


def _run_loss(
    model: UnifiedFloorplanGenerator,
    batch: dict,
    taxonomy: FloorplanTaxonomy,
    weight: float,
) -> tuple[float, dict[str, float], dict[str, float]]:
    model.zero_grad(set_to_none=True)
    output = model(
        batch["boundary_mask"],
        batch["boundary_valid"],
        batch["boundary_points"],
        batch["entrance_points"],
        batch["entrance_valid"],
    )
    loss, components = unified_generator_loss(
        output,
        batch,
        taxonomy,
        partition_consistency_weight=float(weight),
    )
    loss.backward()
    return (
        float(loss.detach()),
        {key: float(value) for key, value in components.items()},
        _gradient_norms(model),
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and str(device) != "cuda:0":
        raise ValueError("FloorFlow GPU diagnostics are restricted to cuda:0")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    taxonomy = FloorplanTaxonomy.from_json(checkpoint["taxonomy"])
    model = UnifiedFloorplanGenerator(
        taxonomy,
        UnifiedGeneratorConfig(**checkpoint["config"]["model"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.train()
    data = args.data or Path(checkpoint["config"]["dataset"]["train"])
    limits = RPlanTensorLimits(
        **{
            key: int(value)
            for key, value in checkpoint["config"]["tensors"].items()
        }
    )
    dataset = RPlanBoundaryDataset(
        data,
        taxonomy,
        rows=load_rplan_rows(
            data,
            offset=int(args.offset),
            limit=int(args.batch_size),
        ),
        limits=limits,
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=0,
                collate_fn=collate_rplan_items,
            )
        )
    )
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    base_loss, _base_components, base_grad = _run_loss(
        model,
        batch,
        taxonomy,
        0.0,
    )
    full_loss, full_components, full_grad = _run_loss(
        model,
        batch,
        taxonomy,
        float(
            checkpoint["config"]["loss_weights"].get(
                "partition_consistency",
                0.5,
            )
        ),
    )
    report = {
        "schema": "floorflow-partitionformer-gradient-audit-v1",
        "checkpoint": str(args.checkpoint),
        "data": str(data),
        "offset": int(args.offset),
        "batch_size": int(args.batch_size),
        "base_loss": base_loss,
        "full_loss": full_loss,
        "partition_component": full_components[
            "room_mask_partition_query_loss"
        ],
        "partition_exact": full_components[
            "room_mask_partition_query_exact"
        ],
        "base_gradient": base_grad,
        "full_gradient": full_grad,
        "gradient_ratio": {
            key: full_grad[key] / max(base_grad[key], 1.0e-12)
            for key in full_grad
        },
        "test_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
