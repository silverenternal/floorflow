#!/usr/bin/env python3
"""Audit guided-CFM objective gradient balance on one deterministic batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.flow.core import target_to_continuous
from floorflow.flow.guided_control import guided_source_from_labels
from floorflow.flow.mask_policies import AttributeMaskPolicy
from floorflow.heads.room_partition import (
    batch_room_partition_target,
    room_partition_loss,
)
from floorflow.model.sparse_control_attention import MaskedControlFlow
from floorflow.objectives.cfm_structural import cfm_structural_loss, predicted_x1
from floorflow.objectives.gradient_balance import gradient_diagnostics
from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.training.masked_conditioning import prepare_masked_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.enable_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        device,
        taxonomy=None,
    )
    model = MaskedControlFlow(model).to(device)
    if "sparse_attention" in payload:
        model.sparse_attention.load_state_dict(
            payload["sparse_attention"],
            strict=False,
        )
    model.train()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.batch_size,
        augment=False,
    )
    target, footprint, floor_ids, _names = next(
        iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False))
    )
    target = target.to(device)
    footprint = footprint.to(device)
    floor_ids = floor_ids.to(device)
    full_source = guided_source_from_labels(target, footprint, taxonomy)
    missing_token = payload.get(
        "missing_token",
        torch.zeros(model.target_channels),
    ).to(device)
    prepared = prepare_masked_condition(
        full_source,
        footprint,
        taxonomy,
        missing_token,
        missing_ratio=args.mask_ratio,
        floor_ids=floor_ids,
        mask_policy=AttributeMaskPolicy(),
    )
    x1 = target_to_continuous(target, model.target_channels)
    time = torch.full((target.shape[0],), 0.5, device=device)
    x_t = 0.5 * prepared.control + 0.5 * x1
    velocity, auxiliary = model.forward_with_aux(
        x_t,
        time,
        footprint[:, None],
        floor_ids=floor_ids,
        control=prepared.control,
        control_mask=prepared.missing_mask,
    )
    mask = footprint[:, None].float()
    flow = (
        (velocity - (x1 - prepared.control)).square() * mask
    ).sum() / mask.sum().clamp_min(1.0) / velocity.shape[1]
    partition_target = batch_room_partition_target(
        target,
        footprint,
        taxonomy,
        fill_occluded=False,
    )
    partition = room_partition_loss(
        auxiliary["room_partition_logits"],
        partition_target,
    )["loss"]
    semantic_logits = auxiliary["semantic_logits"]
    semantic = F.cross_entropy(
        torch.cat(
            [torch.zeros_like(semantic_logits[:, :1]), semantic_logits],
            dim=1,
        ),
        target,
    )
    structural, _components = cfm_structural_loss(
        predicted_x1(x_t, velocity, time).float(),
        x1.float(),
        mask,
        taxonomy,
    )
    losses = {
        "flow": flow,
        "partition": partition,
        "semantic": semantic,
        "x1_structural": structural,
    }
    route_balance = auxiliary.get("sparse_control", {}).get(
        "route_balance_loss"
    )
    if route_balance is not None:
        losses["route_balance"] = route_balance
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    report = {
        "checkpoint": str(args.checkpoint),
        "loss_values": {
            name: float(loss.detach().cpu()) for name, loss in losses.items()
        },
        **gradient_diagnostics(losses, parameters),
        "mask_ratio": args.mask_ratio,
        "best_of_n_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
