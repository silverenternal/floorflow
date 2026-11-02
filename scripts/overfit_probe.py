#!/usr/bin/env python3
"""Check that the current model can fit a small training subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.flow.core import decode_floorplans, flow_matching_loss, interpolation, sample_prior, target_to_continuous
from floorflow.model.backbone import FloorplanFlowModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/probes/overfit")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-positive-weight", type=float, default=5.0)
    parser.add_argument("--topology-aux-weight", type=float, default=0.0)
    return parser.parse_args()


@torch.no_grad()
def teacher_forced_accuracy(
    model,
    target,
    footprint,
    floor_ids,
    taxonomy,
    time_value: float,
    objective_mode: str,
) -> float:
    x1 = target_to_continuous(target, model.target_channels)
    if objective_mode == "simplex":
        from floorflow.flow.core import sample_simplex_prior

        x0 = sample_simplex_prior(footprint, model.target_channels)
    else:
        x0 = sample_prior(footprint, model.target_channels)
    t = torch.full((target.shape[0],), time_value, device=target.device)
    x_t = interpolation(x0, x1, t)
    raw = model(x_t, t, footprint, floor_ids=floor_ids)
    predicted_x1 = x_t + (1.0 - time_value) * raw
    labels = decode_floorplans(predicted_x1)
    mask = footprint > 0.5
    return float(((labels == target) & mask).float().sum().cpu()) / max(1, int(mask.sum().cpu()))


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)
    torch.manual_seed(7)
    device = torch.device(args.device)
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit,
        preload=True,
    )
    target_channels = dataset.taxonomy.num_target_channels
    model = FloorplanFlowModel(
        target_channels=target_channels,
        base_channels=args.base_channels,
        levels=3,
        condition_dropout=0.0,
        semantic_channel_attention=True,
        semantic_relation_bias=False,
        slot_mixing=True,
        num_slots=8,
        dynamic_output_head=True,
        topology_aux=True,
        num_rooms=len(dataset.taxonomy.room_ids),
        channel_relations=dataset.taxonomy.channel_prior(),
        advanced_streams=True,
        floor_conditioning=True,
    ).to(device)
    positive_weights = torch.from_numpy(
        load_positive_weights(args.cache_root, target_channels, args.max_positive_weight)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    indices = list(range(len(dataset)))
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for offset in range(0, len(indices), args.batch_size):
            batch_indices = indices[offset : offset + args.batch_size]
            rows = [dataset[index] for index in batch_indices]
            target = torch.stack([row[0] for row in rows]).to(device)
            footprint = torch.stack([row[1] for row in rows]).to(device)
            floor_ids = torch.tensor([row[2] for row in rows], device=device)
            loss, components = flow_matching_loss(
                model,
                target,
                footprint,
                positive_class_weights=positive_weights,
                floor_ids=floor_ids,
                taxonomy=dataset.taxonomy,
                topology_aux_weight=args.topology_aux_weight,
                objective_mode="simplex",
                semantic_head_weight=0.50,
                direct_semantic_weight=0.50,
                boundary_loss_weight=0.10,
                wall_loss_weight=0.10,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch_indices)
        rows = [dataset[index] for index in indices]
        target = torch.stack([row[0] for row in rows]).to(device)
        footprint = torch.stack([row[1] for row in rows]).to(device)
        floor_ids = torch.tensor([row[2] for row in rows], device=device)
        accuracy = teacher_forced_accuracy(
            model, target, footprint, floor_ids, dataset.taxonomy, 0.80, "simplex"
        )
        accuracy_late = teacher_forced_accuracy(
            model, target, footprint, floor_ids, dataset.taxonomy, 0.95, "simplex"
        )
        history.append(
            {
                "epoch": epoch,
                "loss": total / len(indices),
                "velocity_loss": float(components["velocity_loss"]),
                "category_loss": float(components["category_loss"]),
                "teacher_forced_accuracy_t080": accuracy,
                "teacher_forced_accuracy_t095": accuracy_late,
            }
        )
        print(json.dumps(history[-1]), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "taxonomy": dataset.taxonomy.to_json()},
        args.output_dir / "overfit.pt",
    )
    report = {
        "limit": len(dataset),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "target_channels": target_channels,
        "objective_mode": "simplex",
        "history": history,
        "passed": history[-1]["teacher_forced_accuracy_t080"] >= 0.90,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
