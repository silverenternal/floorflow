#!/usr/bin/env python3
"""Train a contour-only gate over a train-only retrieval prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.room_partition import batch_room_partition_target
from floorflow.model.retrieval_fusion import LearnedRetrievalFusion
from floorflow.objectives.condition_objective import room_macro_iou_from_labels
from floorflow.representation.contour_memory import ContourPartitionMemory
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-train", type=int, default=1024)
    parser.add_argument("--limit-val", type=int, default=191)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--gate-resolution", type=int, default=64)
    parser.add_argument("--base-prior-weight", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def memory_indices(dataset, memory) -> list[int]:
    lookup = {Path(path).resolve(): index for index, path in enumerate(memory.paths)}
    return [lookup[Path(sample.cache_path).resolve()] for sample in dataset.samples]


@torch.inference_mode()
def evaluate(model, fusion, loader, indices, memory, taxonomy, device):
    fusion.eval()
    values = []
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        zero = torch.zeros(
            target.shape[0],
            model.target_channels,
            *target.shape[-2:],
            device=device,
        )
        _, auxiliary = model.forward_with_aux(
            zero,
            torch.zeros(target.shape[0], device=device),
            footprint,
            floor_ids=floor_ids,
        )
        prior = memory.retrieve_logits(
            footprint.cpu().numpy(),
            taxonomy,
            device,
        )
        logits, _ = fusion(
            auxiliary["room_partition_logits"].float(),
            prior,
            footprint,
        )
        room_ids = torch.tensor(taxonomy.room_ids, device=device)
        prediction = room_ids[logits.argmax(dim=1)] * footprint.long()
        for index in range(target.shape[0]):
            values.append(
                room_macro_iou_from_labels(
                    prediction[index : index + 1],
                    target[index : index + 1],
                    taxonomy,
                )
            )
    return sum(values) / max(1, len(values))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, config, taxonomy, _ = load_checkpoint_model(args.checkpoint, device)
    model.eval().requires_grad_(False)
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=False,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    memory = ContourPartitionMemory.from_cache(args.cache_root, top_k=1)
    train_indices = memory_indices(train, memory)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    fusion = LearnedRetrievalFusion(
        len(taxonomy.room_ids),
        hidden_channels=args.hidden_channels,
        gate_resolution=args.gate_resolution,
        base_prior_weight=args.base_prior_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        fusion.train()
        seen = 0
        total = 0.0
        offset = 0
        for target, footprint, floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            floor_ids = floor_ids.to(device)
            zero = torch.zeros(
                target.shape[0],
                model.target_channels,
                *target.shape[-2:],
                device=device,
            )
            with torch.no_grad():
                _, auxiliary = model.forward_with_aux(
                    zero,
                    torch.zeros(target.shape[0], device=device),
                    footprint,
                    floor_ids=floor_ids,
                )
                exclusions = torch.tensor(
                    train_indices[offset : offset + target.shape[0]],
                    device=device,
                )
                prior = memory.retrieve_logits(
                    footprint.cpu().numpy(),
                    taxonomy,
                    device,
                    exclude_indices=exclusions,
                )
            logits, _ = fusion(
                auxiliary["room_partition_logits"].float(),
                prior,
                footprint,
            )
            partition_target = batch_room_partition_target(
                target,
                footprint,
                taxonomy,
                fill_occluded=False,
            )
            valid = partition_target >= 0
            pixel_loss = F.cross_entropy(
                logits,
                partition_target.clamp_min(0),
                reduction="none",
            )
            loss = (pixel_loss * valid).sum() / valid.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * target.shape[0]
            seen += target.shape[0]
            offset += target.shape[0]
        score = evaluate(model, fusion, val_loader, [], memory, taxonomy, device)
        record = {"epoch": epoch, "train_loss": total / max(1, seen), "room_macro_iou": score}
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {
            "model": fusion.state_dict(),
            "config": vars(args),
            "taxonomy": taxonomy.to_json(),
            "metrics": record,
        }
        torch.save(payload, args.output_dir / "last.pt")
        if score > best:
            best = score
            torch.save(payload, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
