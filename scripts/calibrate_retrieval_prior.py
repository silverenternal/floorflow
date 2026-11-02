#!/usr/bin/env python3
"""Calibrate class-wise nearest-contour prior strength on validation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.objectives.condition_objective import room_macro_iou_from_labels
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.data.dataset import CubiCasaFloorplanDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--weights-report", type=Path, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--weights-alpha", type=float, default=1.0)
    parser.add_argument("--weights-base", type=float, default=0.75)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, taxonomy, _ = load_checkpoint_model(args.checkpoint, device)
    memory = ContourPartitionMemory.from_cache(args.cache_root, top_k=1)
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        augment=False,
    )
    stop = (
        len(dataset)
        if args.limit is None
        else min(len(dataset), args.offset + args.limit)
    )
    evaluation_set = Subset(dataset, range(args.offset, stop))
    loader = DataLoader(
        evaluation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    stats = {
        weight: {class_id: [0.0, 0.0] for class_id in taxonomy.room_ids}
        for weight in args.weights
    }
    cached = []
    with torch.no_grad():
        for target, footprint, floor_ids, _names in loader:
            target = target.to(device)
            footprint = footprint.to(device)
            zero = torch.zeros(
                target.shape[0],
                model.target_channels,
                *target.shape[-2:],
                device=device,
            )
            _, aux = model.forward_with_aux(
                zero,
                torch.zeros(target.shape[0], device=device),
                footprint,
                floor_ids=floor_ids.to(device),
            )
            prior = memory.retrieve_logits(
                footprint.cpu().numpy(),
                taxonomy,
                device,
            )
            cached.append((aux["room_partition_logits"], prior, target, footprint))
            if args.weights_report is None:
                for weight in args.weights:
                    prediction = room_ids[
                        (aux["room_partition_logits"] + weight * prior).argmax(dim=1)
                    ] * footprint.long()
                    for class_id in taxonomy.room_ids:
                        predicted = prediction == class_id
                        expected = target == class_id
                        stats[weight][class_id][0] += float(
                            (predicted & expected).sum().cpu()
                        )
                        stats[weight][class_id][1] += float(
                            (predicted | expected).sum().cpu()
                        )
    if args.weights_report is not None:
        calibration = json.loads(
            args.weights_report.read_text(encoding="utf-8")
        )
        selected_weights = [
            float(args.weights_base)
            + float(args.weights_alpha)
            * (float(weight) - float(args.weights_base))
            for weight in calibration["weights"]
        ]
        selected_weights = [
            min(1.0, max(0.0, weight)) for weight in selected_weights
        ]
        diagnostics = calibration.get("classes", [])
    else:
        selected_weights = []
        diagnostics = []
        for class_id in taxonomy.room_ids:
            candidates = []
            for weight in args.weights:
                intersection, union = stats[weight][class_id]
                candidates.append(
                    {
                        "weight": weight,
                        "iou": intersection / max(1.0, union),
                        "support": union,
                    }
                )
            supported = [row for row in candidates if row["support"] > 0]
            selected = max(
                supported,
                key=lambda row: (row["iou"], -row["weight"]),
            ) if supported else {"weight": 0.0, "iou": 0.0, "support": 0.0}
            selected_weights.append(float(selected["weight"]))
            diagnostics.append(
                {
                    "class_id": class_id,
                    "class_name": taxonomy.class_names[class_id],
                    "selected": selected,
                    "candidates": candidates,
                }
            )
    class_weights = torch.tensor(selected_weights, device=device)
    per_case = []
    with torch.no_grad():
        for logits, prior, target, footprint in cached:
            prediction = room_ids[
                (logits + prior * class_weights[None, :, None, None]).argmax(dim=1)
            ] * footprint.long()
            for index in range(target.shape[0]):
                per_case.append(
                    room_macro_iou_from_labels(
                        prediction[index : index + 1],
                        target[index : index + 1],
                        taxonomy,
                    )
                )
    report = {
        "split": "val",
        "offset": args.offset,
        "cases": len(evaluation_set),
        "retrieval_top_k": 1,
        "weights": selected_weights,
        "weights_report": (
            str(args.weights_report) if args.weights_report is not None else None
        ),
        "room_macro_iou": float(np.mean(per_case)),
        "classes": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "room_macro_iou": report["room_macro_iou"],
                "weights": selected_weights,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
