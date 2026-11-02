#!/usr/bin/env python3
"""Calibrate class-wise opening and furniture confidence thresholds on validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.data.dataset import CubiCasaFloorplanDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _group_scores(
    logits: torch.Tensor,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(
        torch.cat([background[:, None], logits], dim=1),
        dim=1,
    )
    return 1.0 - probabilities[:, 0], logits.argmax(dim=1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, taxonomy, _ = load_checkpoint_model(args.checkpoint, device)
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    class_sets = {
        "opening": torch.tensor(taxonomy.opening_ids, device=device),
        "furniture": torch.tensor(taxonomy.furniture_ids, device=device),
    }
    stats = {
        name: {
            int(class_id): {threshold: [0.0, 0.0, 0.0] for threshold in args.thresholds}
            for class_id in class_ids.tolist()
        }
        for name, class_ids in class_sets.items()
    }
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
            outputs = aux["structured_objects"]
            valid = footprint.bool()
            for name, class_ids in class_sets.items():
                confidence, class_index = _group_scores(
                    outputs[f"{name}_logits"],
                    outputs[f"{name}_background_logits"],
                )
                predicted_ids = class_ids[class_index]
                for class_id_tensor in class_ids:
                    class_id = int(class_id_tensor)
                    expected = (target == class_id) & valid
                    for threshold in args.thresholds:
                        predicted = (predicted_ids == class_id) & (confidence > threshold) & valid
                        entry = stats[name][class_id][threshold]
                        entry[0] += float((predicted & expected).sum().cpu())
                        entry[1] += float((predicted & ~expected).sum().cpu())
                        entry[2] += float((~predicted & expected).sum().cpu())
    report = {"split": args.split, "cases": len(dataset), "groups": {}}
    for name, class_ids in class_sets.items():
        rows = []
        thresholds = []
        for class_id in class_ids.tolist():
            candidates = []
            for threshold in args.thresholds:
                tp, fp, fn = stats[name][class_id][threshold]
                f1 = 2.0 * tp / max(1.0e-9, 2.0 * tp + fp + fn)
                candidates.append(
                    {
                        "threshold": threshold,
                        "f1": f1,
                        "precision": tp / max(1.0, tp + fp),
                        "recall": tp / max(1.0, tp + fn),
                        "support": tp + fn,
                    }
                )
            supported = [row for row in candidates if row["support"] > 0]
            selected = max(
                supported,
                key=lambda row: (row["f1"], row["threshold"]),
            ) if supported else {"threshold": 1.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "support": 0.0}
            thresholds.append(float(selected["threshold"]))
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": taxonomy.class_names[class_id],
                    "selected": selected,
                    "candidates": candidates,
                }
            )
        report["groups"][name] = {
            "thresholds": thresholds,
            "macro_f1": float(np.mean([row["selected"]["f1"] for row in rows])),
            "classes": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: report["groups"][name]["macro_f1"] for name in report["groups"]}, indent=2))


if __name__ == "__main__":
    main()
