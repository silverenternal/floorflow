#!/usr/bin/env python3
"""Audit dataset scale, parameter count, and effective training exposure."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_split(root: Path, taxonomy) -> dict:
    paths = sorted((root / "train").rglob("*.npz"))
    val_paths = sorted((root / "val").rglob("*.npz"))
    test_paths = sorted((root / "test").rglob("*.npz"))
    presence = {split: np.zeros(taxonomy.num_classes, dtype=np.int64) for split in ("train", "val", "test")}
    pixels = {split: np.zeros(taxonomy.num_classes, dtype=np.int64) for split in ("train", "val", "test")}
    for split, split_paths in (("train", paths), ("val", val_paths), ("test", test_paths)):
        for path in split_paths:
            with np.load(path) as payload:
                target = payload["target"]
                present = np.unique(target)
                presence[split][present] += 1
                pixels[split] += np.bincount(target.ravel(), minlength=taxonomy.num_classes)
    classes = []
    for class_id in taxonomy.room_ids:
        classes.append(
            {
                "class_id": class_id,
                "class_name": taxonomy.class_names[class_id],
                "train_images": int(presence["train"][class_id]),
                "val_images": int(presence["val"][class_id]),
                "test_images": int(presence["test"][class_id]),
                "train_pixels": int(pixels["train"][class_id]),
                "test_pixels": int(pixels["test"][class_id]),
            }
        )
    return {
        "cases": {
            "train": len(paths),
            "val": len(val_paths),
            "test": len(test_paths),
        },
        "pixels_per_epoch_at_256": len(paths) * 256 * 256,
        "room_classes": sorted(classes, key=lambda row: row["train_images"]),
    }


def main() -> None:
    args = parse_args()
    model, config, taxonomy, _ = load_checkpoint_model(args.checkpoint, "cpu")
    split = audit_split(args.cache_root, taxonomy)
    parameter_counts = {
        "total": sum(value.numel() for value in model.parameters()),
        "condition_encoder": sum(value.numel() for value in model.condition_encoder.parameters()),
        "encoder": sum(value.numel() for value in model.encoder.parameters()),
        "decoder": sum(value.numel() for value in model.decoder.parameters()),
        "room_partition": sum(value.numel() for value in model.room_partition.parameters()),
        "structured_objects": sum(value.numel() for value in model.structured_objects.parameters()),
    }
    stages = [
        {"name": "structured_512_curriculum", "samples": 512, "epochs": 3, "batch_size": 8},
        {"name": "structured_1024_curriculum", "samples": 1024, "epochs": 3, "batch_size": 8},
        {"name": "structured_full", "samples": split["cases"]["train"], "epochs": 6, "batch_size": 16},
        {"name": "structured_full_low_lr", "samples": split["cases"]["train"], "epochs": 3, "batch_size": 16},
        {"name": "object_heads_frozen_backbone", "samples": split["cases"]["train"], "epochs": 5, "batch_size": 16},
    ]
    for stage in stages:
        stage["updates"] = math.ceil(stage["samples"] / stage["batch_size"]) * stage["epochs"]
        stage["sample_exposures"] = stage["samples"] * stage["epochs"]
    report = {
        "checkpoint": str(args.checkpoint),
        "configuration": config,
        "data": split,
        "parameters": parameter_counts,
        "training_stages": stages,
        "main_model_sample_exposures": sum(
            stage["sample_exposures"] for stage in stages if stage["name"] != "object_heads_frozen_backbone"
        ),
        "main_model_updates": sum(
            stage["updates"] for stage in stages if stage["name"] != "object_heads_frozen_backbone"
        ),
        "equivalent_full_train_epochs": sum(
            stage["epochs"] * stage["samples"] / split["cases"]["train"]
            for stage in stages
            if stage["name"] != "object_heads_frozen_backbone"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "parameters": parameter_counts,
                "data": split["cases"],
                "pixels_per_epoch": split["pixels_per_epoch_at_256"],
                "main_model_updates": report["main_model_updates"],
                "equivalent_full_train_epochs": report["equivalent_full_train_epochs"],
                "rare_room_classes": split["room_classes"][:8],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
