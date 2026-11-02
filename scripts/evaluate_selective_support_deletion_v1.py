#!/usr/bin/env python3
"""Evaluate a selective support checkpoint through a chosen label consumer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dense_query_support_gate_v1 import (
    frozen_probabilities,
    metric_row,
)

from floorflow.data.control_contract import sha256_file
from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.heads.dense_query_support_gate import (
    SelectiveRoomSupportDeletion,
    render_support_conditioned_room_labels,
)
from floorflow.heads.raster_control_bank import (
    load_raster_instance_control_model,
    load_raster_room_model,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dense-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/native_room_384_full_32ep_v1/best.pt",
    )
    parser.add_argument("--query-checkpoint", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--topology-safe", action="store_true")
    parser.add_argument("--min-deletable-component-area", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=None,
        image_mode="native",
    )
    dataset.samples = dataset.samples[
        int(args.offset) : int(args.offset) + int(args.limit)
    ]
    taxonomy = dataset.taxonomy
    payload = torch.load(
        args.gate_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = payload["config"]
    gate = SelectiveRoomSupportDeletion(
        len(taxonomy.room_ids),
        fixed_blend=float(config["fixed_blend"]),
        hidden_channels=int(config["hidden_channels"]),
        preserve_confidence=float(config["preserve_confidence"]),
    ).to(device)
    gate.load_state_dict(payload["model"])
    dense_model = load_raster_room_model(
        args.dense_checkpoint,
        taxonomy,
        device,
    )
    query_model = load_raster_instance_control_model(
        args.query_checkpoint,
        taxonomy,
        device,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    rows = {"blend50": [], "selective": []}
    for target, footprint, image, _floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        dense_probability, query_probability, query_support = (
            frozen_probabilities(
                dense_model,
                query_model,
                footprint,
                image,
            )
        )
        gate_output = gate(
            dense_probability,
            query_probability,
            query_support,
            footprint,
            keep_threshold=float(config.get("keep_threshold", 0.5)),
        )
        footprint_4d = footprint[:, None] if footprint.ndim == 3 else footprint
        blend_labels = (
            room_ids[gate_output["probability"].argmax(dim=1)]
            * footprint_4d[:, 0].long()
        )
        selective_labels = render_support_conditioned_room_labels(
            gate_output["probability"],
            gate_output["keep_mask"],
            room_ids,
            footprint_4d,
            topology_safe=args.topology_safe,
            minimum_deletable_component_area=(
                args.min_deletable_component_area
            ),
        )
        for index, name in enumerate(names):
            for arm, labels in (
                ("blend50", blend_labels),
                ("selective", selective_labels),
            ):
                rows[arm].append(
                    {
                        "case": str(name),
                        "index": int(args.offset) + len(rows[arm]),
                        **metric_row(
                            labels[index].cpu().numpy(),
                            target[index].cpu().numpy(),
                            footprint_4d[index, 0].cpu().numpy(),
                            taxonomy,
                        ),
                    }
                )
    summary = {
        arm: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
            if key not in {"case", "index"}
        }
        for arm, values in rows.items()
    }
    paired = {
        key: paired_delta(
            rows["selective"],
            rows["blend50"],
            key,
            iterations=args.iterations,
            seed=args.seed,
            lower_is_better=key == "count_mae",
        )
        for key in ("room_iou", "instance_f1", "count_mae", "room_support_iou")
    }
    gates = {
        "room_iou_non_regression": paired["room_iou"]["mean_delta"] >= 0.0,
        "instance_f1_non_regression": (
            paired["instance_f1"]["mean_delta"] >= 0.0
        ),
        "count_mae_non_regression": (
            paired["count_mae"]["mean_delta"] <= 0.0
        ),
        "support_iou_non_regression": (
            paired["room_support_iou"]["mean_delta"] >= 0.0
        ),
    }
    gates["passed"] = all(gates.values())
    report = {
        "schema_version": "selective-support-deletion-evaluation-v1",
        "protocol": "predicted_control",
        "target_used_at_inference": False,
        "test_opened": False,
        "validation_cases": len(dataset),
        "validation_offset": args.offset,
        "topology_safe": args.topology_safe,
        "min_deletable_component_area": (
            args.min_deletable_component_area
        ),
        "config": {
            "gate_checkpoint": str(args.gate_checkpoint),
            "gate_checkpoint_sha256": sha256_file(args.gate_checkpoint),
            "dense_checkpoint": str(args.dense_checkpoint),
            "query_checkpoint": str(args.query_checkpoint),
            "taxonomy_sha256": sha256_file(args.cache_root / "taxonomy.json"),
            "source_config": config,
        },
        "summary": summary,
        "paired": paired,
        "gates": gates,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "blend50": summary["blend50"],
                "selective": summary["selective"],
                "paired": paired,
                "gates": gates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
