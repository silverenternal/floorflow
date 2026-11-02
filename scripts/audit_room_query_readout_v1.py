#!/usr/bin/env python3
"""Audit dense, query, and existence-gated room-mask readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.data.predicted_control_raster import (
    DualControlRasterDataset,
    PredictedControlRasterDataset,
)
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.raster_control_bank import (
    load_raster_instance_control_model,
)
from floorflow.heads.room_mask_fusion import fused_room_mask_logits

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/train/room_query_fullres_softroute_screen256_v1/best.pt"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / ".data/cubicasa5k",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--control-raster-root", type=Path, default=None)
    parser.add_argument("--dual-control-raster", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/room_query_readout_audit_cal64_v1.json"
        ),
    )
    return parser.parse_args()


def _class_mass_probability(
    mask_probability: torch.Tensor,
    classes: torch.Tensor,
    *,
    existence: torch.Tensor | None = None,
    power: float = 1.0,
) -> torch.Tensor:
    """Fuse query masks into a normalized room-class probability."""

    if existence is not None:
        mask_probability = mask_probability * existence.pow(float(power))[:, :, None, None]
    class_count = classes.shape[-1] - 1
    foreground = classes[..., :class_count].softmax(dim=-1)
    mass = torch.einsum("bqhw,bqk->bkhw", mask_probability, foreground)
    return mass / mass.sum(dim=1, keepdim=True).clamp_min(1.0e-6)


def _winner_probability(
    mask_probability: torch.Tensor,
    classes: torch.Tensor,
    existence: torch.Tensor | None = None,
) -> torch.Tensor:
    class_count = classes.shape[-1] - 1
    class_index = classes[..., :class_count].argmax(dim=-1)
    score = mask_probability
    if existence is not None:
        score = score * existence[:, :, None, None]
    confidence, owner = score.max(dim=1)
    owner_class = class_index.gather(1, owner.flatten(1)).reshape(owner.shape)
    one_hot = torch.nn.functional.one_hot(
        owner_class,
        num_classes=class_count,
    ).permute(0, 3, 1, 2).to(mask_probability.dtype)
    occupied = (confidence >= 0.5)[:, None]
    probability = one_hot * occupied
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(1.0e-6)


def _readout_rows(
    labels: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy,
) -> tuple[dict[str, float], np.ndarray]:
    observed = []
    counts = []
    for index in range(target.shape[0]):
        suite = room_metric_suite(
            labels[index].cpu().numpy(),
            target[index].cpu().numpy(),
            footprint[index, 0].cpu().numpy(),
            taxonomy,
        )
        instance = suite["matched_instance"]
        observed.append(
            {
                "room_iou": suite["visible_semantic"]["room_macro_iou"],
                "instance_f1": instance["instance_f1_at_threshold"],
                "instance_precision": instance[
                    "instance_precision_at_threshold"
                ],
                "instance_recall": instance[
                    "instance_recall_at_threshold"
                ],
                "count_mae": instance["count_abs_error"],
                "type_accuracy": instance["matched_type_accuracy"],
                "geometric_iou": instance["matched_geometric_iou"],
            }
        )
        counts.append(
            (
                instance["predicted_instances"],
                instance["target_instances"],
            )
        )
    return (
        {
            key: float(np.mean([row[key] for row in observed]))
            for key in observed[0]
        },
        np.asarray(counts, dtype=np.float64),
    )


@torch.inference_mode()
def audit(args: argparse.Namespace) -> dict:
    if args.dual_control_raster and args.control_raster_root is None:
        raise ValueError("--dual-control-raster requires --control-raster-root")
    dataset_class = CubiCasaRasterControlDataset
    dataset_kwargs = {"image_mode": "native"}
    if args.control_raster_root is not None:
        dataset_class = (
            DualControlRasterDataset
            if args.dual_control_raster
            else PredictedControlRasterDataset
        )
        dataset_kwargs = {"control_root": args.control_raster_root}
    dataset = dataset_class(
        args.data_root,
        args.split,
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=None,
        **dataset_kwargs,
    )
    dataset.samples = dataset.samples[
        int(args.offset) : int(args.offset) + int(args.limit)
    ]
    taxonomy = dataset.taxonomy
    room_ids = torch.tensor(taxonomy.room_ids, device=args.device)
    model = load_raster_instance_control_model(
        args.checkpoint,
        taxonomy,
        torch.device(args.device),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    values: dict[str, list[dict[str, float]]] = {}
    count_values: dict[str, list[np.ndarray]] = {}
    for target, footprint, image, _floor_ids, _names in loader:
        target = target.to(args.device)
        footprint = footprint.to(args.device)
        image = image.to(args.device)
        footprint_4d = (
            footprint[:, None] if footprint.ndim == 3 else footprint
        )
        outputs = model(footprint, control_image=image)
        dense = outputs["dense_logits"].softmax(dim=1)
        fused = outputs["fused_logits"].softmax(dim=1)
        masks = outputs["mask_logits"].sigmoid()
        classes = outputs["class_logits"]
        existence = 1.0 - classes.softmax(dim=-1)[..., -1]
        probabilities = {
            "dense": dense,
            "blend25": 0.75 * dense + 0.25 * fused,
            "blend50": 0.50 * dense + 0.50 * fused,
            "blend75": 0.25 * dense + 0.75 * fused,
            "query": fused,
            "existence_mix": _class_mass_probability(
                masks,
                classes,
                existence=existence,
            ),
            "existence_mix_sq": _class_mass_probability(
                masks,
                classes,
                existence=existence,
                power=2.0,
            ),
            "existence_winner": _winner_probability(
                masks,
                classes,
                existence=existence,
            ),
            "legacy": fused_room_mask_logits(
                outputs,
                len(taxonomy.room_ids),
                fusion_mode="legacy",
            ).softmax(dim=1),
            "query_mixture": fused_room_mask_logits(
                outputs,
                len(taxonomy.room_ids),
                fusion_mode="query_mixture",
            ).softmax(dim=1),
            "hard_query": fused_room_mask_logits(
                outputs,
                len(taxonomy.room_ids),
                fusion_mode="hard_query",
            ).softmax(dim=1),
            "winner": fused_room_mask_logits(
                outputs,
                len(taxonomy.room_ids),
                fusion_mode="winner_query",
            ).softmax(dim=1),
        }
        for name, probability in probabilities.items():
            labels = (
                room_ids[probability.argmax(dim=1)]
                * footprint_4d[:, 0].long()
            )
            rows, counts = _readout_rows(
                labels,
                target,
                footprint_4d,
                taxonomy,
            )
            values.setdefault(name, []).append(rows)
            count_values.setdefault(name, []).append(counts)
    summary = {
        name: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        for name, rows in values.items()
    }
    for name, rows in count_values.items():
        counts = np.concatenate(rows, axis=0)
        summary[name]["predicted_count"] = float(counts[:, 0].mean())
        summary[name]["target_count"] = float(counts[:, 1].mean())
    return {
        "status": "complete",
        "scope": "zero-training query readout audit",
        "condition_mode": "predicted_control",
        "target_used_at_inference": False,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "offset": int(args.offset),
        "cases": len(dataset),
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
