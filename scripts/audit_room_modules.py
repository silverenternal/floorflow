#!/usr/bin/env python3
"""Run same-case counterfactual attribution across room modules."""

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
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.representation.polygon_targets import decode_polygon_loops
from floorflow.evaluation.room_module_attribution import (
    labels_from_room_logits,
    room_support_metrics,
)
from floorflow.training.runtime_acceleration import RuntimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retrieval-prior-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    runtime = RuntimeConfig(precision="bf16")
    model, config, taxonomy, _state = load_checkpoint_model(args.checkpoint, device)
    dataset = CubiCasaFloorplanDataset(
        REPO_ROOT / ".data/cubicasa5k",
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2",
        augment=False,
        limit=args.limit,
        write_cache=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **runtime.dataloader_options(device, args.workers),
    )
    memory = ContourPartitionMemory.from_cache(
        REPO_ROOT / "outputs/cache/cubicasa5k_full_v2",
        top_k=1,
        metric="cosine",
    )
    memory.precompute_partitions(taxonomy, workers=args.workers)
    totals: dict[str, list[dict[str, float]]] = {}
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        prior = memory.retrieve_logits(
            footprint.detach().cpu().numpy(),
            taxonomy,
            device,
            top_k=1,
        )
        zero = torch.zeros(
            target.shape[0],
            model.target_channels,
            *target.shape[-2:],
            device=device,
        )
        with runtime.autocast(device):
            _, aux = model.forward_with_aux(
                zero,
                torch.zeros(target.shape[0], device=device),
                footprint,
                floor_ids=floor_ids,
            )
            _, aux_prior = model.forward_with_aux(
                zero,
                torch.zeros(target.shape[0], device=device),
                footprint,
                floor_ids=floor_ids,
                layout_prior_logits=prior,
            )
            boundary_enabled = model.hierarchical_rooms.boundary_enabled
            model.hierarchical_rooms.boundary_enabled = False
            _, aux_no_boundary = model.forward_with_aux(
                zero,
                torch.zeros(target.shape[0], device=device),
                footprint,
                floor_ids=floor_ids,
            )
            model.hierarchical_rooms.boundary_enabled = boundary_enabled
        variants = {
            "partition": labels_from_room_logits(
                aux["room_partition_logits"],
                footprint,
                taxonomy,
            ),
            "partition_with_retrieval": labels_from_room_logits(
                aux_prior["room_partition_logits"]
                + args.retrieval_prior_weight * prior,
                footprint,
                taxonomy,
            ),
        }
        if aux.get("semantic_logits") is not None:
            semantic_logits = torch.cat(
                [torch.zeros_like(aux["semantic_logits"][:, :1]), aux["semantic_logits"]],
                dim=1,
            ).argmax(dim=1)
            semantic_room_ids = torch.tensor(
                taxonomy.room_ids,
                device=footprint.device,
            )
            variants["semantic"] = torch.where(
                torch.isin(semantic_logits, semantic_room_ids),
                semantic_logits,
                torch.zeros_like(semantic_logits),
            ) * footprint.long()
        hierarchy = aux.get("hierarchical_rooms")
        if hierarchy is not None:
            variants["coarse"] = labels_from_room_logits(
                torch.nn.functional.interpolate(
                    hierarchy["coarse_room_logits"],
                    size=footprint.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
                footprint,
                taxonomy,
            )
            variants["refined"] = labels_from_room_logits(
                torch.nn.functional.interpolate(
                    hierarchy["refined_room_logits"],
                    size=footprint.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
                footprint,
                taxonomy,
            )
            hierarchy_prior = aux_prior.get("hierarchical_rooms")
            if hierarchy_prior is not None:
                variants["coarse_with_retrieval"] = labels_from_room_logits(
                    torch.nn.functional.interpolate(
                        hierarchy_prior["coarse_room_logits"],
                        size=footprint.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    footprint,
                    taxonomy,
                )
                variants["refined_with_retrieval"] = labels_from_room_logits(
                    torch.nn.functional.interpolate(
                        hierarchy_prior["refined_room_logits"],
                        size=footprint.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    + args.retrieval_prior_weight * prior,
                    footprint,
                    taxonomy,
                )
            hierarchy_no_boundary = aux_no_boundary.get("hierarchical_rooms")
            if hierarchy_no_boundary is not None:
                variants["refined_without_boundary_geometry"] = labels_from_room_logits(
                    torch.nn.functional.interpolate(
                        hierarchy_no_boundary["refined_room_logits"],
                        size=footprint.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    footprint,
                    taxonomy,
                )
        if aux.get("polygon_loops") is not None:
            variants["polygon"] = decode_polygon_loops(
                aux["polygon_loops"],
                taxonomy,
                height=footprint.shape[-2],
                width=footprint.shape[-1],
            )
        for name, prediction in variants.items():
            for index in range(prediction.shape[0]):
                totals.setdefault(name, []).append(
                    room_support_metrics(
                        prediction[index : index + 1],
                        target[index : index + 1],
                        footprint[index : index + 1],
                        taxonomy,
                    )
                )
    summary = {
        name: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        for name, rows in totals.items()
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "case_count": len(next(iter(totals.values()))),
        "variants": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
