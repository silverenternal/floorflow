#!/usr/bin/env python3
"""Replay a frozen room decoder and write the HI-00 error decomposition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader, Subset

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_error_decomposition import (
    aggregate_decomposition,
    decompose_rooms,
)
from floorflow.training.runtime_acceleration import (
    PRECISIONS,
    RuntimeConfig,
    configure_torch_runtime,
)
from floorflow.inference.structured_decode import decode_structured_floorplan
from floorflow.data.taxonomy import FloorplanTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retrieval-prior-weight", type=float, default=0.75)
    parser.add_argument("--retrieval-top-k", type=int, default=1)
    parser.add_argument("--retrieval-metric", default="cosine")
    parser.add_argument("--instance-semantic-vote", action="store_true")
    parser.add_argument("--room-prior-weights-report", type=Path, default=None)
    parser.add_argument("--precision", choices=PRECISIONS, default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    runtime = RuntimeConfig(precision=args.precision)
    configure_torch_runtime(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    taxonomy = (
        FloorplanTaxonomy.from_json(checkpoint["taxonomy"])
        if checkpoint.get("taxonomy")
        else FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    )
    model, config, _taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        device,
        taxonomy=taxonomy,
    )
    model = runtime.prepare_model(model)
    image_size = int(config.get("image_size", 256))
    full_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=image_size,
        cache_root=args.cache_root,
        augment=False,
    )
    stop = (
        len(full_dataset)
        if args.limit is None
        else min(len(full_dataset), args.offset + args.limit)
    )
    dataset = Subset(full_dataset, range(args.offset, stop))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **runtime.dataloader_options(device, args.workers),
    )
    memory = None
    room_prior_weights = None
    if args.room_prior_weights_report is not None:
        room_prior_weights = torch.tensor(
            json.loads(
                args.room_prior_weights_report.read_text(encoding="utf-8")
            )["weights"],
            device=device,
            dtype=torch.float32,
        )
    if args.retrieval_prior_weight > 0:
        memory = ContourPartitionMemory.from_cache(
            args.cache_root,
            top_k=args.retrieval_top_k,
            metric=args.retrieval_metric,
        )

    support_cases = []
    composed_cases = []
    case_rows = []
    room_ids = torch.tensor(taxonomy.room_ids, device=device, dtype=torch.long)
    for target, footprint, floor_ids, names in loader:
        target = target.to(device, non_blocking=True)
        footprint = footprint.to(device, non_blocking=True)
        floor_ids = floor_ids.to(device, non_blocking=True)
        room_prior_logits = None
        if memory is not None:
            room_prior_logits = memory.retrieve_logits(
                footprint.detach().cpu().numpy(),
                taxonomy,
                device,
                top_k=args.retrieval_top_k,
            )
        with runtime.autocast(device):
            composed, room_support = decode_structured_floorplan(
                model,
                target,
                footprint,
                taxonomy,
                floor_ids=floor_ids,
                object_source="heads",
                room_source="partition",
                structural_gate=True,
                return_room_support=True,
                room_prior_logits=room_prior_logits,
                room_prior_weight=args.retrieval_prior_weight,
                room_prior_weights=room_prior_weights,
                instance_semantic_vote=args.instance_semantic_vote,
            )
        support = room_support.long() * footprint.long()
        composed_rooms = torch.where(
            torch.isin(composed, room_ids),
            composed,
            torch.zeros_like(composed),
        )
        for index, name in enumerate(names):
            support_case = decompose_rooms(
                support[index : index + 1].cpu(),
                target[index : index + 1].cpu(),
                footprint[index : index + 1].cpu(),
                taxonomy,
            )
            composed_case = decompose_rooms(
                composed_rooms[index : index + 1].cpu(),
                target[index : index + 1].cpu(),
                footprint[index : index + 1].cpu(),
                taxonomy,
            )
            support_cases.append(support_case)
            composed_cases.append(composed_case)
            case_rows.append(
                {
                    "name": name,
                    "support_macro_iou": support_case["strict_macro_iou"],
                    "composed_macro_iou": composed_case["strict_macro_iou"],
                    "support_occupancy_iou": support_case["occupancy"]["iou"],
                    "composed_occupancy_iou": composed_case["occupancy"]["iou"],
                    "support_fragmentation_gap": support_case["fragmentation_gap"],
                    "missing_class_count": support_case["missing_class_count"],
                    "spurious_class_count": support_case["spurious_class_count"],
                    "allocation": support_case["allocation"],
                }
            )
        print(f"decomposed {len(case_rows)}/{len(dataset)}", flush=True)

    support = aggregate_decomposition(support_cases)
    composed = aggregate_decomposition(composed_cases)
    report = {
        "claim_role": "diagnostic_only_room_iou_error_decomposition",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "case_count": len(case_rows),
        "configuration": {
            "room_source": "partition",
            "retrieval_prior_weight": args.retrieval_prior_weight,
            "retrieval_top_k": args.retrieval_top_k,
            "retrieval_metric": args.retrieval_metric,
            "boundary_tolerance": 2,
            "best_of_n_used": False,
        },
        "partition_support": support,
        "composed_single_raster": composed,
        "worst_cases": sorted(
            case_rows,
            key=lambda row: (
                row["support_macro_iou"],
                -row["support_fragmentation_gap"],
            ),
        )[:100],
        "sampling_policy": {
            "purpose": "single_sample",
            "candidates_per_condition": 1,
            "selection_rule": "none",
            "target_used_for_selection": False,
        },
    }
    _write_json(args.output, report)
    if args.cache_output is not None:
        _write_json(
            args.cache_output,
            {
                "cases": case_rows,
                "support": support,
                "composed": composed,
            },
        )
    print(
        json.dumps(
            {
                "cases": len(case_rows),
                "support_macro_iou": support["strict_macro_iou"],
                "support_dataset_macro_iou": support["dataset_class_macro_iou"],
                "support_occupancy_iou": support["occupancy"]["iou"],
                "support_allocation": support["allocation"],
                "composed_macro_iou": composed["strict_macro_iou"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
