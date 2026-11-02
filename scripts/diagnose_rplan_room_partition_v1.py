"""Audit whether room queries form a mutually exclusive footprint partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    collate_rplan_items,
    load_rplan_rows,
)
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.model.unified_floorplan_generator import (
    UnifiedFloorplanGenerator,
    UnifiedGeneratorConfig,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT / ".data/external/CE2EPlan/dataset/dataset_json/data_val.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--validation-offset", type=int, default=256)
    parser.add_argument("--validation-cases", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def _row_metrics(
    independent_probability: torch.Tensor,
    joint_probability: torch.Tensor | None,
    footprint: torch.Tensor,
    selected: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    selected_probability = independent_probability[selected]
    inside = footprint.bool()
    inside_count = inside.float().sum().clamp_min(1.0)
    independent = selected_probability >= float(threshold)
    active = independent.sum(dim=0)
    partition_probability = (
        joint_probability[selected]
        if joint_probability is not None
        else selected_probability
    )
    owner_probability, _owner = partition_probability.max(dim=0)
    entropy = -(
        partition_probability.clamp_min(1.0e-8).log()
        * partition_probability
    ).sum(dim=0)
    entropy = entropy / np.log(max(2, int(selected.numel())))
    independent_overlap = (
        (inside & (active > 1)).float().sum() / inside_count
    )
    independent_gap = (inside & (active == 0)).float().sum() / inside_count
    active_count = (active.float() * inside).sum() / inside_count
    return {
        "independent_threshold_overlap_fraction": float(
            independent_overlap
        ),
        "independent_threshold_gap_fraction": float(independent_gap),
        "independent_active_masks_per_pixel": float(active_count),
        "joint_partition_overlap_fraction": (
            0.0 if joint_probability is not None else float(independent_overlap)
        ),
        "joint_partition_gap_fraction": (
            0.0 if joint_probability is not None else float(independent_gap)
        ),
        "owner_confidence_mean": float(
            (owner_probability * inside).sum() / inside_count
        ),
        "owner_entropy_mean": float(
            (entropy * inside).sum() / inside_count
        ),
        "low_confidence_fraction": float(
            (
                inside
                & (owner_probability < float(threshold))
            ).float().sum()
            / inside_count
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and str(device) != "cuda:0":
        raise ValueError("RPLAN diagnostics are restricted to cuda:0")
    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    model = UnifiedFloorplanGenerator(
        taxonomy,
        UnifiedGeneratorConfig(**payload["config"]["model"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    limits = RPlanTensorLimits(
        **{
            key: int(value)
            for key, value in payload["config"]["tensors"].items()
        }
    )
    rows = load_rplan_rows(
        args.data,
        offset=int(args.validation_offset),
        limit=int(args.validation_cases),
    )
    loader = DataLoader(
        RPlanBoundaryDataset(
            args.data,
            taxonomy,
            rows=rows,
            limits=limits,
        ),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=collate_rplan_items,
        pin_memory=device.type == "cuda",
    )
    per_case = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            output = model(
                batch["boundary_mask"],
                batch["boundary_valid"],
                batch["boundary_points"],
                batch["entrance_points"],
                batch["entrance_valid"],
            )
            joint_probability = output.get(
                "room_instance_partition_probabilities"
            )
            independent_probability = output[
                "room_instance_mask_logits"
            ].sigmoid()
            independent_probability = torch.nn.functional.interpolate(
                independent_probability,
                size=batch["boundary_mask"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if joint_probability is not None:
                joint_probability = torch.nn.functional.interpolate(
                    joint_probability,
                    size=batch["boundary_mask"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            footprint = batch["boundary_mask"][:, 0]
            count = output["room_count_logits"].argmax(dim=-1).clamp(
                1,
                output["room_exists"].shape[1],
            )
            for index, case_name in enumerate(batch["case_name"]):
                selected = torch.topk(
                    output["room_exists"][index],
                    k=int(count[index]),
                ).indices
                metrics = _row_metrics(
                    independent_probability[index],
                    (
                        joint_probability[index]
                        if joint_probability is not None
                        else None
                    ),
                    footprint[index],
                    selected,
                    args.threshold,
                )
                per_case.append({"case_name": case_name, **metrics})
    keys = tuple(per_case[0])
    summary = {
        key: float(np.mean([row[key] for row in per_case]))
        for key in keys
        if key != "case_name"
    }
    summary["cases"] = len(per_case)
    summary["independent_threshold_overlap_nonzero_case_rate"] = float(
        np.mean(
            [
                row["independent_threshold_overlap_fraction"] > 0.0
                for row in per_case
            ]
        )
    )
    summary["independent_threshold_gap_gt_5pct_case_rate"] = float(
        np.mean(
            [
                row["independent_threshold_gap_fraction"] > 0.05
                for row in per_case
            ]
        )
    )
    output = {
        "schema": "floorflow-rplan-room-partition-diagnostic-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            args.checkpoint.read_bytes()
        ).hexdigest(),
        "data": str(args.data),
        "validation_offset": int(args.validation_offset),
        "validation_cases": int(args.validation_cases),
        "device": str(device),
        "threshold": float(args.threshold),
        "joint_partition_available": bool(
            payload["config"]["model"].get(
                "enable_partition_consistent_queries",
                False,
            )
        ),
        "summary": summary,
        "rows": per_case,
        "test_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
