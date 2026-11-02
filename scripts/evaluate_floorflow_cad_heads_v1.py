"""Evaluate learned wall and opening query heads without canonical fallbacks."""

from __future__ import annotations

import argparse
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
from floorflow.training.unified_plan_generator_training import (
    match_candidate_queries,
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
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-cases", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--wall-tolerance", type=float, default=0.08)
    parser.add_argument("--opening-tolerance", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _wall_errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
    pairs: tuple[np.ndarray, np.ndarray],
) -> torch.Tensor:
    rows, columns = pairs
    if not rows.size:
        return predicted.new_zeros(0)
    left = predicted[rows]
    right = target[torch.from_numpy(columns).to(target.device)]
    direct = (left - right).abs().mean(dim=(1, 2))
    reverse = (left - right[:, [1, 0]]).abs().mean(dim=(1, 2))
    pred_length = (left[:, 1] - left[:, 0]).norm(dim=-1)
    target_length = (right[:, 1] - right[:, 0]).norm(dim=-1)
    return torch.minimum(direct, reverse) + 0.5 * (
        pred_length - target_length
    ).abs()


def _opening_errors(
    centers: torch.Tensor,
    widths: torch.Tensor,
    target_centers: torch.Tensor,
    target_widths: torch.Tensor,
) -> torch.Tensor:
    count = min(len(centers), int(target_centers.shape[0]))
    if count == 0:
        return centers.new_zeros(0)
    return (centers[:count] - target_centers[:count]).norm(dim=-1) + 0.5 * (
        widths[:count] - target_widths[:count]
    ).abs()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and str(device) != "cuda:0":
        raise ValueError("RPLAN evaluation is restricted to cuda:0")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    model = UnifiedFloorplanGenerator(
        taxonomy,
        UnifiedGeneratorConfig(**payload["config"]["model"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    limits = RPlanTensorLimits(
        **{key: int(value) for key, value in payload["config"]["tensors"].items()}
    )
    rows = load_rplan_rows(
        args.data,
        offset=int(args.val_offset),
        limit=int(args.val_cases),
    )
    dataset = RPlanBoundaryDataset(
        args.data,
        taxonomy,
        rows=rows,
        limits=limits,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=collate_rplan_items,
        pin_memory=device.type == "cuda",
    )
    totals = {
        "wall_predicted": 0.0,
        "wall_target": 0.0,
        "wall_true_positive": 0.0,
        "wall_endpoint_error": 0.0,
        "opening_predicted": 0.0,
        "opening_target": 0.0,
        "opening_true_positive": 0.0,
        "opening_localization_error": 0.0,
        "cases": 0.0,
    }
    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            output = model(
                batch["boundary_mask"],
                batch["boundary_valid"],
                batch["boundary_points"],
                batch["entrance_points"],
                batch["entrance_valid"],
            )
            candidate_valid = torch.ones_like(
                output["wall_exists"],
                dtype=torch.bool,
            )
            pairs, _positive = match_candidate_queries(
                output["wall_endpoints"],
                batch["wall_endpoints"],
                batch["wall_valid"],
                candidate_valid,
            )
            for index in range(len(batch["case_name"])):
                predicted_active = (
                    output["wall_exists"][index].sigmoid() >= args.threshold
                )
                pairs_rows, _pairs_columns = pairs[index]
                errors = _wall_errors(
                    output["wall_endpoints"][index],
                    batch["wall_endpoints"][index],
                    pairs[index],
                )
                correct = (
                    predicted_active[torch.from_numpy(pairs_rows).to(device)]
                    & (errors <= float(args.wall_tolerance))
                )
                target_count = int(batch["wall_valid"][index].sum())
                totals["wall_predicted"] += float(predicted_active.sum())
                totals["wall_target"] += target_count
                totals["wall_true_positive"] += float(correct.sum())
                totals["wall_endpoint_error"] += float(errors.mean()) if len(errors) else 0.0
                opening_active = (
                    output["opening_exists"][index].sigmoid() >= args.threshold
                )
                target_opening = int(batch["opening_valid"][index].sum())
                active_center_error = _opening_errors(
                    output["opening_centers"][index, opening_active],
                    output["opening_widths"][index, opening_active],
                    batch["opening_center"][index, :target_opening],
                    batch["opening_width"][index, :target_opening],
                )
                totals["opening_predicted"] += float(opening_active.sum())
                totals["opening_target"] += target_opening
                totals["opening_true_positive"] += float(
                    (active_center_error <= float(args.opening_tolerance)).sum()
                )
                totals["opening_localization_error"] += (
                    float(active_center_error.mean())
                    if len(active_center_error)
                    else 0.0
                )
                totals["cases"] += 1.0
    cases = max(1.0, totals["cases"])
    wall_precision = totals["wall_true_positive"] / max(
        1.0,
        totals["wall_predicted"],
    )
    wall_recall = totals["wall_true_positive"] / max(1.0, totals["wall_target"])
    opening_precision = totals["opening_true_positive"] / max(
        1.0,
        totals["opening_predicted"],
    )
    opening_recall = totals["opening_true_positive"] / max(
        1.0,
        totals["opening_target"],
    )
    report = {
        "schema": "floorflow-cad-head-evaluation-v1",
        "checkpoint": str(args.checkpoint),
        "validation_file": str(args.data),
        "validation_offset": int(args.val_offset),
        "requested_cases": len(rows),
        "test_opened": False,
        "threshold": float(args.threshold),
        "metrics": {
            "wall_count_predicted": totals["wall_predicted"] / cases,
            "wall_count_target": totals["wall_target"] / cases,
            "wall_precision": wall_precision,
            "wall_recall": wall_recall,
            "wall_f1": 2.0 * wall_precision * wall_recall / max(
                1.0e-8,
                wall_precision + wall_recall,
            ),
            "wall_endpoint_error": totals["wall_endpoint_error"] / cases,
            "opening_count_predicted": totals["opening_predicted"] / cases,
            "opening_count_target": totals["opening_target"] / cases,
            "opening_precision": opening_precision,
            "opening_recall": opening_recall,
            "opening_f1": 2.0 * opening_precision * opening_recall / max(
                1.0e-8,
                opening_precision + opening_recall,
            ),
            "opening_localization_error": (
                totals["opening_localization_error"] / cases
            ),
        },
        "claim_boundary": (
            "Raw learned CAD-query diagnostics; this does not consume the "
            "canonical room-partition fallback."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
