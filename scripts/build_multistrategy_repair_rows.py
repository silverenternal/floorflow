#!/usr/bin/env python3
"""Build target-free repair rows from every guarded repair strategy."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics
from floorflow.geometry.core import compose_known_exterior
from floorflow.inference.core import semantic_endpoint_labels
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.inference.repair_acceptance import guarded_summary_decision
from floorflow.inference.repair_features import repair_decision_features
from floorflow.inference.repair_registry import build_floorplan_repair_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        args.device,
    )
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    rows = []
    with torch.no_grad():
        for target, footprint, floor_ids, names in loader:
            target = target.to(args.device)
            footprint = footprint.to(args.device)
            labels = semantic_endpoint_labels(
                model,
                target,
                footprint,
                floor_ids.to(args.device),
            )
            labels = compose_known_exterior(
                labels,
                footprint,
                opening_ids=taxonomy.opening_ids,
            )
            for index, name in enumerate(names):
                raw = labels[index].cpu().numpy()
                expected = target[index].cpu().numpy()
                footprint_np = footprint[index].cpu().numpy().astype("uint8")
                registry = build_floorplan_repair_registry(
                    torch.from_numpy(footprint_np),
                    taxonomy,
                )
                raw_summary = fpeval_raster_metrics(
                    raw,
                    expected,
                    taxonomy,
                    tolerance_fractions=(0.05,),
                )["summary"]["0.05"]
                raw_legacy = floorplan_metrics(
                    torch.from_numpy(raw)[None],
                    torch.from_numpy(expected)[None],
                    torch.from_numpy(footprint_np)[None],
                    taxonomy,
                )
                best = None
                for strategy_name, candidate, guarded in registry.candidates(
                    torch.from_numpy(raw)
                ):
                    if not guarded:
                        continue
                    candidate_np = candidate.detach().cpu().numpy()
                    candidate_summary = fpeval_raster_metrics(
                        candidate_np,
                        expected,
                        taxonomy,
                        tolerance_fractions=(0.05,),
                    )["summary"]["0.05"]
                    decision = guarded_summary_decision(
                        raw_summary,
                        candidate_summary,
                    )
                    if not decision.accepted:
                        continue
                    candidate_legacy = floorplan_metrics(
                        torch.from_numpy(candidate_np)[None],
                        torch.from_numpy(expected)[None],
                        torch.from_numpy(footprint_np)[None],
                        taxonomy,
                    )
                    if (
                        candidate_legacy["room_macro_iou"]
                        < raw_legacy["room_macro_iou"] - 1.0e-8
                        or candidate_legacy["furniture_iou"]
                        < raw_legacy["furniture_iou"] - 1.0e-8
                    ):
                        continue
                    value = candidate_summary["edit_cost"]
                    if best is None or value < best["edit_cost"]:
                        best = {
                            "strategy": strategy_name,
                            "candidate": candidate_np,
                            "edit_cost": value,
                            "summary": candidate_summary,
                            "deltas": decision.deltas,
                            "furniture_iou": candidate_legacy["furniture_iou"],
                            "room_macro_iou": candidate_legacy["room_macro_iou"],
                        }
                selected = raw if best is None else best["candidate"]
                selected_summary = raw_summary if best is None else best["summary"]
                rows.append(
                    {
                        "case": name,
                        **repair_decision_features(
                            raw,
                            selected,
                            footprint_np,
                            taxonomy,
                        ),
                        "accepted": int(best is not None),
                        "selected_strategy": (
                            "none" if best is None else best["strategy"]
                        ),
                        "delta_edit_cost": (
                            selected_summary["edit_cost"]
                            - raw_summary["edit_cost"]
                        ),
                        "delta_room_f1": (
                            selected_summary["room_f1"]
                            - raw_summary["room_f1"]
                        ),
                        "delta_wall_f1": (
                            selected_summary["wall_f1"]
                            - raw_summary["wall_f1"]
                        ),
                        "delta_furniture_iou": (
                            0.0
                            if best is None
                            else best["furniture_iou"]
                            - raw_legacy["furniture_iou"]
                        ),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "accepted": sum(row["accepted"] for row in rows),
                "acceptance_rate": sum(row["accepted"] for row in rows)
                / len(rows),
                "strategies": {
                    name: sum(row["selected_strategy"] == name for row in rows)
                    for name in sorted({row["selected_strategy"] for row in rows})
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
