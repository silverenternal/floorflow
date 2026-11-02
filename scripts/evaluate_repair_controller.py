#!/usr/bin/env python3
"""Evaluate a target-free repair accept/reject controller on frozen test."""

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
from floorflow.geometry.core import compose_known_exterior
from floorflow.inference.core import semantic_endpoint_labels
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.inference.repair_features import repair_decision_features
from floorflow.inference.repair_registry import RepairAcceptanceController
from floorflow.inference.room_repair import apply_floorplan_repairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/eval/repair_controller/full_test.json")
    args = parser.parse_args()
    model, config, taxonomy, _ = load_checkpoint_model(args.checkpoint, args.device)
    payload = json.loads(args.controller.read_text(encoding="utf-8"))
    controller = RepairAcceptanceController(
        len(payload["feature_names"]),
        centers=payload.get("centers"),
        scales=payload.get("scales"),
    )
    controller.weights = [float(value) for value in payload["weights"]]
    controller.threshold = float(payload.get("threshold", 0.0))
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    rows = []
    with torch.no_grad():
        for target, footprint, floor_ids, names in loader:
            target, footprint = target.to(args.device), footprint.to(args.device)
            labels = semantic_endpoint_labels(model, target, footprint, floor_ids.to(args.device))
            labels = compose_known_exterior(labels, footprint, opening_ids=taxonomy.opening_ids)
            for index, name in enumerate(names):
                raw = labels[index].cpu().numpy()
                footprint_np = footprint[index].cpu().numpy()
                repaired = apply_floorplan_repairs(raw, footprint_np.astype("uint8"), taxonomy)
                features = repair_decision_features(raw, repaired, footprint_np, taxonomy)
                accepted = controller.accept([features[key] for key in payload["feature_names"]])
                selected = repaired if accepted else raw
                raw_metrics = floorplan_metrics(
                    labels[index : index + 1],
                    target[index : index + 1],
                    footprint[index : index + 1],
                    taxonomy,
                )
                selected_metrics = floorplan_metrics(
                    torch.from_numpy(selected)[None].to(args.device),
                    target[index : index + 1],
                    footprint[index : index + 1],
                    taxonomy,
                )
                oracle_metrics = floorplan_metrics(
                    torch.from_numpy(repaired)[None].to(args.device),
                    target[index : index + 1],
                    footprint[index : index + 1],
                    taxonomy,
                )
                rows.append(
                    {
                        "case": name,
                        "accepted": accepted,
                        "raw": raw_metrics,
                        "selected": selected_metrics,
                        "repaired_oracle": oracle_metrics,
                    }
                )
    metrics = ("room_macro_iou", "room_area_coverage", "opening_f1", "furniture_iou")
    summary = {}
    for metric in metrics:
        raw = np.asarray([row["raw"][metric] for row in rows])
        selected = np.asarray([row["selected"][metric] for row in rows])
        summary[metric] = {
            "raw_mean": float(raw.mean()),
            "selected_mean": float(selected.mean()),
            "mean_delta": float((selected - raw).mean()),
            "nondegraded_fraction": float((selected >= raw - 1.0e-8).mean()),
        }
    report = {
        "cases": len(rows),
        "claim_role": "guarded_one_sample_repair",
        "best_of_n_used": False,
        "selection": "target-free_controller_accept_reject",
        "acceptance_rate": float(np.mean([row["accepted"] for row in rows])),
        "metrics": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cases", "acceptance_rate", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
