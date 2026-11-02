#!/usr/bin/env python3
"""Audit per-class failures and materialize the worst contour-conditioned cases."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.flow.core import decode_floorplans, sample_floorplans
from floorflow.geometry.core import compose_known_exterior
from floorflow.inference.structured_decode import decode_structured_floorplan
from floorflow.data.svg import rasterize_cubicasa_svg
from floorflow.evaluation.visualize import save_comparison_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--raster-check-cases", type=int, default=10)
    parser.add_argument("--save-worst", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _metric_groups(rows: list[dict]) -> list[dict[str, float | str]]:
    output = []
    prefixes = (
        ("room:", "iou_room:"),
        ("opening:", "iou_door:"),
        ("opening:", "iou_window:"),
        ("furniture:", "iou_furniture:"),
    )
    keys = sorted({key for row in rows for key in row if key.startswith(tuple(prefix[1] for prefix in prefixes))})
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row and row[key] is not None])
        if values.size == 0:
            continue
        family = next(prefix[0].rstrip(":") for prefix in prefixes if key.startswith(prefix[1]))
        output.append(
            {
                "family": family,
                "class": key.split(":", 1)[1],
                "present_cases": int(values.size),
                "mean_iou": float(values.mean()),
                "median_iou": float(np.median(values)),
                "p10_iou": float(np.quantile(values, 0.10)),
                "zero_fraction": float((values == 0.0).mean()),
            }
        )
    return sorted(output, key=lambda row: (row["family"], row["mean_iou"], row["class"]))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _raster_consistency(dataset: CubiCasaFloorplanDataset, cases: int) -> dict[str, float]:
    matched = []
    wall_ious = []
    wall_id = dataset.taxonomy.class_id("wall")
    for index, sample in enumerate(dataset.samples[: max(0, cases)]):
        cached_target, cached_footprint, _floor_id, _name = dataset[index]
        fresh_target, fresh_footprint = rasterize_cubicasa_svg(
            sample.svg_path,
            image_size=dataset.image_size,
            floor_id=sample.floor_id,
            taxonomy=dataset.taxonomy,
        )
        cached_target = cached_target.numpy()
        cached_footprint = cached_footprint.numpy().astype(bool)
        matched.append(
            float(
                np.array_equal(cached_target, fresh_target)
                and np.array_equal(cached_footprint, fresh_footprint.astype(bool))
            )
        )
        wall = cached_target == wall_id
        wall_ious.append(float((wall & cached_footprint).sum() / max(1, wall.sum())))
    return {
        "cases_checked": float(len(matched)),
        "exact_cache_match": float(np.mean(matched)) if matched else 0.0,
        "wall_inside_footprint": float(np.mean(wall_ious)) if wall_ious else 0.0,
    }


def _predict(
    model,
    target: torch.Tensor,
    footprint: torch.Tensor,
    floor_id: int,
    taxonomy,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    floor_ids = torch.tensor([floor_id], device=device)
    if config.get("objective_mode") == "condition_structured":
        return decode_structured_floorplan(
            model,
            target,
            footprint,
            taxonomy,
            floor_ids=floor_ids,
            room_source="partition",
            object_source="heads",
            opening_threshold=0.5,
            furniture_threshold=0.5,
            object_score_scale="probability",
            structural_gate=True,
        )
    sampled, _ = sample_floorplans(
        model,
        footprint,
        steps=int(config.get("steps", 2)),
        heun=True,
        floor_ids=floor_ids,
        semantic_decode_weight=1.0,
        objective_mode=str(config.get("objective_mode", "velocity")),
    )
    return compose_known_exterior(
        decode_floorplans(sampled, smooth_kernel=1),
        footprint,
        opening_ids=taxonomy.opening_ids,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.per_case.read_text(encoding="utf-8"))
    summary = json.loads(args.metrics.read_text(encoding="utf-8"))
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(summary.get("image_size", 256)),
        cache_root=args.cache_root,
        augment=False,
    )
    if len(dataset) != len(rows):
        raise ValueError(f"dataset has {len(dataset)} cases but per-case report has {len(rows)}")
    for index, row in enumerate(rows):
        sample = dataset.samples[index]
        row["name"] = f"{sample.relative_case}__{sample.floor_id}"
    ranked = sorted(rows, key=lambda row: (float(row["room_macro_iou"]), float(row.get("room_component_gap", 0.0))))
    class_rows = _metric_groups(rows)
    report = {
        "summary": {key: summary.get(key) for key in sorted(summary) if key.endswith("_mean")},
        "raster_consistency": _raster_consistency(dataset, args.raster_check_cases),
        "diagnosis": {
            "coverage_high_but_iou_low": bool(
                float(summary.get("room_area_coverage_mean", 0.0)) >= 0.90
                and float(summary.get("room_macro_iou_mean", 0.0)) < 0.20
            ),
            "dominant_rooms_positive": any(
                row["class"] in {"Bedroom", "LivingRoom", "Storage"} and float(row["mean_iou"]) >= 0.20
                for row in class_rows
            ),
            "rare_rooms_zero": any(
                row["family"] == "room" and float(row["zero_fraction"]) >= 0.75 for row in class_rows
            ),
            "fragmentation": float(summary.get("room_component_gap_mean", 0.0)),
        },
        "per_class": class_rows,
        "worst_cases": ranked[: args.top_k],
    }
    (args.output_dir / "failure_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.output_dir / "per_class_iou.csv", class_rows)
    _write_csv(args.output_dir / "worst_cases.csv", report["worst_cases"])
    if args.save_worst:
        if args.checkpoint is None:
            raise ValueError("--save-worst requires --checkpoint")
        device = torch.device(args.device)
        model, config, taxonomy, _ = load_checkpoint_model(args.checkpoint, device)
        for rank, row in enumerate(ranked[: args.top_k], start=1):
            index = next(i for i, item in enumerate(rows) if item["name"] == row["name"])
            target, footprint, floor_id, _name = dataset[index]
            target = target[None].to(device)
            footprint = footprint[None].to(device)
            prediction = _predict(model, target, footprint, floor_id, taxonomy, config, device)
            stem = row["name"].replace("/", "__").replace("\\", "__")
            save_comparison_panel(
                args.output_dir / "worst_examples" / f"{rank:02d}_{stem}.png",
                footprint[0].cpu().numpy().astype(np.uint8),
                target[0].cpu().numpy(),
                prediction[0].cpu().numpy(),
                title=f"{row['name']} roomIoU={row['room_macro_iou']:.4f}",
                taxonomy=taxonomy,
            )
    print(json.dumps(report["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
