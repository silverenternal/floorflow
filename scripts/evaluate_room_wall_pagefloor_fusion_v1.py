#!/usr/bin/env python3
"""Evaluate wall-gated seams on the frozen page-floor room readout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_dense_room_wall_fusion_v1 import _room_metrics, _summary

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.generation_plausibility import (
    generation_plausibility,
)
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.geometry.core import compose_known_exterior
from floorflow.heads.raster_control_bank import (
    load_raster_control_bank,
    load_raster_instance_control_model,
)
from floorflow.inference.dense_room_wall_fusion import fuse_dense_room_walls
from floorflow.training.raster_instance_evaluation import resolve_query_blend

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wall-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/train/raster_full_cad_bank_frozen_room_screen_v4/best.pt"
        ),
    )
    parser.add_argument("--query-checkpoint", type=Path, required=True)
    parser.add_argument("--query-weight", type=float, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / ".data/cubicasa5k",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--boundary-width", type=int, default=7)
    parser.add_argument(
        "--exact-structural",
        action="store_true",
        help="Run Shapely/fpeval checks; otherwise use cheap raster proxies.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _wall_scores(
    prediction: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    wall_id = taxonomy.class_id("wall")
    predicted = prediction == wall_id
    expected = target == wall_id
    intersection = float((predicted & expected).sum())
    precision = intersection / max(1.0, float(predicted.sum()))
    recall = intersection / max(1.0, float(expected.sum()))
    return {
        "wall_precision": precision,
        "wall_recall": recall,
        "wall_f1": (
            2.0 * precision * recall / max(1.0e-8, precision + recall)
        ),
    }


def _cheap_structural(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    structural = (labels == taxonomy.class_id("wall")) | np.isin(
        labels,
        np.asarray(taxonomy.opening_ids),
    )
    components, count = label(structural)
    sizes = np.bincount(components.ravel())[1:] if count else np.zeros(0)
    continuity = (
        float(sizes.max() / max(1, int(structural.sum())))
        if sizes.size
        else 1.0
    )
    outside = float(((labels != 0) & ~footprint.astype(bool)).sum())
    return {
        "wall_component_continuity": continuity,
        "outside_footprint_fraction": outside
        / max(1.0, float(footprint.sum())),
    }


def _arms() -> list[tuple[str, str, float, bool]]:
    arms = [
        ("dense_boundary", "dense", 0.5, False),
        ("query_boundary", "query", 0.5, False),
        ("query_owner_boundary", "query", 0.5, True),
    ]
    arms.extend(
        (
            f"dense_wall_gated_{threshold:.1f}",
            "dense",
            threshold,
            False,
        )
        for threshold in THRESHOLDS
    )
    arms.extend(
        (
            f"query_wall_gated_{threshold:.1f}",
            "query",
            threshold,
            False,
        )
        for threshold in THRESHOLDS
    )
    arms.extend(
        (
            f"query_owner_wall_gated_{threshold:.1f}",
            "query",
            threshold,
            True,
        )
        for threshold in THRESHOLDS
    )
    return arms


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(
        args.wall_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    wall_model = load_raster_control_bank(
        args.wall_checkpoint,
        taxonomy,
        device,
    )
    query_model = load_raster_instance_control_model(
        args.query_checkpoint,
        taxonomy,
        device,
    )
    query_weight, query_weight_source = resolve_query_blend(
        args.query_checkpoint,
        args.query_weight,
    )
    dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
        limit=None,
        image_mode="floor",
    )
    subset = Subset(
        dataset,
        range(int(args.offset), int(args.offset) + int(args.limit)),
    )
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    rows = []
    for target, footprint, image, _floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        wall_output = wall_model(footprint, control_image=image)
        query_output = query_model(footprint, control_image=image)
        dense_probability = wall_output["room_logits"].softmax(dim=1)
        query_probability = query_output["fused_logits"].softmax(dim=1)
        probability = (
            (1.0 - query_weight) * dense_probability
            + query_weight * query_probability
        )
        labels = {
            "dense": (
                room_ids[dense_probability.argmax(dim=1)] * footprint.long()
            ),
            "query": room_ids[probability.argmax(dim=1)] * footprint.long(),
        }
        wall_probability = wall_output["wall_logits"].sigmoid()
        owner_boundary = F.interpolate(
            query_output["owner_boundary"][:, None].float(),
            size=tuple(footprint.shape[-2:]),
            mode="nearest",
        )[:, 0].bool()
        for arm, source, threshold, use_owner in _arms():
            mode = (
                "owner_boundary"
                if use_owner and arm.endswith("_boundary")
                else "owner_wall_gated"
                if use_owner
                else "boundary_wall"
                if arm.endswith("_boundary")
                else "wall_gated"
            )
            for index, name in enumerate(names):
                current_target = target[index].cpu().numpy()
                current_footprint = footprint[index].cpu().numpy()
                fused = fuse_dense_room_walls(
                    labels[source][index].cpu().numpy(),
                    current_footprint,
                    wall_probability[index].cpu().numpy(),
                    taxonomy,
                    mode=mode,
                    wall_threshold=threshold,
                    owner_boundary=(
                        owner_boundary[index].cpu().numpy()
                        if use_owner
                        else None
                    ),
                )
                composed = compose_known_exterior(
                    torch.from_numpy(fused.labels)[None],
                    torch.from_numpy(current_footprint)[None],
                    boundary_width=int(args.boundary_width),
                    opening_ids=taxonomy.opening_ids,
                )[0].numpy()
                if args.exact_structural:
                    structural = generation_plausibility(
                        composed,
                        current_footprint,
                        taxonomy,
                        explicit_rooms=True,
                    )
                    structural_metrics = {
                        "validity": float(structural["validity"]),
                        "functional_watertight": float(
                            structural["functional_watertight"]
                        ),
                        "wall_continuity_rate": float(
                            structural["wall_continuity_rate"]
                        ),
                        "outside_footprint_fraction": float(
                            structural["out_of_boundary_ratio"]
                        ),
                    }
                else:
                    structural_metrics = _cheap_structural(
                        composed,
                        current_footprint,
                        taxonomy,
                    )
                rows.append(
                    {
                        "case": str(name),
                        "arm": arm,
                        **_room_metrics(
                            composed,
                            current_target,
                            current_footprint,
                            taxonomy,
                        ),
                        **_wall_scores(
                            composed,
                            current_target,
                            taxonomy,
                        ),
                        **structural_metrics,
                    }
                )
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["arm"], []).append(row)
    keys = tuple(
        key for key in rows[0] if key not in {"case", "arm"}
    )
    summary = {
        arm: {
            key: _summary([row[key] for row in values])
            for key in keys
        }
        for arm, values in grouped.items()
    }
    control = grouped["query_boundary"]
    paired = {
        arm: {
            key: paired_delta(
                values,
                control,
                key,
                iterations=5000,
                seed=17,
                lower_is_better=key
                in {"count_abs_error", "outside_footprint_fraction"},
            )
            for key in keys
        }
        for arm, values in grouped.items()
    }
    report = {
        "schema_version": "room-wall-pagefloor-fusion-v1",
        "protocol": "pagefloor_native_cubicasa_256",
        "query_checkpoint": str(args.query_checkpoint),
        "query_weight": query_weight,
        "query_weight_source": query_weight_source,
        "wall_checkpoint": str(args.wall_checkpoint),
        "target_used_at_inference": False,
        "structural_mode": (
            "exact_fpeval" if args.exact_structural else "cheap_raster"
        ),
        "offset": args.offset,
        "cases": args.limit,
        "summary": summary,
        "paired_vs_query_boundary": paired,
        "case_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "query_weight": query_weight,
                "cases": args.limit,
                "summary": {
                    arm: {
                        key: values[key]["mean"]
                        for key in (
                            "room_macro_iou",
                            "instance_f1",
                            "wall_f1",
                            "wall_continuity_rate",
                            "functional_watertight",
                        )
                    }
                    for arm, values in summary.items()
                },
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
