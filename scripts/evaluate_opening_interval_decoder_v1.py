#!/usr/bin/env python3
"""Evaluate a trained opening detector with the calibrated wall-interval readout."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.predicted_control_reporting import numeric_row_summary
from floorflow.geometry.opening_readout import (
    detections_from_labels,
    opening_detection_metrics,
)
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    raster_labels_to_wall_plan,
)
from floorflow.inference.opening_candidate_ranker import (
    FEATURE_NAMES,
    load_opening_candidate_ranker,
    opening_candidate_features,
)
from floorflow.inference.opening_center_width_decoder import (
    read_center_width_image_detections,
    read_center_width_opening_detections,
)
from floorflow.inference.opening_heatmap_decoder import (
    learned_opening_detections,
    learned_opening_detections_from_maps,
    load_opening_heatmap_decoder,
    predict_opening_detection_maps,
    predict_opening_family_probabilities,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--min-area", type=int, default=7)
    parser.add_argument(
        "--independent-foreground",
        action="store_true",
        help="Decode presence from foreground while keeping family scores separate.",
    )
    parser.add_argument("--window-logit-bias", type=float, default=0.0)
    parser.add_argument(
        "--family-aggregation",
        choices=("center", "interval"),
        default="center",
    )
    parser.add_argument("--exterior-window-bias", type=float, default=0.0)
    parser.add_argument("--interval-radius", type=int, default=1)
    parser.add_argument("--score-smoothing", type=int, default=3)
    parser.add_argument("--max-gap-samples", type=int, default=2)
    parser.add_argument("--peak-relative-threshold", type=float, default=0.0)
    parser.add_argument("--candidate-ranker", type=Path, default=None)
    parser.add_argument("--candidate-ranker-threshold", type=float, default=None)
    parser.add_argument("--min-peak-distance", type=float, default=4.0)
    parser.add_argument(
        "--center-decode-mode",
        choices=("wall", "image"),
        default="wall",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/opening_interval_decoder_cal64_v1.json",
    )
    return parser.parse_args()


def pixel_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    threshold: float,
) -> dict[str, float]:
    expected = (
        np.isin(target, np.asarray(taxonomy.opening_ids))
        & footprint.astype(bool)
    )
    predicted = probability.max(axis=0) >= threshold
    predicted &= footprint.astype(bool)
    true_positive = int((predicted & expected).sum())
    precision = true_positive / max(1, int(predicted.sum()))
    recall = true_positive / max(1, int(expected.sum()))
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": 2.0 * precision * recall / max(1.0e-8, precision + recall),
    }


def _case_row(
    name: str,
    predicted,
    expected,
    probability_row: np.ndarray,
    target_row: np.ndarray,
    footprint_row: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    threshold: float,
) -> dict:
    vector = opening_detection_metrics(
        predicted,
        expected,
        tolerance=0.05 * float(target_row.shape[-1]),
    )
    return {
        "case": str(name),
        **pixel_metrics(
            probability_row,
            target_row,
            footprint_row,
            taxonomy,
            threshold,
        ),
        "vector_precision": float(vector["precision"]),
        "vector_recall": float(vector["recall"]),
        "vector_f1": float(vector["f1"]),
        "predicted_openings": len(predicted),
        "target_openings": len(expected),
        "matched_openings": int(vector["matched"]),
        "geometry_matched_openings": int(vector["geometry_matched"]),
        "family_consistent_openings": int(vector["family_consistent"]),
        "door_to_window": int(vector["door_to_window"]),
        "window_to_door": int(vector["window_to_door"]),
        "unmatched_predicted_openings": int(vector["unmatched_predicted"]),
        "unmatched_target_openings": int(vector["unmatched_target"]),
        "predicted_doors": int(vector["predicted_doors"]),
        "predicted_windows": int(vector["predicted_windows"]),
        "target_doors": int(vector["target_doors"]),
        "target_windows": int(vector["target_windows"]),
        "matched_doors": int(vector["matched_doors"]),
        "matched_windows": int(vector["matched_windows"]),
        "predicted_door_width_mean": float(
            vector["predicted_door_width_mean"]
        ),
        "predicted_window_width_mean": float(
            vector["predicted_window_width_mean"]
        ),
        "target_door_width_mean": float(vector["target_door_width_mean"]),
        "target_window_width_mean": float(vector["target_window_width_mean"]),
        "geometry_center_error_mean": float(
            vector["geometry_center_error_mean"]
        ),
        "geometry_width_ratio_mean": float(
            vector["geometry_width_ratio_mean"]
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    runtime = load_opening_heatmap_decoder(args.checkpoint, taxonomy, device)
    candidate_ranker = (
        load_opening_candidate_ranker(args.candidate_ranker, device=device)
        if args.candidate_ranker is not None
        else None
    )
    if (
        candidate_ranker is not None
        and args.candidate_ranker_threshold is not None
    ):
        candidate_ranker = replace(
            candidate_ranker,
            presence_threshold=float(args.candidate_ranker_threshold),
        )
    dataset = CubiCasaRasterControlDataset(
        args.data_root,
        args.split,
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit + args.offset,
        augment=False,
        write_cache=False,
    )
    if args.offset:
        dataset.samples = dataset.samples[args.offset : args.offset + args.limit]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    rows = []
    pending_cases = []
    for target, footprint, image, _floor_ids, names in loader:
        maps = {}
        if args.independent_foreground:
            maps = predict_opening_detection_maps(
                runtime,
                footprint.to(device),
                image.to(device),
                taxonomy,
                window_logit_bias=args.window_logit_bias,
            )
            probability = maps["family_probability"].cpu().numpy()
            foreground = maps["foreground"].cpu().numpy()
            family_logits = maps["family_logits"].cpu().numpy()
        else:
            probability = predict_opening_family_probabilities(
                runtime,
                footprint.to(device),
                image.to(device),
                taxonomy,
                window_logit_bias=args.window_logit_bias,
            ).cpu().numpy()
            foreground = None
            family_logits = None
        for index, name in enumerate(names):
            target_row = target[index].numpy()
            target_plan = raster_labels_to_wall_plan(target_row, taxonomy)
            proxy_plan = RasterWallPlan(
                frame=target_plan.frame,
                walls=target_plan.walls,
                rooms=target_plan.rooms,
                openings=[],
            )
            if "center" in maps and "interval_width" in maps:
                decoder = (
                    read_center_width_image_detections
                    if args.center_decode_mode == "image"
                    else read_center_width_opening_detections
                )
                predicted = decoder(
                    proxy_plan,
                    maps["center"][index].cpu().numpy(),
                    maps["interval_width"][index].cpu().numpy(),
                    maps["family_logits"][index].cpu().numpy(),
                    taxonomy,
                    threshold=args.threshold,
                    min_width=float(args.min_area),
                    min_peak_distance=args.min_peak_distance,
                )
            elif foreground is not None and family_logits is not None:
                predicted = learned_opening_detections_from_maps(
                    proxy_plan,
                    foreground[index],
                    family_logits[index],
                    taxonomy,
                    threshold=args.threshold,
                    min_area=args.min_area,
                    family_aggregation=args.family_aggregation,
                    exterior_wall_window_bias=args.exterior_window_bias,
                    interval_radius=args.interval_radius,
                    score_smoothing=args.score_smoothing,
                    max_gap_samples=args.max_gap_samples,
                    peak_relative_threshold=args.peak_relative_threshold,
                )
            else:
                predicted = learned_opening_detections(
                    proxy_plan,
                    probability[index],
                    taxonomy,
                    threshold=args.threshold,
                    min_area=args.min_area,
                )
            expected = detections_from_labels(target_row, taxonomy, target_plan)
            if candidate_ranker is not None:
                if foreground is None or family_logits is None:
                    raise ValueError(
                        "candidate ranker requires independent score maps"
                    )
                candidates = [
                    detection
                    for detection in predicted
                    if detection.wall_index is not None
                ]
                features = np.stack(
                    [
                        opening_candidate_features(
                            detection,
                            foreground[index],
                            family_logits[index],
                            proxy_plan,
                            predicted,
                            frame=float(args.image_size),
                        )
                        for detection in candidates
                    ]
                ) if candidates else np.zeros(
                    (0, len(FEATURE_NAMES)),
                    dtype=np.float32,
                )
                pending_cases.append(
                    {
                        "name": name,
                        "detections": predicted,
                        "features": features,
                        "expected": expected,
                        "probability": probability[index],
                        "target": target_row,
                        "footprint": footprint[index].numpy(),
                    }
                )
                continue
            rows.append(
                _case_row(
                    name,
                    predicted,
                    expected,
                    probability[index],
                    target_row,
                    footprint[index].numpy(),
                    taxonomy,
                    args.threshold,
                )
            )
    if candidate_ranker is not None and pending_cases:
        ranked_groups = candidate_ranker.rank_feature_groups(
            [case["detections"] for case in pending_cases],
            [case["features"] for case in pending_cases],
            taxonomy,
        )
        for case, predicted in zip(pending_cases, ranked_groups, strict=True):
            rows.append(
                _case_row(
                    case["name"],
                    predicted,
                    case["expected"],
                    case["probability"],
                    case["target"],
                    case["footprint"],
                    taxonomy,
                    args.threshold,
                )
            )
    report = {
        "schema_version": "opening-interval-decoder-eval-v1",
        "checkpoint": str(args.checkpoint),
        "scope": "oracle-host vectorization diagnostic",
        "test_opened": False,
        "target_used_at_inference": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "summary": numeric_row_summary(rows),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
