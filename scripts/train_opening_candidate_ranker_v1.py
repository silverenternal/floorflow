#!/usr/bin/env python3
"""Train a lightweight verifier for wall-interval opening candidates."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, replace

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.geometry.opening_readout import (
    OpeningDetection,
    detections_from_labels,
    opening_detection_metrics,
)
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.inference.opening_candidate_ranker import (
    FEATURE_NAMES,
    FEATURE_NAMES_V3,
    OpeningCandidateRanker,
    opening_candidate_features,
    opening_candidate_features_v3,
    save_opening_candidate_ranker,
)
from floorflow.inference.opening_candidate_torch import (
    train_opening_candidate_mlp,
)
from floorflow.inference.opening_center_width_decoder import (
    read_center_width_opening_detections,
)
from floorflow.inference.opening_heatmap_decoder import (
    learned_opening_detections_from_maps,
    load_opening_heatmap_decoder,
    predict_opening_detection_maps,
)
from floorflow.training.opening_candidate_ranker_cli import parse_args


@dataclass(frozen=True)
class CandidateCase:
    case: str
    detections: list[OpeningDetection]
    expected: list[OpeningDetection]
    features: np.ndarray
    presence: np.ndarray
    family: np.ndarray
    baseline: dict[str, float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _diagnostic_metrics(
    predicted: list[OpeningDetection],
    expected: list[OpeningDetection],
    tolerance: float,
) -> dict[str, float]:
    metrics = opening_detection_metrics(
        predicted,
        expected,
        tolerance=tolerance,
    )
    return {
        key: float(value)
        for key, value in metrics.items()
    }


def _candidate_labels(
    detections: list[OpeningDetection],
    expected: list[OpeningDetection],
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    presence = np.zeros(len(detections), dtype=np.int64)
    family = np.full(len(detections), -1, dtype=np.int64)
    if not detections or not expected:
        return presence, family
    cost = np.full((len(detections), len(expected)), np.inf)
    for row, detection in enumerate(detections):
        for column, target in enumerate(expected):
            distance = float(
                np.linalg.norm(
                    np.asarray(detection.center) - np.asarray(target.center)
                )
            )
            width_ratio = min(detection.width, target.width) / max(
                1.0,
                detection.width,
                target.width,
            )
            if distance <= tolerance and width_ratio >= 0.3:
                cost[row, column] = distance
    rows, columns = linear_sum_assignment(
        np.where(np.isfinite(cost), cost, 1.0e9)
    )
    for row, column in zip(rows, columns, strict=True):
        if not np.isfinite(cost[row, column]):
            continue
        presence[row] = 1
        family[row] = int(expected[column].family == "window")
    return presence, family


def _dataset(
    args: argparse.Namespace,
    split: str,
    offset: int,
    limit: int,
) -> CubiCasaRasterControlDataset:
    dataset = CubiCasaRasterControlDataset(
        args.data_root,
        split,
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=offset + limit,
        augment=False,
        write_cache=False,
    )
    if offset:
        dataset.samples = dataset.samples[offset : offset + limit]
    return dataset


@torch.inference_mode()
def collect_cases(
    args: argparse.Namespace,
    runtime,
    taxonomy: FloorplanTaxonomy,
    dataset: CubiCasaRasterControlDataset,
    device: torch.device,
) -> list[CandidateCase]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    cases = []
    tolerance = 0.05 * float(args.image_size)
    feature_function = (
        opening_candidate_features_v3
        if args.candidate_feature_version == "v3"
        else opening_candidate_features
    )
    feature_names = (
        FEATURE_NAMES_V3
        if args.candidate_feature_version == "v3"
        else FEATURE_NAMES
    )
    for target, footprint, image, _floor_ids, names in loader:
        maps = predict_opening_detection_maps(
            runtime,
            footprint.to(device),
            image.to(device),
            taxonomy,
        )
        foreground = maps["foreground"].cpu().numpy()
        family_logits = maps["family_logits"].cpu().numpy()
        for index, name in enumerate(names):
            target_row = target[index].cpu().numpy()
            host_labels = target_row
            if args.opening_host_raster_root is not None:
                host_path = (
                    args.opening_host_raster_root
                    / f"{str(name).replace('/', '__', 1)}.npz"
                )
                host_labels = np.load(host_path, allow_pickle=True)["prediction"]
                if host_labels.shape != target_row.shape:
                    host_labels = cv2.resize(
                        host_labels,
                        (target_row.shape[1], target_row.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
            plan = raster_labels_to_wall_plan(host_labels, taxonomy)
            if "center" in maps and "interval_width" in maps:
                detections = read_center_width_opening_detections(
                    plan,
                    maps["center"][index].cpu().numpy(),
                    maps["interval_width"][index].cpu().numpy(),
                    family_logits[index],
                    taxonomy,
                    threshold=args.candidate_threshold,
                    min_width=7.0,
                    min_peak_distance=args.min_peak_distance,
                )
            else:
                detections = learned_opening_detections_from_maps(
                    plan,
                    foreground[index],
                    family_logits[index],
                    taxonomy,
                    threshold=args.candidate_threshold,
                    min_area=7,
                )
            expected = detections_from_labels(target_row, taxonomy, plan)
            candidates = [
                detection
                for detection in detections
                if detection.wall_index is not None
            ]
            features = (
                np.stack(
                    [
                        feature_function(
                            detection,
                            foreground[index],
                            family_logits[index],
                            plan,
                            detections,
                            frame=float(args.image_size),
                        )
                        for detection in candidates
                    ]
                )
                if candidates
                else np.zeros(
                    (0, len(feature_names)),
                    dtype=np.float32,
                )
            )
            presence, family = _candidate_labels(
                candidates,
                expected,
                tolerance,
            )
            cases.append(
                CandidateCase(
                    case=str(name),
                    detections=candidates,
                    expected=expected,
                    features=features,
                    presence=presence,
                    family=family,
                    baseline=_diagnostic_metrics(
                        candidates,
                        expected,
                        tolerance,
                    ),
                )
            )
    return cases


def _mean_metrics(cases: list[CandidateCase], key: str) -> float:
    return float(np.mean([case.baseline[key] for case in cases]))


def _family_consistency_rate(metrics: dict[str, float]) -> float:
    return metrics["family_consistent"] / max(1.0, metrics["geometry_matched"])


def _fit_models(
    cases: list[CandidateCase],
    seed: int,
) -> tuple[HistGradientBoostingClassifier, HistGradientBoostingClassifier]:
    features = np.concatenate([case.features for case in cases])
    presence = np.concatenate([case.presence for case in cases])
    family = np.concatenate([case.family for case in cases])
    positive_family = family >= 0
    if set(np.unique(presence)) != {0, 1}:
        raise ValueError("candidate training data must contain positive and negative labels")
    if len(np.unique(family[positive_family])) != 2:
        raise ValueError("candidate training data must contain both opening families")
    common = {
        "max_iter": 240,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.5,
        "random_state": int(seed),
        "early_stopping": False,
    }
    presence_model = HistGradientBoostingClassifier(**common)
    presence_model.fit(
        features,
        presence,
        sample_weight=compute_sample_weight("balanced", presence),
    )
    family_model = HistGradientBoostingClassifier(**common)
    family_model.fit(
        features[positive_family],
        family[positive_family],
        sample_weight=compute_sample_weight(
            "balanced",
            family[positive_family],
        ),
    )
    return presence_model, family_model


def _fit_torch_ranker(
    cases: list[CandidateCase],
    args: argparse.Namespace,
    device: torch.device,
) -> OpeningCandidateRanker:
    features = np.concatenate([case.features for case in cases])
    presence = np.concatenate([case.presence for case in cases])
    family = np.concatenate([case.family for case in cases])
    torch_ranker = train_opening_candidate_mlp(
        features,
        presence,
        family,
        device=device,
        hidden=args.ranker_hidden,
        epochs=args.ranker_epochs,
        learning_rate=args.ranker_lr,
    )
    return OpeningCandidateRanker(
        presence_model=None,
        family_model=None,
        presence_threshold=0.5,
        family_mode="model",
        torch_ranker=torch_ranker,
        feature_names=FEATURE_NAMES_V3
        if args.candidate_feature_version == "v3"
        else FEATURE_NAMES,
    )


def _evaluate(
    cases: list[CandidateCase],
    ranker: OpeningCandidateRanker,
    taxonomy: FloorplanTaxonomy,
    tolerance: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = []
    for case in cases:
        predicted = ranker.rank_features(
            case.detections,
            case.features,
            taxonomy,
        )
        rows.append(
            {
                "case": case.case,
                **_diagnostic_metrics(
                    predicted,
                    case.expected,
                    tolerance,
                ),
            }
        )
    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key != "case"
    }
    return metrics, rows


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    runtime = load_opening_heatmap_decoder(args.checkpoint, taxonomy, device)
    train_cases = collect_cases(
        args,
        runtime,
        taxonomy,
        _dataset(
            args,
            args.train_split,
            args.train_offset,
            args.train_limit,
        ),
        device,
    )
    cal_cases = collect_cases(
        args,
        runtime,
        taxonomy,
        _dataset(args, "val", args.cal_offset, args.cal_limit),
        device,
    )
    holdout_cases = collect_cases(
        args,
        runtime,
        taxonomy,
        _dataset(args, "val", args.holdout_offset, args.holdout_limit),
        device,
    )
    tolerance = 0.05 * float(args.image_size)
    presence_model = family_model = None
    torch_ranker = None
    if args.torch_ranker:
        torch_ranker = _fit_torch_ranker(train_cases, args, device)
    else:
        presence_model, family_model = _fit_models(train_cases, args.seed)
    calibration = []
    for family_mode in ("detector", "model"):
        base_ranker = (
            replace(torch_ranker, family_mode=family_mode)
            if args.torch_ranker
            else OpeningCandidateRanker(
                presence_model=presence_model,
                family_model=family_model,
                presence_threshold=0.5,
                family_mode=family_mode,
                feature_names=FEATURE_NAMES_V3
                if args.candidate_feature_version == "v3"
                else FEATURE_NAMES,
            )
        )
        threshold_grid = (
            np.linspace(0.03, 0.30, 28)
            if args.torch_ranker
            else np.linspace(0.1, 0.9, 17)
        )
        for threshold in threshold_grid:
            ranker = replace(base_ranker, presence_threshold=float(threshold))
            metrics, rows = _evaluate(
                cal_cases,
                ranker,
                taxonomy,
                tolerance,
            )
            calibration.append(
                {
                    "threshold": float(threshold),
                    "family_mode": family_mode,
                    "metrics": metrics,
                    "cases": rows,
                }
            )
    selected = max(
        calibration,
        key=lambda row: float(row["metrics"]["f1"]),
    )
    ranker = replace(
        (
            torch_ranker
            if args.torch_ranker
            else OpeningCandidateRanker(
                presence_model=presence_model,
                family_model=family_model,
                presence_threshold=0.5,
                family_mode="model",
                feature_names=FEATURE_NAMES_V3
                if args.candidate_feature_version == "v3"
                else FEATURE_NAMES,
            )
        ),
        family_mode=str(selected["family_mode"]),
        presence_threshold=float(selected["threshold"]),
    )
    holdout_metrics, holdout_rows = _evaluate(
        holdout_cases,
        ranker,
        taxonomy,
        tolerance,
    )
    baseline = {
        "cal": {
            key: _mean_metrics(cal_cases, key)
            for key in cal_cases[0].baseline
        },
        "holdout": {
            key: _mean_metrics(holdout_cases, key)
            for key in holdout_cases[0].baseline
        },
    }
    paired_f1 = paired_delta(
        [
            {"value": row["f1"]}
            for row in holdout_rows
        ],
        [
            {"value": case.baseline["f1"]}
            for case in holdout_cases
        ],
        "value",
        iterations=5000,
        seed=args.seed,
    )
    baseline_family_rate = _family_consistency_rate(baseline["holdout"])
    holdout_family_rate = _family_consistency_rate(holdout_metrics)
    gate = {
        "cal_f1_above_baseline": (
            selected["metrics"]["f1"] > baseline["cal"]["f1"]
        ),
        "holdout_f1_above_baseline": (
            holdout_metrics["f1"] > baseline["holdout"]["f1"]
        ),
        "holdout_f1_ci_lower_positive": paired_f1["ci_low"] > 0.0,
        "holdout_precision_above_baseline": (
            holdout_metrics["precision"]
            > baseline["holdout"]["precision"]
        ),
        "holdout_recall_noninferior": (
            holdout_metrics["recall"]
            >= baseline["holdout"]["recall"] - 0.01
        ),
        "holdout_geometry_retention_at_least_0_95": (
            holdout_metrics["geometry_matched"]
            >= 0.95 * baseline["holdout"]["geometry_matched"]
        ),
        "holdout_family_rate_noninferior": (
            holdout_family_rate >= baseline_family_rate - 0.02
        ),
    }
    gate["promoted"] = all(gate.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / (
        "ranker.pt" if args.torch_ranker else "ranker.joblib"
    )
    save_opening_candidate_ranker(model_path, ranker)
    report = {
        "schema_version": "opening-candidate-ranker-v1",
        "scope": "oracle-host low-threshold interval candidate verification",
        "test_opened": False,
        "target_used_at_inference": False,
        "checkpoint": str(args.checkpoint),
        "opening_host_raster_root": (
            str(args.opening_host_raster_root)
            if args.opening_host_raster_root is not None
            else None
        ),
        "candidate_threshold": args.candidate_threshold,
        "budget": {
            "train_cases": len(train_cases),
            "train_candidates": int(
                sum(len(case.detections) for case in train_cases)
            ),
            "cal_cases": len(cal_cases),
            "holdout_cases": len(holdout_cases),
        },
        "baseline": baseline,
        "selected_threshold": float(selected["threshold"]),
        "selected_family_mode": str(selected["family_mode"]),
        "baseline_family_consistency_rate": baseline_family_rate,
        "holdout_family_consistency_rate": holdout_family_rate,
        "holdout_paired_f1": paired_f1,
        "calibration": calibration,
        "holdout": {
            "metrics": holdout_metrics,
            "cases": holdout_rows,
        },
        "gate": gate,
        "decision": "promote_candidate_ranker" if gate["promoted"] else "hold",
        "model": str(model_path),
        "model_format": "torch_mlp" if args.torch_ranker else "hist_gradient_boosting",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "baseline": baseline,
                "selected_threshold": selected["threshold"],
                "cal": selected["metrics"],
                "holdout": holdout_metrics,
                "gate": gate,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
