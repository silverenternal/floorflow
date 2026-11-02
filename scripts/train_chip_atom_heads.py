#!/usr/bin/env python3
"""Train case-disjoint OOF atom heads over frozen candidate features."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.chip_flow.atom_score_heads import (
    ATOM_NAMES,
    ATOM_SPECS,
    selector_metrics_to_atom_targets,
)
from floorflow.chip_flow.candidate_manifest import TARGET_FREE_FEATURE_NAMES
from floorflow.chip_flow.candidate_structural_selector import (
    SELECTOR_METRICS,
    candidate_structural_features,
)
from floorflow.chip_flow.training_runtime import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("outputs/cohorts/chip_s205_consensus_500_v4"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("outputs/reports/chip_s3_candidate_legalized_labels_v3_500_v1.json"),
    )
    parser.add_argument(
        "--split-report",
        type=Path,
        default=Path("outputs/reports/chip_s205_consensus_500_split_v3.json"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument("--seed", type=int, default=241)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _labels(path: Path) -> dict[int, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["case_index"]): row["rows"]
        for row in payload["rows"]
    }


def _cases(
    cohort_dir: Path,
    labels: dict[int, dict],
    allowed: set[int],
) -> list[dict]:
    cases = []
    for path in sorted(cohort_dir.glob("*.npz")):
        payload = np.load(path)
        index = int(payload["case_index"])
        if index not in allowed or index not in labels:
            continue
        mask = np.asarray(payload["candidate_mask"])[0].astype(bool)
        features = np.asarray(payload["candidate_features"])[0][mask].astype(
            np.float64
        )
        names = [f"candidate_{value}" for value in range(len(features))]
        metrics = np.asarray(
            [
                [float(labels[index][name][metric]) for metric in SELECTOR_METRICS]
                for name in names
            ],
            dtype=np.float64,
        )
        atom_targets, atom_mask = selector_metrics_to_atom_targets(metrics)
        cases.append(
            {
                "index": index,
                "features": features,
                "metrics": metrics,
                "atom_targets": atom_targets.numpy(),
                "atom_mask": atom_mask.numpy(),
            }
        )
    if not cases:
        raise ValueError("atom-head training selected no cases")
    return cases


def _new_head(seed: int, trees: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=trees,
        max_depth=4,
        min_samples_leaf=8,
        max_features=0.7,
        random_state=seed,
        n_jobs=-1,
    )


def _fit_heads(
    cases: list[dict],
    trees: int,
    seed: int,
) -> dict[str, dict]:
    models: dict[str, dict] = {}
    for atom_index, spec in enumerate(ATOM_SPECS):
        features = np.concatenate(
            [candidate_structural_features(case["features"]) for case in cases]
        )
        targets = np.concatenate(
            [case["atom_targets"][:, atom_index] for case in cases]
        )
        masks = np.concatenate([case["atom_mask"][:, atom_index] for case in cases])
        targets = targets[masks]
        features = features[masks]
        mean = float(targets.mean()) if len(targets) else 0.0
        scale = float(targets.std()) if len(targets) else 1.0
        scale = max(scale, 1.0e-8)
        model = _new_head(seed + atom_index, trees)
        model.fit(features, (targets - mean) / scale)
        models[spec.name] = {
            "model": model,
            "mean": mean,
            "scale": scale,
        }
    return models


def _predict_case(models: dict[str, dict], case: dict) -> tuple[np.ndarray, np.ndarray]:
    features = candidate_structural_features(case["features"])
    predictions = []
    for atom_index, spec in enumerate(ATOM_SPECS):
        head = models[spec.name]
        values = (
            head["model"].predict(features) * head["scale"] + head["mean"]
        )
        values = np.where(case["atom_mask"][:, atom_index], values, 0.0)
        predictions.append(values)
    return np.stack(predictions, axis=-1), features


def _atom_stats(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> dict:
    stats = {}
    for atom_index, name in enumerate(ATOM_NAMES):
        spearman = []
        pair_total = pair_correct = 0
        top1 = 0
        regrets = []
        valid_cases = 0
        for row in range(len(predictions)):
            valid = masks[row, :, atom_index]
            if int(valid.sum()) < 2:
                continue
            valid_cases += 1
            predicted = predictions[row, valid, atom_index]
            target = targets[row, valid, atom_index]
            value = spearmanr(predicted, target).statistic
            if np.isfinite(value):
                spearman.append(float(value))
            for left in range(len(target)):
                for right in range(left + 1, len(target)):
                    if abs(target[left] - target[right]) <= 1.0e-12:
                        continue
                    pair_total += 1
                    pair_correct += int(
                        np.sign(predicted[right] - predicted[left])
                        == np.sign(target[right] - target[left])
                    )
            selected = int(np.argmin(predicted))
            optimal = int(np.argmin(target))
            top1 += int(selected == optimal)
            regrets.append(float(target[selected] - target[optimal]))
        stats[name] = {
            "valid_cases": valid_cases,
            "mean_spearman": float(np.mean(spearman)) if spearman else None,
            "pairwise_accuracy": (
                pair_correct / pair_total if pair_total else None
            ),
            "oracle_top1_rate": top1 / valid_cases if valid_cases else None,
            "mean_rank_regret": float(np.mean(regrets)) if regrets else None,
        }
    return stats


def main() -> None:
    args = parse_args()
    split = json.loads(args.split_report.read_text(encoding="utf-8"))
    cases = _cases(
        args.cohort_dir,
        _labels(args.labels),
        {int(value) for value in split["train_cases"]},
    )
    if len(cases) < args.folds:
        raise ValueError("not enough cases for atom-head folds")
    oof_predictions = None
    fold_rows = []
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_index, val_index) in enumerate(kfold.split(cases)):
        train_cases = [cases[int(index)] for index in train_index]
        val_cases = [cases[int(index)] for index in val_index]
        models = _fit_heads(train_cases, args.trees, args.seed + 100 * fold)
        if oof_predictions is None:
            oof_predictions = np.full(
                (
                    len(cases),
                    max(len(case["features"]) for case in cases),
                    len(ATOM_NAMES),
                ),
                np.nan,
                dtype=np.float64,
            )
        for index in val_index:
            case = cases[int(index)]
            prediction, _features = _predict_case(models, case)
            oof_predictions[int(index), : len(prediction)] = prediction
        fold_rows.append(
            {
                "fold": fold,
                "train_cases": len(train_cases),
                "val_cases": len(val_cases),
            }
        )
    targets = np.full_like(oof_predictions, np.nan)
    masks = np.zeros_like(oof_predictions, dtype=bool)
    for index, case in enumerate(cases):
        count = len(case["features"])
        targets[index, :count] = case["atom_targets"]
        masks[index, :count] = case["atom_mask"]
    finite = np.isfinite(oof_predictions) & masks
    clean_predictions = np.where(finite, oof_predictions, 0.0)
    clean_targets = np.where(finite, targets, 0.0)
    stats = _atom_stats(clean_predictions, clean_targets, masks)
    final_models = _fit_heads(cases, args.trees, args.seed + 1000)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.write_bytes(pickle.dumps(final_models, protocol=pickle.HIGHEST_PROTOCOL))
    report = {
        "version": "chip-atom-heads-oof-v1",
        "status": "complete",
        "scope": "case-disjoint OOF atom-head screen; no target leakage",
        "cases": len(cases),
        "folds": fold_rows,
        "atom_names": list(ATOM_NAMES),
        "feature_contract": list(TARGET_FREE_FEATURE_NAMES),
        "stats": stats,
        "checkpoint": str(args.checkpoint),
        "oof_tensor": {
            "indices": [case["index"] for case in cases],
            "predictions": clean_predictions.tolist(),
            "targets": clean_targets.tolist(),
            "masks": masks.tolist(),
        },
    }
    write_json_atomic(args.output, report)
    print(json.dumps({"status": "complete", "cases": len(cases), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
