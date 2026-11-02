#!/usr/bin/env python3
"""Train a lightweight repair accept/reject controller from guarded rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression

from floorflow.inference.repair_registry import RepairAcceptanceController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--feature-prefix", default="feature_")
    parser.add_argument("--label", default="accepted")
    parser.add_argument("--validation-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/repair_controller.json")
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validation_rows = rows
    if args.validation_csv is not None:
        with args.validation_csv.open(newline="", encoding="utf-8") as handle:
            validation_rows = list(csv.DictReader(handle))
    feature_names = [name for name in rows[0] if name.startswith(args.feature_prefix)]
    features = [[float(row[name] or 0.0) for name in feature_names] for row in rows]
    labels = [str(row[args.label]).strip().lower() in {"1", "true", "yes"} for row in rows]
    validation_features = [
        [float(row[name] or 0.0) for name in feature_names]
        for row in validation_rows
    ]
    validation_labels = [
        str(row[args.label]).strip().lower() in {"1", "true", "yes"}
        for row in validation_rows
    ]
    values = np.asarray(features, dtype=np.float64)
    centers = np.median(values, axis=0)
    scales = np.std(values, axis=0)
    scales = np.where(scales > 1.0e-8, scales, 1.0)
    controller = RepairAcceptanceController(
        len(feature_names),
        centers=centers.tolist(),
        scales=scales.tolist(),
    )
    classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=17)
    normalized = (values - centers) / scales
    classifier.fit(normalized, np.asarray(labels, dtype=np.int64))
    controller.weights = [float(value) for value in classifier.coef_[0]]
    scores = np.asarray(
        [controller.score(row) for row in validation_features]
    )
    candidates = sorted(set(scores.tolist()))
    best = None
    for threshold in candidates:
        predictions = scores >= threshold
        true_positive = sum(
            pred and label
            for pred, label in zip(predictions, validation_labels)
        )
        false_positive = sum(
            pred and not label
            for pred, label in zip(predictions, validation_labels)
        )
        false_negative = sum(
            not pred and label
            for pred, label in zip(predictions, validation_labels)
        )
        true_negative = sum(
            not pred and not label
            for pred, label in zip(predictions, validation_labels)
        )
        recall = true_positive / max(1, true_positive + false_negative)
        false_accept = false_positive / max(1, false_positive + true_negative)
        precision = true_positive / max(1, true_positive + false_positive)
        if false_accept <= 0.20 and (best is None or (recall, precision) > (best["recall"], best["precision"])):
            best = {
                "threshold": float(threshold),
                "recall": recall,
                "false_accept_rate": false_accept,
                "precision": precision,
            }
    if best is None:
        raise RuntimeError("no validation threshold satisfies false_accept_rate <= 0.20")
    controller.threshold = best["threshold"]
    predictions = [
        controller.accept(row) for row in validation_features
    ]
    true_positive = sum(
        pred and label
        for pred, label in zip(predictions, validation_labels)
    )
    false_positive = sum(
        pred and not label
        for pred, label in zip(predictions, validation_labels)
    )
    false_negative = sum(
        not pred and label
        for pred, label in zip(predictions, validation_labels)
    )
    true_negative = sum(
        not pred and not label
        for pred, label in zip(predictions, validation_labels)
    )
    metrics = {
        "recall": true_positive / max(1, true_positive + false_negative),
        "false_accept_rate": false_positive / max(1, false_positive + true_negative),
        "precision": true_positive / max(1, true_positive + false_positive),
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "weights": controller.weights,
                "centers": controller.centers,
                "scales": controller.scales,
                "threshold": controller.threshold,
                "train_rows": len(rows),
                "validation_rows": len(validation_rows),
                **metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "features": len(feature_names), **metrics, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
