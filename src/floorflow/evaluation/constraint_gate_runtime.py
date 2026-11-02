"""Reusable candidate ranking and calibration helpers for CAGN."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from floorflow.evaluation.constraint_calibration import (
    apply_isotonic_calibration,
    binary_auroc,
    expected_calibration_error,
)
from floorflow.evaluation.constraint_ranking import ndcg_at_k


def combined_scores(group: dict, alpha: float) -> np.ndarray:
    return group["scores"] + float(alpha) * group["legal_logits"]


def fallback_rows(
    groups: list[dict],
    *,
    alpha: float,
    legal_threshold: float,
) -> list[dict]:
    rows = []
    for group in groups:
        scores = combined_scores(group, alpha).copy()
        feasible = group["hard"].astype(bool)
        selected = int(np.argmax(scores))
        legal_probability = float(
            torch.sigmoid(torch.tensor(group["legal_logits"][selected]))
        )
        if legal_probability < legal_threshold:
            legal_scores = np.where(feasible, group["legal_logits"], -np.inf)
            legal_selected = int(np.argmax(legal_scores))
            scores[legal_selected] = max(float(scores[selected]) + 1.0, 1.0)
        rows.append({**group, "scores": scores})
    return rows


def hard_scores(group: dict) -> np.ndarray:
    feasible = group["hard"].astype(np.float64)
    tie_break = group["hashes"].astype(np.float64)
    tie_break = tie_break / max(float(tie_break.max()) + 1.0, 1.0)
    return feasible * 1000.0 + tie_break


def gate_metrics(groups: list[dict], *, learned: bool) -> dict:
    ndcgs = []
    selected_legal = []
    selected_violation = []
    operation_ndcg = defaultdict(list)
    for group in groups:
        relevance = group["legal"].astype(np.float64)
        scores = group["scores"] if learned else hard_scores(group)
        scores = np.where(group["hard"], scores, -np.inf)
        score = ndcg_at_k(scores, relevance)
        ndcgs.append(score)
        operation_ndcg[str(group["operation"])].append(score)
        if group["hard"].any():
            index = int(np.argmax(scores))
            selected_legal.append(bool(group["legal"][index]))
            selected_violation.append(bool(group["labels"][index].any()))
    return {
        "ndcg_at_5": float(np.mean(ndcgs)),
        "selected_legal_rate": float(np.mean(selected_legal)),
        "selected_violation_rate": float(np.mean(selected_violation)),
        "groups": len(groups),
        "operation_ndcg": {
            key: float(np.mean(values))
            for key, values in operation_ndcg.items()
        },
    }


def gate_calibration_metrics(
    groups: list[dict],
    *,
    alpha: float,
    calibration_parameters: dict[str, list[float]],
) -> dict:
    probabilities = []
    labels = []
    for group in groups:
        feasible = group["hard"]
        scores = combined_scores(group, alpha)[feasible]
        probabilities.extend(torch.sigmoid(torch.tensor(scores)).tolist())
        labels.extend(group["legal"][feasible].astype(float).tolist())
    probability = apply_isotonic_calibration(
        np.asarray(probabilities, dtype=np.float64),
        calibration_parameters,
    )
    target = np.asarray(labels, dtype=bool)
    auroc = (
        binary_auroc(probability, target)
        if target.any() and (~target).any()
        else None
    )
    return {
        "legal_auroc": auroc,
        "ece": expected_calibration_error(probability, target),
        "rows": int(target.size),
    }


__all__ = [
    "combined_scores",
    "fallback_rows",
    "gate_calibration_metrics",
    "gate_metrics",
    "hard_scores",
]
