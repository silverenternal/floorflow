"""Calibration and discrimination metrics for constraint feasibility scores."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with average ranks for ties."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[start:stop] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    rank_by_index = np.empty_like(ranks)
    rank_by_index[order] = ranks
    return float(
        (
            rank_by_index[labels].sum()
            - positive * (positive + 1) / 2.0
        )
        / (positive * negative)
    )


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    boundaries = np.linspace(0.0, 1.0, int(bins) + 1)
    indices = np.clip(
        np.digitize(probabilities, boundaries[1:-1], right=True),
        0,
        int(bins) - 1,
    )
    error = 0.0
    for index in range(int(bins)):
        mask = indices == index
        if not mask.any():
            continue
        error += mask.mean() * abs(
            probabilities[mask].mean() - labels[mask].mean()
        )
    return float(error)


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(
        np.square(
            np.asarray(probabilities, dtype=np.float64)
            - np.asarray(labels, dtype=np.float64)
        ).mean()
    )


def select_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    max_false_feasible_rate: float = 0.05,
) -> dict[str, float]:
    """Choose maximum balanced accuracy under a validation FFR constraint."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    candidates = np.unique(
        np.quantile(probabilities, np.linspace(0.0, 1.0, 201))
    )
    best = {
        "threshold": 1.0,
        "false_feasible_rate": 0.0,
        "true_feasible_rate": 0.0,
        "balanced_accuracy": -1.0,
    }
    for threshold in candidates:
        predicted = probabilities >= threshold
        ffr = float(predicted[~labels].mean()) if (~labels).any() else 0.0
        tfr = float(predicted[labels].mean()) if labels.any() else 0.0
        balanced = 0.5 * (ffr + tfr)
        if ffr <= max_false_feasible_rate and balanced > best["balanced_accuracy"]:
            best = {
                "threshold": float(threshold),
                "false_feasible_rate": ffr,
                "true_feasible_rate": tfr,
                "balanced_accuracy": balanced,
            }
    return best


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Fit one positive temperature on a calibration split only."""

    probability = torch.as_tensor(probabilities, dtype=torch.float32).clamp(
        1.0e-6,
        1.0 - 1.0e-6,
    )
    logits = torch.logit(probability)
    target = torch.as_tensor(labels, dtype=torch.float32)
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.25, 20.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / temperature,
            target,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.25, 20.0).detach())


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    probability = torch.as_tensor(probabilities, dtype=torch.float32).clamp(
        1.0e-6,
        1.0 - 1.0e-6,
    )
    return (
        (torch.logit(probability) / max(float(temperature), 1.0e-6))
        .sigmoid()
        .numpy()
    )


def fit_platt_scaling(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Fit slope and intercept on calibration logits."""

    probability = torch.as_tensor(probabilities, dtype=torch.float32).clamp(
        1.0e-6,
        1.0 - 1.0e-6,
    )
    logits = torch.logit(probability)
    target = torch.as_tensor(labels, dtype=torch.float32)
    slope = torch.nn.Parameter(torch.ones(()))
    intercept = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [slope, intercept],
        lr=0.1,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            slope * logits + intercept,
            target,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "slope": float(slope.detach()),
        "intercept": float(intercept.detach()),
    }


def apply_platt_scaling(
    probabilities: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    probability = torch.as_tensor(probabilities, dtype=torch.float32).clamp(
        1.0e-6,
        1.0 - 1.0e-6,
    )
    logits = torch.logit(probability)
    calibrated = (
        float(parameters["slope"]) * logits
        + float(parameters["intercept"])
    )
    return calibrated.sigmoid().numpy()


def fit_isotonic_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, list[float]]:
    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    model = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )
    model.fit(probability, target)
    return {
        "x": model.X_thresholds_.astype(float).tolist(),
        "y": model.y_thresholds_.astype(float).tolist(),
    }


def apply_isotonic_calibration(
    probabilities: np.ndarray,
    parameters: dict[str, list[float]],
) -> np.ndarray:
    return np.interp(
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(parameters["x"], dtype=np.float64),
        np.asarray(parameters["y"], dtype=np.float64),
    )


__all__ = [
    "apply_isotonic_calibration",
    "apply_platt_scaling",
    "apply_temperature",
    "binary_auroc",
    "brier_score",
    "expected_calibration_error",
    "fit_isotonic_calibration",
    "fit_platt_scaling",
    "fit_temperature",
    "select_threshold",
]
