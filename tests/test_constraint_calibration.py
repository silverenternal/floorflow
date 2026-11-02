"""Tests for constraint embedding calibration metrics."""

from __future__ import annotations

import numpy as np

from floorflow.evaluation.constraint_calibration import (
    apply_isotonic_calibration,
    apply_platt_scaling,
    apply_temperature,
    binary_auroc,
    brier_score,
    expected_calibration_error,
    fit_isotonic_calibration,
    fit_platt_scaling,
    fit_temperature,
    select_threshold,
)


def test_binary_auroc_handles_ties() -> None:
    scores = np.array([0.1, 0.4, 0.4, 0.9])
    labels = np.array([False, True, False, True])
    assert binary_auroc(scores, labels) == 0.875


def test_calibration_metrics_and_constrained_threshold() -> None:
    labels = np.array([True, True, True, False, False, False])
    probabilities = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    assert expected_calibration_error(probabilities, labels, bins=3) >= 0.0
    assert 0.0 <= brier_score(probabilities, labels) <= 1.0
    selected = select_threshold(
        probabilities,
        labels,
        max_false_feasible_rate=0.0,
    )
    assert selected["threshold"] > 0.2
    assert selected["false_feasible_rate"] == 0.0


def test_temperature_scaling_is_finite_and_order_preserving() -> None:
    probabilities = np.array([0.11, 0.25, 0.6, 0.9, 0.45, 0.75])
    labels = np.array([False, False, True, True, False, True])
    temperature = fit_temperature(probabilities, labels)
    calibrated = apply_temperature(probabilities, temperature)
    assert 0.25 <= temperature <= 20.0
    assert np.array_equal(np.argsort(probabilities), np.argsort(calibrated))


def test_platt_scaling_is_finite_and_reduces_calibration_error() -> None:
    probabilities = np.array([0.11, 0.25, 0.6, 0.9, 0.45, 0.75])
    labels = np.array([False, False, True, True, False, True])
    parameters = fit_platt_scaling(probabilities, labels)
    calibrated = apply_platt_scaling(probabilities, parameters)
    assert np.isfinite(parameters["slope"])
    assert np.isfinite(parameters["intercept"])
    assert expected_calibration_error(calibrated, labels) <= (
        expected_calibration_error(probabilities, labels) + 1.0e-6
    )


def test_isotonic_calibration_reduces_calibration_error() -> None:
    probabilities = np.linspace(0.01, 0.99, 50)
    labels = probabilities > 0.65
    parameters = fit_isotonic_calibration(probabilities, labels)
    calibrated = apply_isotonic_calibration(probabilities, parameters)
    assert np.all(np.diff(calibrated) >= -1.0e-12)
    assert expected_calibration_error(calibrated, labels) <= (
        expected_calibration_error(probabilities, labels)
    )
