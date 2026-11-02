"""Contour-only inference adapter for a trained style classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from floorflow.representation.core import contour_descriptor


class ContourStylePredictor:
    """Predict a user-facing style id from footprint geometry only."""

    def __init__(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        coefficients: np.ndarray,
        intercepts: np.ndarray,
        classes: np.ndarray,
    ) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.intercepts = np.asarray(intercepts, dtype=np.float64)
        self.classes = np.asarray(classes, dtype=np.int64)

    @classmethod
    def load(cls, path: str | Path) -> ContourStylePredictor:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("input_mode") != "contour_only":
            raise ValueError("style predictor must use contour-only input")
        return cls(
            payload["center"],
            payload["scale"],
            payload["coefficients"],
            payload["intercepts"],
            payload["classes"],
        )

    def predict(self, footprints: np.ndarray) -> np.ndarray:
        array = np.asarray(footprints)
        if array.ndim == 2:
            array = array[None]
        descriptors = np.stack(
            [contour_descriptor(mask.astype(bool)) for mask in array]
        )
        standardized = (descriptors - self.center) / self.scale
        scores = standardized @ self.coefficients.T + self.intercepts
        return self.classes[scores.argmax(axis=1)]


__all__ = ["ContourStylePredictor"]
