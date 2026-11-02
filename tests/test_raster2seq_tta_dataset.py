"""Tests for deterministic Raster2Seq TTA geometry."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_raster2seq_tta_dataset_v1",
    ROOT / "scripts/build_raster2seq_tta_dataset_v1.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("could not load TTA dataset builder")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_transform_polygon_round_trip():
    polygon = [1.0, 2.0, 7.0, 2.0, 7.0, 5.0, 1.0, 5.0]
    inverse = {
        "hflip": "hflip",
        "vflip": "vflip",
        "rot180": "rot180",
        "rot90": "rot270",
        "rot270": "rot90",
        "transpose": "transpose",
        "transverse": "transverse",
    }
    for transform, inverse_transform in inverse.items():
        transformed = MODULE.transform_polygon(
            polygon,
            9,
            9,
            transform,
        )
        restored = MODULE.transform_polygon(
            transformed,
            9,
            9,
            inverse_transform,
        )
        assert restored == polygon
