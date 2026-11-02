"""Tests for the residential CAGN curriculum configuration."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cagn_curriculum_has_five_seeds_and_device_zero() -> None:
    config = json.loads(
        (ROOT / "configs/residential_cagn_joint_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["seeds"] == [19, 20, 21, 22, 23]
    assert config["device"] == "cuda:0"
    assert config["data"]["test_access"].startswith("forbidden")
    assert len(config["arms"]) == 5


def test_embedding_gate_requires_auroc_ffr_and_calibration() -> None:
    config = json.loads(
        (
            ROOT / "configs/residential_constraint_embedding_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["selection"]["gate"] == {
        "auroc_min": 0.9,
        "false_feasible_rate_max": 0.05,
        "ece_max": 0.05,
    }
    assert config["optimization"]["device"] == "cuda:0"
