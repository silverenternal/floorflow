from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from trinity.floorplan.legalize import pack_lp
from trinity.floorplan.types import FloorplanInstance, Placement

from floorflow.chip_flow.legalization import (
    EVALUATION_TIME_LIMIT_MS,
    scale_pack_legalize_reproducible,
)
from floorflow.chip_flow.legalizer_policy import LegalizerPolicy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_chip_official import score_case


def _placement() -> Placement:
    count = 24
    areas = np.ones(count, dtype=np.float64)
    constraints = np.zeros((count, 5), dtype=np.float64)
    constraints[:4, 3] = 1
    xywh = np.stack(
        [
            np.arange(count, dtype=np.float64) * 0.15,
            np.mod(np.arange(count, dtype=np.float64), 4) * 0.15,
            np.ones(count),
            np.ones(count),
        ],
        axis=1,
    )
    targets = np.full((count, 4), -1.0, dtype=np.float64)
    instance = FloorplanInstance(
        block_count=count,
        area_targets=areas,
        constraints=constraints,
        b2b=np.empty((0, 3), dtype=np.float64),
        p2b=np.empty((0, 3), dtype=np.float64),
        pins_pos=np.empty((0, 2), dtype=np.float64),
        target_positions=targets,
        gt_positions=xywh.copy(),
    )
    return Placement(xywh=xywh, instance=instance)


def test_reproducible_legalizer_is_deterministic_and_restores_timeout() -> None:
    previous = pack_lp.TIME_LIMIT_MS
    placement = _placement()
    first = scale_pack_legalize_reproducible(placement)
    second = scale_pack_legalize_reproducible(placement)
    assert pack_lp.TIME_LIMIT_MS == previous
    assert np.array_equal(first.xywh, second.xywh)
    assert pack_lp.TIME_LIMIT_MS < EVALUATION_TIME_LIMIT_MS


def test_reproducible_legalizer_honors_explicit_matched_timeout() -> None:
    previous = pack_lp.TIME_LIMIT_MS
    scale_pack_legalize_reproducible(_placement(), time_limit_ms=2000)
    assert pack_lp.TIME_LIMIT_MS == previous


def test_official_score_can_return_the_scored_legalized_boxes() -> None:
    placement = _placement()
    instance = placement.instance
    row = score_case(
        {
            "index": 0,
            "blocks": instance.block_count,
            "boxes": placement.xywh,
            "gt_boxes": instance.gt_positions,
            "areas": instance.area_targets,
            "constraints": instance.constraints,
            "b2b": instance.b2b,
            "p2b": instance.p2b,
            "pins": instance.pins_pos,
            "latent_mse": 0.0,
            "legalize": True,
            "legalizer_options": {"mu": 0.5},
            "return_legalized_boxes": True,
            "legalizer_time_limit_ms": 2000,
        }
    )
    assert row["legalized_feasible"]
    assert np.asarray(row["legalized_xywh"]).shape == placement.xywh.shape


def test_legalizer_policy_round_trips_reshape_and_trust_controls() -> None:
    policy = LegalizerPolicy(
        relation="expanded",
        reshape=False,
        trust=1.1,
        outline=False,
        time_limit_ms=3000,
    )
    assert policy.kwargs() == {
        "relation": "expanded",
        "reshape": False,
        "trust": 1.1,
        "outline": False,
    }
    placement = _placement()
    row = score_case(
        {
            "index": 0,
            "blocks": placement.instance.block_count,
            "boxes": placement.xywh,
            "gt_boxes": placement.instance.gt_positions,
            "areas": placement.instance.area_targets,
            "constraints": placement.instance.constraints,
            "b2b": placement.instance.b2b,
            "p2b": placement.instance.p2b,
            "pins": placement.instance.pins_pos,
            "latent_mse": 0.0,
            "legalize": True,
            "legalizer_options": policy.kwargs(),
            "legalizer_time_limit_ms": policy.time_limit_ms,
        }
    )
    assert row["legalized_feasible"]
    assert row["legalizer_reshaped"] == 0
