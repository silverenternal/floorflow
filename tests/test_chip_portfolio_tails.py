from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_chip_portfolio_tails import metric, strata, worst_cases


def _row(
    index: int,
    blocks: int,
    cost: float,
    *,
    legalized: bool = False,
) -> dict:
    prefix = "legalized_" if legalized else ""
    return {
        "index": index,
        "blocks": blocks,
        f"{prefix}cost": cost,
        f"{prefix}v_rel": 0.1,
        f"{prefix}area_gap": 0.2,
        f"{prefix}b2b_hpwl_gap": 0.3,
    }


def test_metric_reads_legalized_or_normalized_keys() -> None:
    assert metric({"cost": 1.2}, "cost") == 1.2
    assert metric({"legalized_cost": 1.1}, "cost") == 1.1


def test_worst_cases_preserves_selection_provenance() -> None:
    candidate = [
        {**_row(0, 21, 1.4), "selection": "candidate:a"},
        {**_row(1, 22, 1.0), "selection": "candidate:b"},
    ]
    reference = [
        _row(0, 21, 1.0, legalized=True),
        _row(1, 22, 1.1, legalized=True),
    ]
    worst = worst_cases(candidate, reference, 2)
    assert [row["index"] for row in worst] == [0, 1]
    assert worst[0]["selection"] == "candidate:a"
    assert worst[0]["cost_delta"] > 0.0


def test_strata_reports_absolute_and_relative_deltas() -> None:
    candidate = [_row(0, 21, 1.2), _row(1, 41, 1.4)]
    reference = [
        _row(0, 21, 1.0, legalized=True),
        _row(1, 41, 1.2, legalized=True),
    ]
    report = strata(candidate, reference)
    assert report["21-40"]["cases"] == 1
    assert report["41-41"]["cases"] == 1
    assert report["21-40"]["delta"]["cost"] == pytest.approx(0.2)
    assert report["21-40"]["relative"]["cost"] == pytest.approx(0.2)
