from __future__ import annotations

from floorflow.chip_flow.ppa_external import parse_global_route_overflow


def test_global_route_overflow_parses_final_total_row() -> None:
    text = """
Layer         Resource        Demand        Usage (%)    Max H / Max V / Total Congestion
metal1               0             0            0.00%             1 /  2 /  3
Total          1000          500           50.00%            4 /  5 /  9
"""
    assert parse_global_route_overflow(text) == {
        "max_horizontal": 4,
        "max_vertical": 5,
        "total": 9,
    }


def test_global_route_overflow_returns_na_without_total_row() -> None:
    assert parse_global_route_overflow("no report") == {
        "max_horizontal": None,
        "max_vertical": None,
        "total": None,
    }
