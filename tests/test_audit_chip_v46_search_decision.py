from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/audit_chip_v46_search_decision.py"
    spec = importlib.util.spec_from_file_location("v46_search_decision", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_uses_lowest_index_for_ties() -> None:
    selection = _module().selection
    proposals = [
        {"proposal": 2, "score": 1.0},
        {"proposal": 0, "score": 1.0},
        {"proposal": 1, "score": 2.0},
    ]
    result = selection(proposals, "score")
    assert result == {
        "score": 1.0,
        "tied_proposals": [2, 0],
        "default_proposal": 0,
    }


def test_selection_rejects_all_missing() -> None:
    selection = _module().selection
    try:
        selection([{"proposal": 0, "score": None}], "score")
    except ValueError as error:
        assert "no finite values" in str(error)
    else:
        raise AssertionError("missing scores must be rejected")
