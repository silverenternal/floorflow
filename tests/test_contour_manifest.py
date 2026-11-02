from __future__ import annotations

import json

import pytest

from floorflow.evaluation.contour_manifest import validate_contour_only_report


def test_contour_manifest_rejects_target_control(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "cases": 1,
                "best_of_n_used": False,
                "masked_control": True,
                "guided_control": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="masked"):
        validate_contour_only_report(path)


def test_contour_manifest_accepts_contour_report(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "cases": 1,
                "best_of_n_used": False,
                "masked_control": False,
                "guided_control": False,
            }
        ),
        encoding="utf-8",
    )
    assert validate_contour_only_report(path)["cases"] == 1
