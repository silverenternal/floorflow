from __future__ import annotations

import json

import pytest

from floorflow.evaluation.dense_query_blend_multiseed import (
    aggregate_split,
    load_split_reports,
    promotion_gates,
)


def _report(
    seed: int,
    delta: float,
    *,
    lower_is_better: bool = False,
    key: str = "score",
) -> dict:
    dense = 0.30
    blend = dense + delta
    return {
        "target_used_at_inference": False,
        "test_opened": False,
        "validation_cases": 491,
        "cases": 491,
        "seed": seed,
        "path": f"report_seed{seed}.json",
        "summary": {
            "dense": {key: dense},
            "blend": {key: blend},
        },
        "paired": {
            key: {
                "mean_delta": delta,
                "ci_low": delta - (0.01 if not lower_is_better else 0.1),
                "ci_high": delta + (0.1 if not lower_is_better else 0.01),
            }
        },
    }


def test_aggregate_split_keeps_seed_level_direction_and_ci() -> None:
    reports = [
        _report(19, 0.02),
        _report(20, 0.01),
        _report(21, 0.00),
        _report(22, -0.01),
        _report(23, 0.03),
    ]
    result = aggregate_split(reports, metric_keys=("score",))
    metric = result["metrics"]["score"]
    assert metric["improved_seeds"] == 3
    assert metric["non_regression_seeds"] == 4
    assert metric["ci_excludes_zero_seeds"] == 2
    assert metric["worst_seed"] == 22
    assert metric["best_seed"] == 23
    assert result["cases_per_seed"] == [491]


def test_lower_is_better_direction_is_not_reversed() -> None:
    reports = [
        _report(
            seed,
            -0.1 * seed,
            lower_is_better=True,
            key="count_mae",
        )
        for seed in (19, 20, 21, 22, 23)
    ]
    result = aggregate_split(reports, metric_keys=("count_mae",))
    metric = result["metrics"]["count_mae"]
    assert metric["improved_seeds"] == 5
    assert metric["best_seed"] == 23
    assert metric["delta_min"] == pytest.approx(-2.3)


def test_aggregate_accepts_structural_mean_summary_fields() -> None:
    reports = [
        _report(seed, 0.01, key="room_macro_iou_mean")
        for seed in (19, 20, 21, 22, 23)
    ]
    for report in reports:
        report["paired"]["room_macro_iou"] = report["paired"].pop(
            "room_macro_iou_mean"
        )
    result = aggregate_split(reports, metric_keys=("room_macro_iou",))
    assert result["metrics"]["room_macro_iou"]["improved_seeds"] == 5


def test_structural_count_error_uses_lower_is_better_direction() -> None:
    reports = [
        _report(
            seed,
            -0.1 * seed,
            lower_is_better=True,
            key="instance_count_abs_error_mean",
        )
        for seed in (19, 20, 21, 22, 23)
    ]
    for report in reports:
        report["paired"]["instance_count_abs_error"] = report["paired"].pop(
            "instance_count_abs_error_mean"
        )
    result = aggregate_split(
        reports,
        metric_keys=("instance_count_abs_error",),
    )
    metric = result["metrics"]["instance_count_abs_error"]
    assert metric["improved_seeds"] == 5
    assert metric["best_seed"] == 23


def test_loader_rejects_duplicate_seed_reports(tmp_path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"dense_query_blend_seed19_v{index}.json"
        path.write_text(
            json.dumps(
                {
                    "target_used_at_inference": False,
                    "test_opened": False,
                    "validation_cases": 64,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    with pytest.raises(ValueError, match="duplicate seed"):
        load_split_reports(paths)


def test_promotion_gate_holds_when_a_split_is_missing() -> None:
    result = promotion_gates({}, expected_seeds=[19, 20, 21, 22, 23])
    assert result["passed"] is False
    assert "val491" in result["missing_splits"]
