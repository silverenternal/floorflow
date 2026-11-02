from __future__ import annotations

from floorflow.evaluation.opening_failure_strata import (
    analyze_opening_failure_strata,
    diagnosis_category,
    target_count_bin,
)


def _oracle(
    case: str,
    target: int,
    predicted: int,
    matched: int,
    vector_f1: float,
) -> dict:
    return {
        "case": case,
        "target_openings": target,
        "predicted_openings": predicted,
        "matched_openings": matched,
        "pixel_f1": vector_f1,
        "vector_f1": vector_f1,
        "vector_precision": vector_f1,
        "vector_recall": vector_f1,
    }


def test_opening_failure_categories_cover_miss_extra_and_count_correct() -> None:
    assert diagnosis_category(_oracle("x", 5, 0, 0, 0.0)) == "complete_miss"
    assert diagnosis_category(_oracle("x", 5, 3, 3, 0.5)) == "under_detection"
    assert diagnosis_category(_oracle("x", 3, 6, 3, 0.5)) == "over_detection"
    assert diagnosis_category(_oracle("x", 5, 6, 2, 0.3)) == "mixed_miss_and_extra"
    assert (
        diagnosis_category(_oracle("x", 3, 3, 3, 0.8))
        == "count_correct_residual"
    )
    assert target_count_bin(0) == "0"
    assert target_count_bin(3) == "1-3"
    assert target_count_bin(8) == "8-11"
    assert target_count_bin(12) == "12+"


def test_opening_failure_strata_join_case_names() -> None:
    baseline = {
        "cases": [
            {"case": "group__1__Floor-1", "fpeval_opening_f1": 0.0},
            {"case": "group__2__Floor-1", "fpeval_opening_f1": 0.5},
        ]
    }
    candidate = {
        "cases": [
            {"case": "group__1__Floor-1", "fpeval_opening_f1": 0.5},
            {"case": "group__2__Floor-1", "fpeval_opening_f1": 0.25},
        ]
    }
    oracle = {
        "cases": [
            _oracle("group/1__Floor-1", 2, 1, 1, 0.5),
            _oracle("group/2__Floor-1", 2, 0, 0, 0.0),
        ]
    }
    report = analyze_opening_failure_strata(
        baseline,
        candidate,
        oracle,
        iterations=20,
    )
    assert report["cases"] == 2
    assert report["overall"]["worsened_cases"] == 1
    assert report["overall"]["improved_cases"] == 1
    assert report["by_diagnosis"]["complete_miss"]["cases"] == 1
    assert report["by_diagnosis"]["under_detection"]["cases"] == 1


