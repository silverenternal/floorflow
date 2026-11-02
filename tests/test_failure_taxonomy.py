from floorflow.evaluation.failure_taxonomy import (
    classify_room_failure,
    summarise_failure_taxonomy,
)


def _row(missing=0.0, semantic=0.0, spurious=0.0, iou=0.1, gap=0.0):
    return {
        "support_macro_iou": iou,
        "support_occupancy_iou": 0.9,
        "support_fragmentation_gap": gap,
        "allocation": {
            "missing_room_fraction": missing,
            "semantic_mismatch_fraction": semantic,
            "spurious_room_fraction": spurious,
        },
    }


def test_failure_taxonomy_prefers_dominant_error_mass() -> None:
    assert classify_room_failure(
        _row(semantic=0.6, spurious=0.2),
        fragmentation_threshold=3.0,
    ) == "semantic_confusion"
    assert classify_room_failure(
        _row(spurious=0.5),
        fragmentation_threshold=3.0,
    ) == "spurious_room"
    report = summarise_failure_taxonomy(
        [_row(semantic=0.6), _row(spurious=0.5), _row(gap=4.0)],
        fragmentation_threshold=3.0,
    )
    assert report["case_count"] == 3
    assert sum(value["count"] for value in report["categories"].values()) == 3
