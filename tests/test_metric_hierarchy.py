from __future__ import annotations

from floorflow.evaluation.metric_hierarchy import (
    LEVEL_METRICS,
    audit_module_ownership,
    build_metric_hierarchy,
    hierarchy_markdown,
)


def test_metric_hierarchy_separates_levels_and_links() -> None:
    report = build_metric_hierarchy(
        {
            "cad_metrics": {
                "summary": {
                    "room_macro_iou": 0.2,
                    "furniture_iou": 0.1,
                    "furniture_room_containment": 0.5,
                    "fpeval_validity": 1.0,
                    "conditional_watertight": 0.97,
                    "room_target_recall": 0.96,
                    "outside_footprint_pixels": 0.0,
                    "fpeval_watertight": 0.9,
                    "fpeval_room_polygon_f1": 0.2,
                }
            },
            "room_metrics": {"overlay": {"room_macro_iou": 0.2}},
            "object_metrics": {"overlay": {"opening_iou": 0.1}},
            "constraint_metrics": {"opening_wall_alignment": 0.4},
        },
        style_report={
            "style_separation": 0.05,
            "seed_iou_gap": 0.0,
            "worst_style_room_iou": 0.18,
            "all_style_mean_room_iou": 0.19,
            "runtime_seconds_mean": 0.2,
        },
        recursive_report={"direct": {"room_macro_iou": 0.18}},
        decision_audit={
            "by_action": {
                "direct": {"room_macro_iou": 0.15},
                "recursive": {"room_macro_iou": 0.18},
            }
        },
        identifiability={"strict_0_95_identifiable_fraction": 0.006},
        diversity_report={
            "styles": {
                "0": {
                    "distribution": {
                        "mean_within_condition_feature_distance": 0.3,
                    }
                }
            }
        },
    )
    assert set(report["metrics"]) == set(LEVEL_METRICS)
    assert report["links"]["style"]["predicted"] == 0.18
    assert report["links"]["recursion"]["predicted"] > 0
    assert report["metrics"]["L5_generation"]["diversity"] == 0.3
    assert report["dominant_link"] is not None
    assert "L0_legal" in hierarchy_markdown(report)
    assert report["module_ownership"]["complete"] is True


def test_module_ownership_audit_flags_unowned_modules() -> None:
    assert audit_module_ownership()["complete"] is True
    broken = audit_module_ownership(
        {"floorflow.example": (), "floorflow.owned": ("L2",)}
    )
    assert broken["complete"] is False
    assert broken["missing_levels"] == ["floorflow.example"]
