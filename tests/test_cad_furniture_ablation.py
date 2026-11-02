from floorflow.evaluation.cad_furniture_ablation import (
    compare_furniture_arm_reports,
)


def _history(value: float, furniture: float):
    return [
        {
            "furniture_f1": furniture * (index + 1) / 8.0,
            "room_macro_iou": value,
            "instance_f1": value,
            "wall_f1": value,
            "opening_f1": value,
        }
        for index in range(8)
    ]


def test_furniture_ablation_requires_positive_trend_and_safety() -> None:
    dense = {"history": _history(0.5, 0.0)}
    query = {"history": _history(0.5, 0.2)}
    report = compare_furniture_arm_reports(dense, query)
    assert report["gates"]["safety_pass"]
    assert report["gates"]["furniture_pass"]
    assert report["gates"]["promotion_ready"]


def test_furniture_ablation_rejects_safety_regression() -> None:
    dense = {"history": _history(0.5, 0.0)}
    query = {"history": _history(0.4, 0.2)}
    report = compare_furniture_arm_reports(dense, query)
    assert not report["gates"]["safety_pass"]
    assert not report["gates"]["promotion_ready"]
