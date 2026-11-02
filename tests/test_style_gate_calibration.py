from floorflow.evaluation.style_gate_calibration import select_style_gates


def _report(style_id: int, gate: float, quality: float, coverage: float, density: float):
    return {
        "style_id": style_id,
        "style_gate": gate,
        "sample_quality": {"mean": quality},
        "distribution": {
            "coverage": coverage,
            "density": density,
        },
    }


def test_style_gate_selection_favors_quality_within_coverage_floor() -> None:
    reports = [
        _report(0, 1.0, 0.14, 0.0, 0.0),
        _report(0, 0.5, 0.13, 0.56, 1.0),
        _report(0, 0.25, 0.12, 0.60, 2.0),
    ]
    selected = select_style_gates(reports, minimum_coverage=0.56)
    assert selected["0"]["gate"] == 0.5
    assert selected["0"]["quality"] == 0.13


def test_style_gate_selection_falls_back_when_no_gate_has_coverage() -> None:
    reports = [_report(2, 1.0, 0.14, 0.0, 0.0)]
    selected = select_style_gates(reports, minimum_coverage=0.5)
    assert selected["2"]["gate"] == 1.0
    assert selected["2"]["coverage_gate_satisfied"] is False
