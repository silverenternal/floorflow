import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/analyze_chip_outline_failure_modes.py"
    spec = importlib.util.spec_from_file_location("outline_failure_modes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outline_failure_modes_separate_outline_and_overlap() -> None:
    report = {
        "cases": [
            {
                "id": "gsrc/hard/a",
                "candidate": {
                    "draws": [
                        {"fits_outline": False, "overlap_free": True, "shapes_kept": True},
                        {"fits_outline": True, "overlap_free": False, "shapes_kept": True},
                        {"fits_outline": True, "overlap_free": True, "shapes_kept": True},
                    ]
                },
                "control": {"draws": []},
            }
        ]
    }
    result = _module().arm_modes(report, "candidate")
    assert result["outline_miss_no_overlap"] == 1
    assert result["outline_pass_overlap_fail"] == 1
    assert result["fully_feasible"] == 1
    assert result["shape_violation"] == 0
