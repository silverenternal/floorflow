import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/analyze_chip_outline_fallback_effect.py"
    spec = importlib.util.spec_from_file_location("outline_fallback_effect", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fallback_effect_counts_gain_and_loss() -> None:
    def case(case_id: str, before: bool, after: bool):
        return {
            "id": case_id,
            "candidate": {
                "selected": {
                    "feasible": before,
                    "hpwl": 10.0,
                }
            },
            "control": {
                "selected": {
                    "feasible": before,
                    "hpwl": 10.0,
                }
            },
        }, {
            "id": case_id,
            "candidate": {
                "selected": {
                    "feasible": after,
                    "hpwl": 11.0,
                }
            },
            "control": {
                "selected": {
                    "feasible": after,
                    "hpwl": 11.0,
                }
            },
        }

    first, first_after = case("gsrc/hard/a", False, True)
    second, second_after = case("gsrc/hard/b", True, False)
    fallback = {"draws": 4, "cases": [first_after, second_after]}
    no_fallback = {"draws": 4, "cases": [first, second]}
    result = _module().analyze(fallback, no_fallback)
    assert result["overall"]["candidate"]["gained_cases"] == 1
    assert result["overall"]["candidate"]["lost_cases"] == 1
