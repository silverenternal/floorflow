import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/analyze_chip_matched_components.py"
    spec = importlib.util.spec_from_file_location("matched_components", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matched_component_gap_keeps_mib_no_discrimination() -> None:
    report = {
        "version": "synthetic",
        "status": "rejected",
        "inference_seed": 19,
        "seeds": {
            "19": {
                "candidate": {
                    field: 0.0
                    for field in _module().METRICS.values()
                },
                "anchor": {
                    field: 0.0
                    for field in _module().METRICS.values()
                },
            }
        },
    }
    result = _module().analyze(report)
    assert result["summary"]["mib"]["no_discrimination"] is True
