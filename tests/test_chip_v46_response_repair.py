import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_chip_v46_response_repair.py"
    spec = importlib.util.spec_from_file_location("response_repair", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_skips_only_complete_arm_reports(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"status": "complete"}))
    monkeypatch.setattr(module, "arm_report_path", lambda arm, count: path)
    assert module.arm_complete("candidate", 1) is True
    path.write_text(json.dumps({"status": "incomplete"}))
    assert module.arm_complete("candidate", 1) is False
