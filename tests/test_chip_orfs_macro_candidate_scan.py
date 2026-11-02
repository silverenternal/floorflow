import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/scan_chip_orfs_macro_candidates.py"
    spec = importlib.util.spec_from_file_location("orfs_macro_scan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scan_contract_is_static_and_requires_measured_instances() -> None:
    _module()
    report = json.loads(
        (
            ROOT / "outputs/reports/chip_orfs_macro_candidate_scan_v1.json"
        ).read_text()
    )
    assert report["status"] == "static_only_macro_counts_and_routes_pending"
    assert "measured nonzero" in report["gate"]
    assert report["configs_scanned"] >= 1
