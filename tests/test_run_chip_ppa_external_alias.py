import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_chip_ppa_external.py"
    spec = importlib.util.spec_from_file_location("ppa_external_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case_record_accepts_design_key_suffix_alias() -> None:
    manifest = {
        "inventory": [
            {
                "case": "bp_be_top",
                "design_key": "nangate45/bp_be_top",
                "platform": "nangate45",
                "design_name": "bp_be_top",
                "design_nickname": "bp_be",
            }
        ]
    }
    row = _module().case_record(manifest, "nangate45__bp_be_top")
    assert row["design_key"] == "nangate45/bp_be_top"
