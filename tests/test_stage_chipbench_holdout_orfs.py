from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/stage_chipbench_holdout_orfs.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage_chipbench", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_preserves_verilog_extension() -> None:
    module = _module()
    row = {
        "id": "or1200",
        "flow_config": (
            "third_party/ChiPBench/flow/designs/nangate45/or1200/config.mk"
        ),
        "verilog": ".data/chipbench_d_v1/data/or1200/or1200.v",
        "sdc": ".data/chipbench_d_v1/data/or1200/constraint.sdc",
        "macro_lefs": [],
        "libs": [],
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        design_dir = module.stage_design(row, Path(temp))
        assert (design_dir / "rtl" / "or1200.v").is_symlink()
        assert not (design_dir / "rtl" / "or1200.sv").exists()
