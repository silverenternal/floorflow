from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_chip_ppa_external.py"
    spec = importlib.util.spec_from_file_location("ppa_external", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_chipbench_config_mounts_through_work() -> None:
    module = _module()
    lock = json.loads(
        (ROOT / "configs/chip_orfs_external_lock_v1.json").read_text()
    )
    args = Namespace(
        variant="test",
        work_home=ROOT / "outputs/chip_ppa/orfs",
        macro_placement_tcl=None,
        flow_target="finish",
        num_cores=1,
        chipbench_flow=False,
    )
    case = {
        "config": "third_party/ChiPBench/flow/designs/nangate45/VeriGPU/config.mk",
        "platform": "nangate45",
        "design_nickname": "VeriGPU",
    }
    command = " ".join(module.build_flow_command(args, lock, case))
    assert "DESIGN_CONFIG=/work/third_party/ChiPBench/" in command
    assert "DESIGN_HOME=/work/third_party/ChiPBench/flow/designs" in command
    assert "/src/OpenROAD-flow-scripts/third_party/ChiPBench" not in command


def test_chipbench_flow_uses_its_own_makefile() -> None:
    module = _module()
    lock = json.loads(
        (ROOT / "configs/chip_orfs_external_lock_v1.json").read_text()
    )
    args = Namespace(
        variant="test",
        work_home=ROOT / "outputs/chip_ppa/orfs",
        macro_placement_tcl=None,
        flow_target="finish",
        num_cores=1,
        chipbench_flow=True,
    )
    case = {
        "config": "outputs/chipbench_holdout_ppa_v1/orfs_designs/nangate45/VeriGPU/config.mk",
        "platform": "nangate45",
        "design_nickname": "VeriGPU",
    }
    command = " ".join(module.build_flow_command(args, lock, case))
    assert "-f /work/third_party/ChiPBench/flow/Makefile" in command
    assert "FLOW_HOME=/work/third_party/ChiPBench/flow" in command


def test_chipbench_def2db_uses_candidate_def() -> None:
    module = _module()
    lock = json.loads(
        (ROOT / "configs/chip_orfs_external_lock_v1.json").read_text()
    )
    args = Namespace(
        variant="test",
        work_home=ROOT / "outputs/chip_ppa/orfs",
        num_cores=1,
        chipbench_input_def=(
            ROOT
            / "outputs/chipbench_holdout_ppa_v1/candidate/VeriGPU/candidate_seed19.def"
        ),
    )
    case = {
        "config": "outputs/chipbench_holdout_ppa_v1/orfs_designs/nangate45/VeriGPU/config.mk",
        "platform": "nangate45",
        "design_nickname": "VeriGPU",
    }
    command = " ".join(module.build_chipbench_def2db_command(args, lock, case))
    assert "TARGET_DEF_PATH=/work/outputs/chipbench_holdout_ppa_v1" in command
    assert "TARGET_DB_PATH=2_3_floorplan_macro.odb" in command
    assert command.endswith("def2db")
