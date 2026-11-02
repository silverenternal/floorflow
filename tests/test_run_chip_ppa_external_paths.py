from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_chip_ppa_external.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_chip_ppa_external", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chipbench_commands_accept_repo_relative_paths() -> None:
    run_chip_ppa_external = _module()
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        directory = Path(temp)
        work_home = directory / "work"
        tcl = directory / "macro.tcl"
        input_def = directory / "input.def"
        tcl.write_text("place_macro\n")
        input_def.write_text("VERSION 5.8 ;\n")
        args = argparse.Namespace(
            orfs_root=ROOT / "third_party/OpenROAD-flow-scripts",
            lock=ROOT / "configs/chip_orfs_external_lock_v1.json",
            manifest=ROOT / "unused.json",
            variant="test",
            work_home=work_home.relative_to(ROOT),
            macro_placement_tcl=tcl.relative_to(ROOT),
            chipbench_input_def=input_def.relative_to(ROOT),
            num_cores=1,
            smoke_output=ROOT / "unused-output.json",
            chipbench_flow=True,
            chipbench_def2db=True,
            flow_target="finish",
            skip_cts_repair_timing=True,
            detailed_route_end_iter=10,
        )
        args = run_chip_ppa_external.resolve_paths(args)
        assert args.work_home.is_absolute()
        assert args.macro_placement_tcl.is_absolute()
        assert args.chipbench_input_def.is_absolute()
        manifest = {
            "inventory": [
                {
                    "case": "case",
                    "design_key": "case",
                    "platform": "nangate45",
                    "design_name": "case",
                    "design_nickname": "case",
                    "config": str(
                        directory.relative_to(ROOT) / "config.mk"
                    ),
                }
            ]
        }
        case = run_chip_ppa_external.case_record(manifest, "case")
        lock = {
            "image": {"local_tag": "openroad/orfs:test"},
            "flow_settings": {"num_cores": 1},
        }
        command = run_chip_ppa_external.build_chipbench_def2db_command(
            args,
            lock,
            case,
        )
        assert any("TARGET_DEF_PATH=" in item for item in command)
        assert "--name" in command
        pre_macro_command = run_chip_ppa_external.build_chipbench_pre_macro_command(
            args,
            lock,
            case,
        )
        assert "--name" in pre_macro_command
        pre_macro_shell = pre_macro_command[-1]
        assert "INPUT_ODB=" in pre_macro_shell
        assert "chipbench_pre_macro_odb.tcl" in pre_macro_shell
        assert "2_3_floorplan_macro.firm.odb" in pre_macro_shell
        flow_command = run_chip_ppa_external.build_flow_command(
            args,
            lock,
            case,
        )
        assert "--name" in flow_command
        shell = flow_command[-1]
        assert (
            "do-2_4_floorplan_tapcell do-2_5_floorplan_pdn "
            "do-2_floorplan do-place do-cts do-route do-finish"
        ) in shell
        assert "MACRO_PLACEMENT_TCL" not in shell
        assert "SKIP_CTS_REPAIR_TIMING=1" in flow_command
        assert "DETAILED_ROUTE_ARGS='-droute_end_iter 10'" in shell
        metrics_command = run_chip_ppa_external.build_metrics_command(
            args,
            lock,
            case,
            work_home / "logs",
            work_home / "reports",
            work_home / "results",
            work_home / "metrics.json",
        )
        metrics_shell = metrics_command[-1]
        assert "--flowPath" in metrics_shell
        assert "--logs" not in metrics_shell
