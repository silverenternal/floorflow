import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.audit_chip_external_benchmarks as external_audit
from scripts.audit_chip_external_benchmarks import (
    audit_case,
    embedded_license_evidence,
    expand_make_value,
    make_paths,
    parse_config,
)


def test_parse_config_reads_conditional_and_append_exports(tmp_path: Path) -> None:
    config = tmp_path / "designs" / "asap7" / "demo" / "config.mk"
    config.parent.mkdir(parents=True)
    config.write_text(
        "export DESIGN_NAME = demo\n"
        "export PLATFORM = asap7\n"
        "export DESIGN_NICKNAME ?= nickname\n"
        "\texport ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/demo.lef\n"
        "export ADDITIONAL_LIBS += $(PLATFORM_DIR)/lib/demo.lib\n",
        encoding="utf-8",
    )

    variables = parse_config(config, tmp_path)

    assert variables["DESIGN_NICKNAME"] == "nickname"
    assert variables["ADDITIONAL_LEFS"].endswith("/platforms/asap7/lef/demo.lef")
    assert variables["ADDITIONAL_LIBS"].endswith("/platforms/asap7/lib/demo.lib")


def test_embedded_license_evidence_finds_long_generated_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(external_audit, "ROOT", tmp_path)
    source = tmp_path / "generated.sv"
    source.write_text(
        "// generated netlist\n" * 800
        + "// Licensed under the Apache License, Version 2.0\n",
        encoding="utf-8",
    )

    evidence = embedded_license_evidence([source])

    assert evidence[0]["spdx"] == "Apache-2.0"


def test_make_paths_expands_globs(tmp_path: Path) -> None:
    first = tmp_path / "first.v"
    second = tmp_path / "second.v"
    first.write_text("module first; endmodule\n", encoding="utf-8")
    second.write_text("module second; endmodule\n", encoding="utf-8")

    assert make_paths(f"{tmp_path}/*.v") == [first, second]


def test_expand_make_value_handles_multiple_wildcards(tmp_path: Path) -> None:
    first = tmp_path / "first.v"
    second = tmp_path / "second.v"
    first.write_text("module first; endmodule\n", encoding="utf-8")
    second.write_text("module second; endmodule\n", encoding="utf-8")

    expanded = expand_make_value(
        "$(sort $(wildcard $(TMP)/*.v))",
        {"TMP": str(tmp_path)},
    )

    assert expanded.split() == [str(first), str(second)]


def test_audit_case_discovers_wrapper_macro_lefs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "orfs"
    design = root / "flow/designs/asap7/demo"
    source = root / "flow/designs/src/demo/demo.v"
    platform = root / "flow/platforms/asap7"
    for directory in (design, source.parent, platform):
        directory.mkdir(parents=True)
    (design / "config.mk").write_text(
        "export DESIGN_NAME = demo\n"
        "export PLATFORM = asap7\n"
        "export DESIGN_NICKNAME = demo\n"
        "export VERILOG_FILES = $(DESIGN_HOME)/src/demo/demo.v\n"
        "export SDC_FILE = $(DESIGN_DIR)/demo.sdc\n"
        "export WRAP_LEFS = $(DESIGN_DIR)/macro.lef\n"
        "export WRAP_LIBS = $(DESIGN_DIR)/macro.lib\n",
        encoding="utf-8",
    )
    (design / "demo.sdc").write_text("create_clock -period 1 clk\n", encoding="utf-8")
    (design / "macro.lef").write_text("MACRO wrapper_macro\n", encoding="utf-8")
    (design / "macro.lib").write_text("library (macro) { }\n", encoding="utf-8")
    source.write_text(
        "// SPDX-License-Identifier: Apache-2.0\nmodule demo; endmodule\n",
        encoding="utf-8",
    )
    (platform / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    monkeypatch.setattr(external_audit, "ROOT", root)

    row = audit_case(root, tmp_path / "work", "asap7/demo", "unused")

    assert row["macro_master_count"] == 1
    assert row["checks"]["macro_capable"]
