"""Shared Docker and path utilities for pinned ORFS runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORFS = ROOT / "third_party/OpenROAD-flow-scripts"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("orfs_root", "lock", "manifest", "work_home", "smoke_output"):
        setattr(args, name, getattr(args, name).resolve())
    for name in ("macro_placement_tcl", "chipbench_input_def"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args


def run_logged(command: list[str], log_path: Path, timeout: int) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            exit_code = result.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            exit_code = None
            timed_out = True
            if command[:2] == ["docker", "run"] and "--name" in command:
                name = command[command.index("--name") + 1]
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
    return {
        "command": command,
        "log": str(log_path.relative_to(ROOT)),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": time.time() - started,
    }


def docker_prefix(
    lock: dict[str, Any],
    num_cores: int | None = None,
    extra_env: dict[str, str] | None = None,
    container_name: str | None = None,
) -> list[str]:
    cores = int(
        num_cores
        if num_cores is not None
        else lock["flow_settings"]["num_cores"]
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{DEFAULT_ORFS}:/src/OpenROAD-flow-scripts:ro",
        "-v",
        f"{ROOT}:/work",
        "-w",
        "/src/OpenROAD-flow-scripts/flow",
        "-e",
        "FLOW_HOME=/src/OpenROAD-flow-scripts/flow",
        "-e",
        "OPENROAD_EXE=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad",
        "-e",
        "YOSYS_EXE=/OpenROAD-flow-scripts/tools/install/yosys/bin/yosys",
        "-e",
        "KLAYOUT_CMD=/usr/bin/klayout",
        "-e",
        "PYTHON_EXE=/usr/bin/python3",
        "-e",
        f"NUM_CORES={cores}",
        "-e",
        "REPORT_ENDPOINT_SLACK=1",
    ]
    if container_name is not None:
        safe = "".join(
            character
            if character.isalnum() or character in "-_."
            else "_"
            for character in container_name
        )
        command.extend(["--name", safe[:120] + f"-{os.getpid()}"])
    for key, value in (extra_env or {}).items():
        command.extend(["-e", f"{key}={value}"])
    command.extend(
        [
            "--entrypoint",
            "/bin/bash",
            lock["image"]["local_tag"],
        ]
    )
    return command


def docker_name(args: argparse.Namespace, role: str) -> str:
    return f"floorflow-{args.variant}-{role}"


def flow_paths(
    args: argparse.Namespace,
    case: dict[str, Any],
) -> tuple[Path, Path, Path]:
    base = (
        args.work_home
        / "logs"
        / case["platform"]
        / case["design_nickname"]
        / args.variant
    )
    reports = (
        args.work_home
        / "reports"
        / case["platform"]
        / case["design_nickname"]
        / args.variant
    )
    results = (
        args.work_home
        / "results"
        / case["platform"]
        / case["design_nickname"]
        / args.variant
    )
    return base, reports, results


__all__ = [
    "DEFAULT_ORFS",
    "ROOT",
    "docker_name",
    "docker_prefix",
    "flow_paths",
    "load_json",
    "resolve_paths",
    "run_logged",
]
