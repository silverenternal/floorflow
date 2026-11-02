#!/usr/bin/env python3
"""Run only OpenROAD global placement as label-free proposal feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.analytical_feedback import summarize_analytical_response
from scripts.run_chip_ppa_external import (
    DEFAULT_LOCK,
    DEFAULT_MANIFEST,
    DEFAULT_WORK_HOME,
    case_record,
    docker_prefix,
    flow_paths,
    load_json,
    run_logged,
)

SEED_ARTIFACTS = (
    "1_1_yosys_canonicalize.rtlil",
    "1_2_yosys.sdc",
    "1_2_yosys.v",
    "1_synth.odb",
    "1_synth.sdc",
    "2_1_floorplan.odb",
    "2_1_floorplan.sdc",
    "mem.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--macro-placement-tcl", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-home", type=Path, default=DEFAULT_WORK_HOME)
    parser.add_argument("--seed-variant")
    parser.add_argument("--num-cores", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--run-flow", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_response.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_command(
    args: argparse.Namespace,
    lock: dict[str, Any],
    case: dict[str, Any],
) -> list[str]:
    config = (
        "/src/OpenROAD-flow-scripts/"
        + case["config"].split("third_party/OpenROAD-flow-scripts/")[-1]
    )
    placement = args.macro_placement_tcl.resolve().relative_to(ROOT)
    make = [
        "make",
        f"DESIGN_CONFIG={config}",
        f"FLOW_VARIANT={args.variant}",
        f"WORK_HOME=/work/{args.work_home.relative_to(ROOT)}",
        "OR_SEED=1",
        "GRT_SEED=1",
        f"MACRO_PLACEMENT_TCL=/work/{placement}",
        "3_3_place_gp",
    ]
    return [
        *docker_prefix(lock, args.num_cores),
        "-lc",
        " ".join(make),
    ]


def seed_previous_stage(source: Path, target: Path) -> list[str]:
    """Copy invariant synthesis and base-floorplan artifacts into a new variant."""

    copied = []
    target.mkdir(parents=True, exist_ok=True)
    for name in SEED_ARTIFACTS:
        source_path = source / name
        target_path = target / name
        if source_path.is_file() and not target_path.is_file():
            shutil.copyfile(source_path, target_path)
            copied.append(str(target_path))
    return copied


def main() -> None:
    args = parse_args()
    lock = load_json(args.lock)
    manifest = load_json(args.manifest)
    case = case_record(manifest, args.case)
    logs, _, results = flow_paths(args, case)
    seeded_artifacts = []
    if args.seed_variant and args.seed_variant != args.variant:
        seeded_artifacts = seed_previous_stage(
            args.work_home
            / "results"
            / case["platform"]
            / case["design_nickname"]
            / args.seed_variant,
            results,
        )
    stage_path = logs / "3_3_place_gp.json"
    flow_run = None
    if args.run_flow:
        command = build_command(args, lock, case)
        flow_log = (
            ROOT
            / "outputs/reports/orfs"
            / case["platform"]
            / case["design_nickname"]
            / f"{args.variant}_analytical_response.log"
        )
        flow_run = run_logged(command, flow_log, args.timeout_seconds)
    metrics = load_json(stage_path) if stage_path.is_file() else {}
    response = summarize_analytical_response(metrics)
    checks = {
        "flow_succeeded_or_reused": (
            flow_run["exit_code"] == 0
            if flow_run is not None
            else stage_path.is_file()
        ),
        "stage_metrics_present": stage_path.is_file(),
        "score_available": response["score"] is not None,
        "no_detailed_route_endpoint_used": True,
    }
    report = {
        "version": "chip-v46-analytical-response-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "case": case["design_key"],
        "variant": args.variant,
        "macro_placement_tcl": str(args.macro_placement_tcl),
        "macro_placement_tcl_sha256": sha256(args.macro_placement_tcl),
        "source": {
            "orfs_commit": lock["repository"]["commit"],
            "image_manifest_digest": lock["image"]["linux_amd64_manifest_digest"],
            "platform": case["platform"],
            "flow_target": "3_3_place_gp",
            "or_seed": 1,
            "grt_seed": 1,
        },
        "stage_metrics": str(stage_path.relative_to(ROOT)),
        "seeded_from_variant": args.seed_variant,
        "seeded_artifacts": seeded_artifacts,
        "response": response,
        "flow_run": flow_run,
        "checks": checks,
        "policy": {
            "target_metric_used_for_ranking": False,
            "detailed_route_executed": False,
            "physical_response_time_must_be_reported": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "case": report["case"],
                "score": response["score"],
                "elapsed_seconds": (
                    flow_run["elapsed_seconds"] if flow_run is not None else None
                ),
                "checks": checks,
            },
            indent=2,
        )
    )
    if report["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
