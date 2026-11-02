#!/usr/bin/env python3
"""Run a predeclared matched-flow-budget search with analytical feedback."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.refiner_config import append_refiner_protocol
from scripts.run_chip_def_adapter_suite import case_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_v46_analytical_search.json",
    )
    parser.add_argument(
        "--placement-config",
        type=Path,
        default=ROOT / "configs/chip_external_holdout_v45.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_inventory_v45_expanded_audit.json",
    )
    parser.add_argument("--arm", choices=("candidate", "control"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--proposal-count", type=int, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-timeout-seconds", type=int, default=7200)
    parser.add_argument("--response-timeout-seconds", type=int, default=3600)
    parser.add_argument("--response-num-cores", type=int, default=8)
    parser.add_argument("--response-seed-variant")
    parser.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-shelf", action="store_true")
    parser.add_argument("--position-aware-shelf", action="store_true")
    parser.add_argument("--snapshot-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--generate-only", action="store_true")
    phases.add_argument("--response-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_search_pilot.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str], timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "exit_code": result.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def case_config(
    placement_config: dict[str, Any],
    design_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case = placement_config["cases"][design_key]
    platform, nickname = design_key.split("/", maxsplit=1)
    slugged = {
        **case,
        "design_key": design_key,
        "platform": platform,
        "design_nickname": nickname,
    }
    return slugged, placement_config["primary_protocol"]


def generation_command(
    args: argparse.Namespace,
    case: dict[str, Any],
    policy: dict[str, Any],
    *,
    protocol_tag: str,
    proposal: int,
    seed: int,
    nfe: int,
    refiner_steps: int,
    output: Path,
    force_shelf: bool,
    position_aware_shelf: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_chip_external_placement.py"),
        "--case",
        case_slug(case),
        "--checkpoint",
        str(args.checkpoint),
        "--arm",
        f"proposal{proposal}",
        "--seed",
        str(seed),
        "--odb",
        str(ROOT / case["input_odb"]),
        "--device",
        args.device,
        "--nfe",
        str(nfe),
        "--refiner-steps",
        str(refiner_steps),
        "--projector-iterations",
        str(policy["candidate_projector_iterations"]),
        "--macro-halo-um",
        str(case["macro_halo_um"]),
        "--legalizer-relation",
        str(policy.get("legalizer_relation", "restored")),
        "--protocol-tag",
        protocol_tag,
        "--output",
        str(output),
    ]
    append_refiner_protocol(command, policy)
    if policy.get("instance_hierarchy"):
        command.append("--instance-hierarchy")
    if policy.get("preserve_reference_orientation"):
        command.append("--preserve-reference-orientation")
    if force_shelf:
        command.append("--force-shelf")
    if position_aware_shelf:
        command.append("--position-aware-shelf")
    return command


def response_command(
    args: argparse.Namespace,
    *,
    case: dict[str, Any],
    variant: str,
    placement_tcl: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_chip_analytical_response.py"),
        "--case",
        case_slug(case),
        "--manifest",
        str(args.manifest),
        "--variant",
        variant,
        "--macro-placement-tcl",
        str(placement_tcl),
        "--run-flow",
        "--timeout-seconds",
        str(args.response_timeout_seconds),
        "--num-cores",
        str(args.response_num_cores),
        *(
            ["--seed-variant", args.response_seed_variant]
            if args.response_seed_variant
            else []
        ),
        "--output",
        str(output),
    ]


def select(
    rows: list[dict[str, Any]],
    score_key: str,
) -> dict[str, Any]:
    selected = min(rows, key=lambda row: row[score_key])
    return {
        "proposal": selected["proposal"],
        "seed": selected["seed"],
        "score": selected[score_key],
        "placement_tcl": selected["placement_tcl"],
        "placement_report": selected["placement_report"],
        "analytical_report": selected["analytical_report"],
    }


def main() -> None:
    args = parse_args()
    config = load(args.config)
    placement_config = load(args.placement_config)
    protocol = config["protocol"]
    if args.response_seed_variant is None:
        args.response_seed_variant = protocol.get("response_seed_variant")
    proposal_count = int(args.proposal_count)
    if proposal_count not in protocol["proposal_counts"]:
        raise ValueError("proposal count is not predeclared")
    total_nfe = int(protocol["total_nfe"])
    total_refiner_steps = int(protocol["total_refiner_steps"])
    if total_nfe % proposal_count or total_refiner_steps % proposal_count:
        raise ValueError("total NFE and refiner budgets must divide proposal count")
    nfe = total_nfe // proposal_count
    refiner_steps = total_refiner_steps // proposal_count
    designs = args.case or list(config["cases"])
    protocol_tag = f"v46_analytical_{args.arm}_n{proposal_count}"
    results = []
    for design_key in designs:
        case, policy = case_config(placement_config, design_key)
        slug = case_slug(case)
        proposals = []
        for proposal in range(proposal_count):
            seed = (
                int(protocol["inference_seed"])
                + proposal * int(protocol["proposal_seed_stride"])
            )
            stem = (
                ROOT
                / "outputs/chip_ppa"
                / f"placements_{protocol_tag}"
                / slug
                / f"proposal{proposal}_seed{seed}"
            )
            placement_report = stem.with_suffix(".json")
            placement_tcl = stem.with_suffix(".tcl")
            trace_path = placement_report.with_name(
                f"{placement_report.stem}.trace.json"
            )
            variant = f"ppa_{protocol_tag}_p{proposal}_s{seed}"
            analytical_report = (
                ROOT
                / "outputs/reports/chip_v46_analytical"
                / f"{slug}_{protocol_tag}_p{proposal}_s{seed}.json"
            )
            previous = load(placement_report) if placement_report.is_file() else {}
            if args.snapshot_existing:
                for path in (
                    placement_report,
                    placement_tcl,
                    trace_path,
                    analytical_report,
                ):
                    if not path.is_file():
                        continue
                    snapshot = path.with_name(f"{path.stem}_snapshot{path.suffix}")
                    if not snapshot.exists():
                        shutil.copy2(path, snapshot)
            force_regeneration = (
                args.force_shelf
                and previous.get("status") == "pass"
                and previous.get("legalizer_route") != "shelf_pack_forced"
            )
            initial_artifacts = []
            if force_regeneration:
                for path in (
                    placement_report,
                    placement_tcl,
                    trace_path,
                    analytical_report,
                ):
                    if not path.is_file():
                        continue
                    initial_path = path.with_name(
                        f"{path.stem}_initial{path.suffix}"
                    )
                    if not initial_path.exists():
                        shutil.copy2(path, initial_path)
                    initial_artifacts.append(str(initial_path.relative_to(ROOT)))
                if analytical_report.is_file():
                    analytical_report.unlink()
            generate = generation_command(
                args,
                case,
                policy,
                protocol_tag=protocol_tag,
                proposal=proposal,
                seed=seed,
                nfe=nfe,
                refiner_steps=refiner_steps,
                output=placement_report,
                force_shelf=args.force_shelf,
                position_aware_shelf=args.position_aware_shelf,
            )
            response = response_command(
                args,
                case=case,
                variant=variant,
                placement_tcl=placement_tcl,
                output=analytical_report,
            )
            generation_run = None
            response_run = None
            if not args.dry_run:
                generation_ready = (
                    args.reuse
                    and not force_regeneration
                    and placement_report.is_file()
                    and load(placement_report).get("status") == "pass"
                )
                if not generation_ready:
                    generation_run = run_command(
                        generate,
                        args.generation_timeout_seconds,
                    )
                    generation_ready = (
                        placement_report.is_file()
                        and load(placement_report).get("status") == "pass"
                    )
                response_ready = (
                    args.reuse
                    and analytical_report.is_file()
                    and load(analytical_report).get("status") == "pass"
                )
                if (
                    generation_ready
                    and not response_ready
                    and not args.generate_only
                ):
                    response_run = run_command(
                        response,
                        args.response_timeout_seconds,
                    )
            placement = load(placement_report) if placement_report.is_file() else {}
            analytical = (
                load(analytical_report) if analytical_report.is_file() else {}
            )
            proposal_rows = {
                "proposal": proposal,
                "seed": seed,
                "nfe": nfe,
                "refiner_steps": refiner_steps,
                "placement_report": str(placement_report.relative_to(ROOT)),
                "placement_tcl": str(placement_tcl.relative_to(ROOT)),
                "analytical_report": str(analytical_report.relative_to(ROOT)),
                "forced_shelf_regeneration": force_regeneration,
                "forced_shelf_initial_artifacts": initial_artifacts or None,
                "initial_legalizer_route": (
                    previous.get("legalizer_route") if force_regeneration else None
                ),
                "legalizer_route": placement.get("legalizer_route"),
                "p2b_hpwl_um": placement.get("connectivity", {})
                .get("legalized_output", {})
                .get("p2b_hpwl_um"),
                "globalplace_wirelength": analytical.get("response", {}).get(
                    "score"
                ),
                "globalplace_seconds": analytical.get("flow_run", {}).get(
                    "elapsed_seconds"
                )
                if analytical.get("flow_run")
                else None,
                "generation_run": generation_run,
                "response_run": response_run,
                "generation_command": generate,
                "response_command": response,
            }
            proposals.append(proposal_rows)
        common_route_enforced = False
        common_route_changes = []
        initial_routes = {
            row["legalizer_route"]
            for row in proposals
            if row["legalizer_route"] is not None
        }
        if (
            not args.dry_run
            and len(initial_routes) > 1
            and all(row["p2b_hpwl_um"] is not None for row in proposals)
        ):
            if args.response_only:
                raise ValueError(
                    "mixed legalizer routes require regeneration before response-only scoring"
                )
            common_route_enforced = True
            for proposal, row in enumerate(proposals):
                placement_report = ROOT / row["placement_report"]
                placement_tcl = ROOT / row["placement_tcl"]
                trace_path = placement_report.with_name(
                    f"{placement_report.stem}.trace.json"
                )
                analytical_report = ROOT / row["analytical_report"]
                initial_paths = []
                for path in (
                    placement_report,
                    placement_tcl,
                    trace_path,
                    analytical_report,
                ):
                    if not path.is_file():
                        continue
                    initial_path = path.with_name(
                        f"{path.stem}_initial{path.suffix}"
                    )
                    if not initial_path.exists():
                        shutil.copy2(path, initial_path)
                    initial_paths.append(str(initial_path.relative_to(ROOT)))
                if analytical_report.is_file():
                    analytical_report.unlink()
                rerun = generation_command(
                    args,
                    case,
                    policy,
                    protocol_tag=protocol_tag,
                    proposal=proposal,
                    seed=row["seed"],
                    nfe=nfe,
                    refiner_steps=refiner_steps,
                    output=placement_report,
                    force_shelf=True,
                    position_aware_shelf=args.position_aware_shelf,
                )
                row["initial_legalizer_route"] = row["legalizer_route"]
                row["initial_artifacts"] = initial_paths
                row["common_route_generation_run"] = run_command(
                    rerun,
                    args.generation_timeout_seconds,
                )
                row["generation_command"] = rerun
                updated = (
                    load(placement_report) if placement_report.is_file() else {}
                )
                row["legalizer_route"] = updated.get("legalizer_route")
                row["p2b_hpwl_um"] = (
                    updated.get("connectivity", {})
                    .get("legalized_output", {})
                    .get("p2b_hpwl_um")
                )
                if (
                    updated.get("status") == "pass"
                    and not args.generate_only
                ):
                    row["response_run"] = run_command(
                        response_command(
                            args,
                            case=case,
                            variant=f"ppa_{protocol_tag}_p{proposal}_s{row['seed']}",
                            placement_tcl=placement_tcl,
                            output=analytical_report,
                        ),
                        args.response_timeout_seconds,
                    )
                common_route_changes.append(
                    {
                        "proposal": proposal,
                        "initial_route": row["initial_legalizer_route"],
                        "final_route": row["legalizer_route"],
                    }
                )
                analytical = (
                    load(analytical_report) if analytical_report.is_file() else {}
                )
                row["globalplace_wirelength"] = analytical.get("response", {}).get(
                    "score"
                )
                row["globalplace_seconds"] = (
                    analytical.get("flow_run", {}).get("elapsed_seconds")
                    if analytical.get("flow_run")
                    else None
                )
        generation_complete = all(
            row["p2b_hpwl_um"] is not None for row in proposals
        )
        complete = all(
            row["p2b_hpwl_um"] is not None
            and row["globalplace_wirelength"] is not None
            for row in proposals
        )
        if args.dry_run:
            status = "dry_run"
        elif args.generate_only:
            status = "generated" if generation_complete else "incomplete"
        else:
            status = "complete" if complete else "incomplete"
        results.append(
            {
                "case": design_key,
                "proposals": proposals,
                "status": status,
                "initial_legalizer_routes": sorted(initial_routes),
                "common_route_enforced": common_route_enforced,
                "common_route_changes": common_route_changes,
                "analytical_selection": select(
                    proposals,
                    "globalplace_wirelength",
                )
                if complete
                else None,
                "p2b_selection": select(proposals, "p2b_hpwl_um")
                if complete
                else None,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "version": "chip-v46-analytical-search-pilot-v1",
                    "status": "running",
                    "arm": args.arm,
                    "proposal_count": proposal_count,
                    "total_nfe": total_nfe,
                    "total_refiner_steps": total_refiner_steps,
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    expected_status = (
        "dry_run"
        if args.dry_run
        else ("generated" if args.generate_only else "complete")
    )
    report = {
        "version": "chip-v46-analytical-search-pilot-v1",
        "status": (
            expected_status
            if all(row["status"] == expected_status for row in results)
            else "incomplete"
        ),
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "proposal_count": proposal_count,
        "total_nfe": total_nfe,
        "total_refiner_steps": total_refiner_steps,
        "selection_policy": config["selection"],
        "results": results,
        "boundary": config["scope"],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "arm": args.arm,
                "proposal_count": proposal_count,
                "cases": [row["status"] for row in results],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if report["status"] == "incomplete":
        sys.exit(1)


if __name__ == "__main__":
    main()
