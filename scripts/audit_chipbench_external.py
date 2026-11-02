#!/usr/bin/env python3
"""Freeze the ChiPBench external cohort before any candidate outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
from audit_chip_external_benchmarks import (
    count_macros_in_def,
    macro_masters,
)

from floorflow.chip_flow.def_graph import parse_def_graph
from floorflow.chip_flow.ppa_external import graph_edges

DEFAULT_CONFIG = ROOT / "configs/chip_external_holdout_chipbench_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/reports/chip_chipbench_holdout_manifest_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-pending-macro-count", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_design_files(root: Path, name: str) -> dict[str, Path]:
    design = root / "data" / name
    verilog = next(iter(sorted(design.glob("*.v"))), None)
    sdc = design / "constraint.sdc"
    pre_place = design / "def/pre_place.def"
    macro_placed = design / "def/macro_placed.def"
    lefs = sorted((design / "lef").glob("*.lef"))
    libs = sorted((design / "lib").glob("*.lib"))
    if verilog is None:
        raise ValueError(f"{name}: missing Verilog")
    files = {
        "verilog": verilog,
        "sdc": sdc,
        "pre_place_def": pre_place,
        "macro_placed_def": macro_placed,
    }
    for label, path in files.items():
        if not path.is_file():
            raise ValueError(f"{name}: missing {label}: {path}")
    if not lefs or not libs:
        raise ValueError(f"{name}: missing LEF or LIB")
    return {**files, "lefs": lefs, "libs": libs}


def audit_design(root: Path, name: str, family: str) -> dict[str, Any]:
    files = required_design_files(root, name)
    flow_config = (
        ROOT
        / "third_party/ChiPBench/flow/designs/nangate45"
        / name
        / "config.mk"
    )
    if not flow_config.is_file():
        raise ValueError(f"{name}: ChiPBench flow config is missing")
    lefs = files["lefs"]
    assert isinstance(lefs, list)
    macro_lefs = [
        path
        for path in lefs
        if not path.name.startswith("NangateOpenCellLibrary")
    ]
    macro_count = count_macros_in_def(
        files["macro_placed_def"],
        macro_masters(macro_lefs),
    )
    if macro_count <= 0:
        raise ValueError(f"{name}: macro_placed.def has no macro instances")
    evidence_paths = [
        files["verilog"],
        files["sdc"],
        files["pre_place_def"],
        files["macro_placed_def"],
        flow_config,
        *lefs,
        *files["libs"],
    ]
    graph = parse_def_graph(files["pre_place_def"], macro_lefs)
    edges = graph_edges(graph)
    return {
        "id": name,
        "family": family,
        "source": "ChiPBench-D",
        "design_dir": str((root / "data" / name).relative_to(ROOT)),
        "verilog": str(files["verilog"].relative_to(ROOT)),
        "sdc": str(files["sdc"].relative_to(ROOT)),
        "pre_place_def": str(files["pre_place_def"].relative_to(ROOT)),
        "reference_macro_def": str(files["macro_placed_def"].relative_to(ROOT)),
        "macro_lefs": [
            str(path.relative_to(ROOT))
            for path in lefs
            if not path.name.startswith("NangateOpenCellLibrary")
        ],
        "libs": [str(path.relative_to(ROOT)) for path in files["libs"]],
        "flow_config": str(flow_config.relative_to(ROOT)),
        "macro_instance_count": macro_count,
        "macro_count_status": "measured_macro_placed_def_vs_published_metadata",
        "graph_audit": {
            "status": "pass",
            "macros": len(graph.macros),
            "ports": len(graph.ports),
            "b2b_edges": len(edges["b2b"]),
            "p2b_edges": len(edges["p2b"]),
        },
        "files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in evidence_paths
        },
        "candidate_outcome_opened": False,
    }


def observed_design_names(path: Path) -> set[str]:
    payload = load(path)
    names = set()
    for row in payload["cases"]:
        key = row.get("design_key") or row.get("case")
        names.add(str(key).rsplit("/", 1)[-1])
    return names


def orf_supplement(config: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    config_path = ROOT / row["config"]
    platform, nickname = row["design_key"].split("/", 1)
    evidence_paths = [config_path]
    measured = None
    if row.get("macro_count_evidence"):
        evidence = ROOT / row["macro_count_evidence"]
        evidence_paths.append(evidence)
        text = evidence.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Number of macros:\s*(\d+)", text)
        measured = int(match.group(1)) if match else None
        if measured != int(row["macro_instance_count"]):
            raise ValueError(f"{row['design_key']} macro-count evidence mismatch")
    else:
        evidence_paths.extend(ROOT / path for path in row["macro_count_source"])
        if not all(path.is_file() for path in evidence_paths):
            raise ValueError(f"{row['design_key']} source evidence is missing")
    return {
        "id": row["design_key"],
        "platform": platform,
        "design_name": row.get("design_name", nickname),
        "design_nickname": row.get("design_nickname", nickname),
        "flow_config": row["config"],
        "family": config["selection"]["independent_families"][row["design_key"]],
        "source": "OpenROAD-flow-scripts",
        "macro_instance_count": measured,
        "macro_master_count": row.get("macro_master_count"),
        "macro_instance_count_static_estimate": row.get(
            "macro_instance_count_static_estimate"
        ),
        "static_estimate": row.get("static_estimate"),
        "static_estimate_rule": row.get("static_estimate_rule"),
        "preflight_report": row.get("preflight_report"),
        "pre_place_def": row.get("pre_place_def"),
        "macro_lefs": row.get("macro_lefs", []),
        "files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in evidence_paths
        },
        "macro_count_status": (
            "measured_floorplan_log"
            if measured is not None
            else "static_macro_capable_floorplan_measurement_pending"
        ),
        "candidate_outcome_opened": False,
    }


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    config = load(args.config)
    root = ROOT / config["source"]["dataset_local_root"]
    names = config["selection"]["chipbench_designs"]
    families = config["selection"]["independent_families"]
    expected_counts = config["selection"]["expected_macro_counts"]
    observed = observed_design_names(ROOT / config["selection"]["observed_v45_manifest"])
    overlap = sorted(set(names) & observed)
    if overlap:
        raise ValueError(f"ChiPBench holdout overlaps observed v45 designs: {overlap}")
    rows = [audit_design(root, name, families[name]) for name in names]
    rows.extend(
        orf_supplement(config, row)
        for row in config["selection"]["orf_supplements"]
    )
    expected = int(config["selection"]["required_total_designs"])
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise ValueError("external cohort count or uniqueness gate failed")
    count_mismatches = {
        row["id"]: {
            "expected": expected_counts[row["id"]],
            "observed": row["macro_instance_count"],
        }
        for row in rows
        if (
            row["macro_instance_count"] is not None
            and int(row["macro_instance_count"]) != int(expected_counts[row["id"]])
        )
    }
    if count_mismatches:
        raise ValueError(f"macro-count validation failed: {count_mismatches}")
    adapter_smoke = set(config["evaluation_contract"]["adapter_smoke_cases"])
    for row in rows:
        row["adapter_smoke_accessed"] = row["id"] in adapter_smoke
        row["routed_or_scored_outcome_opened"] = False
    license_path = ROOT / "third_party/ChiPBench/LICENSE"
    dataset_readme = root / "README.md"
    metadata = root / "chipbench_meta_data.json"
    all_checks = {
        "minimum_15_designs": len(rows) >= expected,
        "unique_design_ids": len({row["id"] for row in rows}) == len(rows),
        "no_observed_design_overlap": not overlap,
        "all_macro_counts_positive": all(
            row["macro_instance_count"] is None
            or int(row["macro_instance_count"]) > 0
            for row in rows
        ),
        "all_macro_counts_verified": all(
            row["macro_instance_count"] is not None
            and int(row["macro_instance_count"]) > 0
            for row in rows
        ),
        "macro_counts_match_frozen_source": not count_mismatches,
        "candidate_outcomes_unopened": all(
            row["candidate_outcome_opened"] is False for row in rows
        ),
        "chipbench_graph_audits_pass": all(
            row.get("graph_audit", {}).get("status") == "pass"
            for row in rows
            if row["source"] == "ChiPBench-D"
        ),
        "chipbench_flow_configs_present": all(
            (ROOT / row["flow_config"]).is_file()
            for row in rows
            if row["source"] == "ChiPBench-D"
        ),
    }
    pending_only = not all_checks["all_macro_counts_verified"] and all(
        value
        for key, value in all_checks.items()
        if key != "all_macro_counts_verified"
    )
    if not all(all_checks.values()) and not (
        args.allow_pending_macro_count and pending_only
    ):
        raise ValueError(f"external holdout gate failed: {all_checks}")
    report = {
        "version": "chip-chipbench-holdout-manifest-v1",
        "status": (
            "frozen_before_candidate_outcome"
            if all(all_checks.values())
            else "frozen_before_candidate_outcome_macro_preflight_pending"
        ),
        "config": str(args.config.relative_to(ROOT)),
        "source": config["source"],
        "licenses": {
            "repository_license": str(license_path.relative_to(ROOT)),
            "repository_license_sha256": sha256(license_path),
            "dataset_readme_sha256": sha256(dataset_readme),
            "croissant_metadata_sha256": sha256(metadata),
            "repository_spdx": "BSD-3-Clause",
            "dataset_card_spdx": "BSD-3-Clause",
            "croissant_spdx": "MIT",
            "decision": (
                "Both documented licenses are permissive; retain the metadata "
                "conflict in provenance and do not omit it from the final archive."
            ),
        },
        "observed_design_names": sorted(observed),
        "overlap_with_observed_designs": overlap,
        "design_count": len(rows),
        "family_count": len({row["family"] for row in rows}),
        "designs": rows,
        "checks": all_checks,
        "adapter_smoke_cases": sorted(adapter_smoke),
        "adapter_smoke_boundary": config["evaluation_contract"][
            "adapter_smoke_boundary"
        ],
        "claim_boundary": (
            "A frozen supply manifest is not external PPA evidence. Three cases "
            "were accessed only for adapter plumbing without scores or routed "
            "outcomes; quality outcomes remain unopened."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report["checks"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
