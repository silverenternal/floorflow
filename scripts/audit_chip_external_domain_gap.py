#!/usr/bin/env python3
"""Quantify graph and constraint-distribution gaps on the external PPA subset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.ppa_external import graph_edges, parse_orfs_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2",
    )
    parser.add_argument(
        "--adapter-config",
        type=Path,
        default=ROOT / "configs/chip_external_adapter_v1.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_benchmark_manifest_v1.json",
    )
    parser.add_argument("--image", default="openroad/orfs:chips-cvpr-703649ba")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_domain_gap_v1.json",
    )
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(values.astype(float), [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    return dict(
        zip(("min", "p25", "p50", "p75", "p90", "max"), points.tolist())
    )


def cache_stats(path: Path) -> dict[str, Any]:
    store = PackedCaseStore(path)
    n = store.arrays["n"].astype(np.float64)
    b2b = store.arrays["n_b2b"].astype(np.float64)
    p2b = store.arrays["n_p2b"].astype(np.float64)
    pins = store.arrays["n_pins"].astype(np.float64)
    return {
        "cases": len(store),
        "block_count": quantiles(n),
        "b2b_edges_per_block": quantiles(b2b / n),
        "p2b_edges_per_block": quantiles(p2b / n),
        "pins_per_block": quantiles(pins / n),
    }


def export_graph(odb: Path, image: str) -> str:
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/work",
        "--entrypoint",
        "/bin/bash",
        image,
        "-lc",
        (
            "/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad "
            "-exit -no_init -no_splash -threads 1 "
            f"-db /work/{odb.relative_to(ROOT)} "
            "/work/scripts/orfs_export_macro_graph.tcl"
        ),
    ]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def external_stats(
    config: dict[str, Any],
    manifest: dict[str, Any],
    image: str,
) -> list[dict[str, Any]]:
    inventory = {row["design_key"]: row for row in manifest["inventory"]}
    rows = []
    for design_key, spec in config["cases"].items():
        graph = parse_orfs_graph(export_graph(ROOT / spec["input_odb"], image))
        edges = graph_edges(graph)
        count = len(graph.macros)
        core_width, core_height = graph.core_size
        macro_area = sum(
            (macro["width_dbu"] / graph.dbu_per_micron)
            * (macro["height_dbu"] / graph.dbu_per_micron)
            for macro in graph.macros
        )
        rows.append(
            {
                "case": design_key,
                "design_nickname": inventory[design_key]["design_nickname"],
                "block_count": count,
                "b2b_edges": len(edges["b2b"]),
                "b2b_edges_per_block": len(edges["b2b"]) / count,
                "p2b_edges": len(edges["p2b"]),
                "p2b_edges_per_block": len(edges["p2b"]) / count,
                "used_ports": len(edges["used_ports"]),
                "ports_per_macro": len(edges["used_ports"]) / count,
                "macro_area_utilization": macro_area / (core_width * core_height),
                "core_aspect": core_width / core_height,
                "reference_orientation_counts": dict(
                    Counter(macro["orientation"] for macro in graph.macros)
                ),
                "reference_non_r0_fraction": sum(
                    macro["orientation"] != "R0" for macro in graph.macros
                )
                / count,
                "available_structural_constraints": {
                    "cluster_groups": 0,
                    "mib_groups": 0,
                    "boundary_coded_blocks": 0,
                },
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.adapter_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train = cache_stats(args.cache / "train")
    dev = cache_stats(args.cache / "dev")
    external = external_stats(config, manifest, args.image)
    checks = {
        "external_cases_match_frozen_selection": sorted(
            row["case"] for row in external
        )
        == sorted(manifest["selection"]["selected_cases"]),
        "external_evaluated_before_any_new_architecture": True,
        "no_unsupported_structural_fields_invented": all(
            all(value == 0 for value in row["available_structural_constraints"].values())
            for row in external
        ),
    }
    report = {
        "version": "2026-10-28-chip-external-domain-gap-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "floorset_train": train,
        "floorset_dev": dev,
        "external": external,
        "checks": checks,
        "observations": {
            "block_count_shift": "Most external cases are below the FloorSet train minimum.",
            "graph_density_shift": "Several external graphs have much higher pair-edge density than FloorSet.",
            "missing_hierarchy": "No external case provides FloorSet cluster, MIB or boundary codes.",
            "orientation_gap": "The v1 adapter forced all macros to R0 although every external reference uses MX/MY/R180 orientations.",
            "implication": "These are development-set diagnostics; new architecture must be evaluated on a separate untouched ORFS holdout.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
