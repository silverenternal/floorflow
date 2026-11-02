#!/usr/bin/env python3
"""Assemble the bounded typed-hierarchy CVPR experiment package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_chip_cvpr_package import (
    digest,
    python_file_sizes,
    read,
    write,
)

EVIDENCE = (
    "outputs/reports/chip_pure_cvpr_claim_v1.json",
    "outputs/reports/chip_typed_hierarchy_three_seed_v3_ema.json",
    "outputs/reports/chip_typed_hierarchy_three_seed_v2.json",
    "outputs/reports/chip_typed_hierarchy_ablation_v3.json",
    "outputs/reports/chip_typed_hierarchy_promoted_controls_v1.json",
    "outputs/reports/chip_typed_hierarchy_bbox_repair_5k_v1.json",
    "outputs/reports/chip_typed_hierarchy_bbox_repair_20k_v1.json",
    "outputs/reports/chip_typed_hierarchy_bbox_anneal_20k_v1.json",
    "outputs/reports/chip_typed_hierarchy_diagnostics_v2.json",
    "outputs/reports/chip_typed_hierarchy_screen_5k_v2.json",
    "outputs/reports/chip_hierarchical_integration_cpu_v1.json",
    "outputs/reports/chip_trinity_d192_component_gap_v1.json",
    "outputs/reports/chip_pytest_v6.xml",
)
SOURCE_FILES = (
    "chipstodo.json",
    "docs/chip_floorplanning_concepts.md",
    "src/floorflow/chip_flow/hierarchical_flow.py",
    "src/floorflow/chip_flow/hierarchical_flow_runtime.py",
    "src/floorflow/chip_flow/typed_graph_encoder.py",
    "src/floorflow/chip_flow/hard_constraint_projector.py",
    "src/floorflow/chip_flow/legalization.py",
    "src/floorflow/chip_flow/flow_adapter.py",
    "src/floorflow/chip_flow/train_cli.py",
    "src/floorflow/chip_flow/train_objective.py",
    "src/floorflow/chip_flow/checkpoint_loading.py",
    "scripts/train_chip_slot.py",
    "scripts/evaluate_chip_official.py",
    "scripts/evaluate_chip_refiner.py",
    "scripts/analyze_chip_typed_hierarchy_diagnostics.py",
    "scripts/build_chip_bbox_repair_report.py",
    "scripts/build_chip_hierarchy_ablation_report.py",
    "scripts/build_chip_promoted_control_report.py",
    "scripts/build_chip_hierarchy_three_seed_report.py",
    "scripts/build_chip_typed_hierarchy_final_figure.py",
    "scripts/build_chip_typed_hierarchy_claim.py",
    "tests/test_chip_hierarchical_flow.py",
    "tests/test_chip_hierarchical_integration.py",
    "tests/test_chip_typed_graph_encoder.py",
    "tests/test_chip_hard_constraint_projector.py",
    "tests/test_chip_legalization_reproducibility.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/cvpr_package_v14_chips",
    )
    return parser.parse_args()


def pytest_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError(f"{path} has no testsuite")
    return {
        key: int(suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    }


def budget_table() -> dict:
    data = ROOT / "outputs/chip_hierarchy/v10_three_seed/reproducible_v1"
    candidate_nfe = read(data / "candidate_seed19_ema_nfe.json")["results"]
    baseline_nfe = read(data / "baseline_seed19_ema_nfe.json")["results"]
    candidate_refiner = read(
        data / "candidate_seed19_ema_refiner_short.json"
    )["results"]
    candidate_refiner["nfe4_refine400"] = read(
        data / "candidate_seed19_ema_primary.json"
    )["results"]["nfe4_refine400"]
    baseline_primary = read(
        data / "baseline_seed19_ema_primary.json"
    )["results"]["nfe4_refine400"]
    return {
        "candidate_nfe": candidate_nfe,
        "control_nfe": baseline_nfe,
        "candidate_refiner": candidate_refiner,
        "control_refiner400": baseline_primary,
        "equal_quality": {
            "candidate_nfe4_cost": candidate_nfe["nfe4_refine0"][
                "legalized_hard_cost"
            ],
            "control_nfe8_cost": baseline_nfe["nfe8_refine0"][
                "legalized_hard_cost"
            ],
            "candidate_nfe4_v_rel": candidate_nfe["nfe4_refine0"][
                "legalized_v_rel"
            ],
            "control_nfe8_v_rel": baseline_nfe["nfe8_refine0"][
                "legalized_v_rel"
            ],
            "supported": (
                candidate_nfe["nfe4_refine0"]["legalized_hard_cost"]
                <= baseline_nfe["nfe8_refine0"]["legalized_hard_cost"]
                and candidate_nfe["nfe4_refine0"]["legalized_v_rel"]
                <= baseline_nfe["nfe8_refine0"]["legalized_v_rel"]
            ),
        },
        "refiner_boundary": (
            "At seed 19, candidate 400-step refinement remains slightly "
            "behind the control; no lower-refiner equal-quality claim is made."
        ),
    }


def main() -> None:
    args = parse_args()
    evidence_paths = [ROOT / relative for relative in EVIDENCE]
    missing = [
        str(path.relative_to(ROOT))
        for path in evidence_paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(missing)
    claim = read(ROOT / EVIDENCE[0])
    ema = read(ROOT / EVIDENCE[1])
    raw = read(ROOT / EVIDENCE[2])
    ablation = read(ROOT / EVIDENCE[3])
    promoted = read(ROOT / EVIDENCE[4])
    tests = pytest_summary(ROOT / EVIDENCE[-1])
    sizes = python_file_sizes()
    oversized = {
        relative: lines
        for relative, lines in sizes.items()
        if lines >= 600
    }
    checks = {
        "claim_bounded_positive": (
            claim["status"] == "bounded_positive_ready_for_manuscript"
        ),
        "ema_three_seed_pass": ema["status"] == "pass",
        "raw_three_seed_negative_retained": (
            raw["status"] == "mixed_or_rejected"
        ),
        "ablation_pass": (
            ablation["status"] == "pass_with_hierarchy_boundary"
        ),
        "promoted_controls_pass": (
            promoted["status"] == "pass_with_reported_tradeoffs"
        ),
        "all_seed_legality": ema["checks"]["all_seeds_legalized_feasible"],
        "nfe_equal_quality_supported": budget_table()["equal_quality"][
            "supported"
        ],
        "full_tests_pass": (
            tests["failures"] == 0 and tests["errors"] == 0
        ),
        "no_python_file_at_or_over_600": not oversized,
        "final_figure_present": all(
            (
                ROOT
                / f"outputs/figures/chip_typed_hierarchy_final_pareto_v1.{suffix}"
            ).is_file()
            for suffix in ("pdf", "svg")
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write(args.out_dir / "tables/main_results.json", ema)
    write(args.out_dir / "tables/raw_negative_results.json", raw)
    write(
        args.out_dir / "tables/mechanism_and_ablation.json",
        {"ablation": ablation, "promoted_controls": promoted},
    )
    write(args.out_dir / "tables/budget.json", budget_table())
    write(
        args.out_dir / "tables/claim_and_boundaries.json",
        {
            "main_claim": claim["main_claim"],
            "boundaries": claim["boundaries"],
            "external_anchor": claim["external_anchor"],
            "statistical_boundary": claim["statistical_boundary"],
        },
    )
    figure = "outputs/figures/chip_typed_hierarchy_final_pareto_v1"
    for suffix in ("pdf", "svg", "png"):
        target = args.out_dir / "figures" / f"chip_typed_hierarchy_final_pareto_v1.{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f"{figure}.{suffix}", target)
    source_out = args.out_dir / "source"
    source_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "chipstodo.json", source_out / "chipstodo.json")
    shutil.copy2(
        ROOT / "docs/chip_floorplanning_concepts.md",
        source_out / "chip_floorplanning_concepts.md",
    )
    manifest = {
        "root": str(ROOT),
        "evidence_sha256": {
            relative: digest(ROOT / relative) for relative in EVIDENCE
        },
        "source_sha256": {
            relative: digest(ROOT / relative) for relative in SOURCE_FILES
        },
        "figures_sha256": {
            f"figures/chip_typed_hierarchy_final_pareto_v1.{suffix}": digest(
                args.out_dir
                / "figures"
                / f"chip_typed_hierarchy_final_pareto_v1.{suffix}"
            )
            for suffix in ("pdf", "svg", "png")
        },
        "pytest": {
            "path": EVIDENCE[-1],
            "sha256": digest(ROOT / EVIDENCE[-1]),
            "summary": tests,
        },
        "max_python_file_lines": max(sizes.values()),
        "python_files_600_or_more": oversized,
    }
    write(args.out_dir / "reproducibility_manifest.json", manifest)
    checklist = {
        "status": "ready" if all(checks.values()) else "blocked",
        "scope": "CVPR experiment evidence package; no manuscript text",
        "checks": checks,
        "main_claim": claim["main_claim"],
        "primary_table": "tables/main_results.json",
        "mechanism_ablation": "tables/mechanism_and_ablation.json",
        "budget": "tables/budget.json",
        "claim_boundaries": "tables/claim_and_boundaries.json",
        "negative_results": "tables/raw_negative_results.json",
        "figure": "figures/chip_typed_hierarchy_final_pareto_v1.svg/.pdf",
        "required_reporting": [
            "Report raw and EMA outcomes separately.",
            "Report pooled intervals that cross zero and per-seed heterogeneity.",
            "Report Trinity d192 remains stronger.",
            "Report no opening, port-direction, halo, MIB-separation or SOTA claim.",
            "Report fixed-weight bbox supervision was rejected at 20k.",
            "Report no-hierarchy has lower grouping error.",
            "Report candidate NFE4 equal-quality evidence and 400-step refiner boundary.",
        ],
    }
    write(args.out_dir / "submission_checklist.json", checklist)
    archive = args.out_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(args.out_dir, arcname=args.out_dir.name)
    print(json.dumps({"status": checklist["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
