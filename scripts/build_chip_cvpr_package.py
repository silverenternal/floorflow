#!/usr/bin/env python3
"""Assemble the reproducible chip CVPR evidence package without writing a paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    "outputs/reports/chip_final_three_seed_evidence_v1.json",
    "outputs/reports/chip_graph_mechanism_ablation_v1.json",
    "outputs/reports/chip_output_parameterization_and_dropout_v1.json",
    "outputs/reports/chip_range_warm_restart_v1.json",
    "outputs/reports/chip_graph_slot_20k_v1.json",
    "outputs/reports/chip_slot_predictability_probe_v1.json",
    "outputs/reports/chip_training_framework_benchmark_v1.json",
    "outputs/reports/trinity_d192_matched_protocol_v2.json",
    "outputs/reports/two_domain_dynamic_hierarchy_v1.json",
    "outputs/reports/chip_final_claim_gate_v1.json",
    "outputs/reports/chip_global_representation_probe_v1.json",
    "outputs/reports/chip_global_representation_stats_v1.json",
    "outputs/reports/chip_global_causality_probe_v1.json",
    "outputs/reports/chip_global_pool_stats_screen_v1.json",
    "outputs/reports/chip_global_representation_film_v1.json",
    "outputs/reports/chip_global_causality_film_v1.json",
    "outputs/reports/chip_global_film_screen_v1.json",
    "outputs/reports/chip_path_consistency_screen_v1.json",
    "outputs/reports/chip_refiner_target_probe_v1.json",
    "outputs/reports/chip_inference_ensemble_screen_v1.json",
)
SOURCE_FILES = (
    "chipstodo.json",
    "docs/chip_cvpr_status_20261027.md",
    "docs/chip_training_runtime_v1.md",
    "src/floorflow/chip_flow/slot_model.py",
    "src/floorflow/chip_flow/graph_conditioning.py",
    "src/floorflow/chip_flow/projections.py",
    "src/floorflow/chip_flow/training_runtime.py",
    "src/floorflow/chip_flow/train_cli.py",
    "src/floorflow/chip_flow/train_objective.py",
    "src/floorflow/chip_flow/train_evaluation.py",
    "src/floorflow/chip_flow/training_runtime.py",
    "scripts/train_chip_slot.py",
    "scripts/evaluate_chip_official.py",
    "scripts/evaluate_chip_refiner.py",
    "scripts/build_chip_final_seed_report.py",
    "scripts/build_chip_final_claim_gate.py",
    "scripts/build_chip_graph_mechanism_report.py",
    "scripts/build_chip_architecture_report.py",
    "scripts/build_chip_range_adaptation_report.py",
    "scripts/audit_chip_global_representation.py",
    "scripts/audit_chip_global_causality.py",
    "scripts/audit_chip_refiner_target.py",
    "scripts/build_chip_global_rescue_report.py",
    "scripts/build_chip_global_film_report.py",
    "scripts/build_chip_path_consistency_report.py",
    "scripts/build_chip_ensemble_report.py",
    "scripts/build_chip_cvpr_package.py",
    "tests/test_chip_graph_conditioning.py",
    "tests/test_chip_train_objective.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/cvpr_package_v10_chips",
    )
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def python_file_sizes() -> dict[str, int]:
    sizes = {}
    for root in ("src", "scripts", "tests"):
        for path in (ROOT / root).rglob("*.py"):
            sizes[str(path.relative_to(ROOT))] = sum(1 for _ in path.open())
    return sizes


def main() -> None:
    args = parse_args()
    evidence_paths = [ROOT / relative for relative in EVIDENCE]
    missing = [
        str(path.relative_to(ROOT))
        for path in evidence_paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing evidence: {missing}")
    final = read(ROOT / EVIDENCE[0])
    mechanism = read(ROOT / EVIDENCE[1])
    architecture = read(ROOT / EVIDENCE[2])
    range_result = read(ROOT / EVIDENCE[3])
    screen = read(ROOT / EVIDENCE[4])
    runtime = read(ROOT / EVIDENCE[6])
    external = read(ROOT / EVIDENCE[7])
    hierarchy = read(ROOT / EVIDENCE[8])
    pytest_path = ROOT / "outputs/reports/chip_pytest_v1.xml"
    if not pytest_path.is_file():
        raise FileNotFoundError(pytest_path)
    pytest_root = ET.parse(pytest_path).getroot()
    pytest_suite = (
        pytest_root
        if pytest_root.tag == "testsuite"
        else pytest_root.find("testsuite")
    )
    if pytest_suite is None:
        raise ValueError("pytest report has no testsuite element")
    pytest_summary = {
        key: int(pytest_suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    }
    chip_pytest_path = ROOT / "outputs/reports/chip_chip_tests_v1.xml"
    if not chip_pytest_path.is_file():
        raise FileNotFoundError(chip_pytest_path)
    chip_pytest_root = ET.parse(chip_pytest_path).getroot()
    chip_pytest_suite = (
        chip_pytest_root
        if chip_pytest_root.tag == "testsuite"
        else chip_pytest_root.find("testsuite")
    )
    if chip_pytest_suite is None:
        raise ValueError("chip pytest report has no testsuite element")
    chip_pytest_summary = {
        key: int(chip_pytest_suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    }
    sizes = python_file_sizes()
    oversized = {
        path: lines for path, lines in sizes.items() if lines > 600
    }
    manifest = {
        "root": str(ROOT),
        "evidence": {
            str(path.relative_to(ROOT)): digest(path)
            for path in evidence_paths
        },
        "sources": {
            relative: digest(ROOT / relative)
            for relative in dict.fromkeys(SOURCE_FILES)
        },
        "pytest_report": {
            "path": str(pytest_path.relative_to(ROOT)),
            "sha256": digest(pytest_path),
            "summary": pytest_summary,
        },
        "chip_pytest_report": {
            "path": str(chip_pytest_path.relative_to(ROOT)),
            "sha256": digest(chip_pytest_path),
            "summary": chip_pytest_summary,
        },
        "python_file_max_lines": max(sizes.values()),
        "python_files_over_600": oversized,
    }
    primary = {
        key: {
            "raw": value["raw"],
            "refined": value["refined"],
            "refined_paired": value["refined_paired"],
        }
        for key, value in final["seeds"].items()
    }
    claim = {
        "supported": (
            "Full graph conditioning (rich features plus graph attention/messages) "
            "improves the parameter-matched global-slot control after a common refiner."
        ),
        "mechanism": (
            "The gain requires both the 19-column conditioning and graph topology; "
            "neither single-mechanism ablation is sufficient."
        ),
        "not_supported": [
            "FloorSet state of the art",
            "Superiority over the same-protocol Trinity d192 checkpoint",
            "Promotion of absolute or residual slot transport",
            "Promotion of x0, conditioning dropout, or raw-only range adaptation",
            "A strong causal contribution from the sampled global state beyond decoder shortcuts",
            "Raw placement feasibility without scale-pack legalization",
        ],
        "external_comparator": {
            "trinity_hard_cost": external["mean_cost"],
            "trinity_v_rel": external["mean_v_rel"],
            "final_graph_mean_refined_hard_cost": final[
                "seed_mean_absolute"
            ]["refined"]["graph"]["legalized_hard_cost"],
            "final_graph_mean_refined_v_rel": final["seed_mean_absolute"][
                "refined"
            ]["graph"]["legalized_v_rel"],
            "final_control_mean_refined_hard_cost": final[
                "seed_mean_absolute"
            ]["refined"]["control"]["legalized_hard_cost"],
            "final_control_mean_refined_v_rel": final["seed_mean_absolute"][
                "refined"
            ]["control"]["legalized_v_rel"],
            "graph_hard_cost_gap_to_trinity": (
                final["seed_mean_absolute"]["refined"]["graph"][
                    "legalized_hard_cost"
                ]
                / external["mean_cost"]
                - 1.0
            ),
        },
        "observed_bottleneck": (
            "The target global latent has std 3.28e-5 and effective rank 6.33/192; "
            "shuffling it changes dev geometry by only 0.85%."
        ),
    }
    checks = {
        "three_seed_gate_passed": final["status"] == "pass",
        "mechanism_gate_passed": mechanism["status"] == "pass",
        "architecture_gate_closed": architecture["status"]
        == "keep_velocity_full_graph",
        "range_negative_result_retained": range_result["status"]
        == "reject_range_adaptation",
        "runtime_gate_passed": runtime["status"] == "complete",
        "fixed_hierarchy_negative_retained": hierarchy["status"] == "complete",
        "external_comparison_reported": (
            external["mean_cost"]
            < final["seed_mean_absolute"]["refined"]["graph"][
                "legalized_hard_cost"
            ]
        ),
        "bottleneck_review_retained": all(
            (ROOT / relative).is_file()
            for relative in EVIDENCE[10:]
        ),
        "engineering_tests_pass": (
            pytest_summary["failures"] == 0
            and pytest_summary["errors"] == 0
            and chip_pytest_summary["failures"] == 0
            and chip_pytest_summary["errors"] == 0
        ),
        "no_over_600_line_python_file": not oversized,
    }
    checklist = {
        "status": "ready" if all(checks.values()) else "blocked",
        "scope": "CVPR experiment evidence package; not manuscript text",
        "checks": checks,
        "claim_boundary": claim,
        "manifest": "reproducibility_manifest.json",
        "primary_results": "tables/primary_results.json",
        "mechanism": "tables/mechanism_ablation.json",
        "negative_results": "tables/negative_results.json",
        "efficiency": "tables/efficiency.json",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write(args.out_dir / "reproducibility_manifest.json", manifest)
    write(args.out_dir / "tables/primary_results.json", primary)
    write(args.out_dir / "tables/mechanism_ablation.json", mechanism)
    write(
        args.out_dir / "tables/negative_results.json",
        {
            "range_adaptation": range_result,
            "slot_predictability": read(ROOT / EVIDENCE[5]),
            "fixed_hierarchy": hierarchy,
            "global_pool_stats": read(
                ROOT / "outputs/reports/chip_global_pool_stats_screen_v1.json"
            ),
            "global_film": read(
                ROOT / "outputs/reports/chip_global_film_screen_v1.json"
            ),
            "path_consistency": read(
                ROOT / "outputs/reports/chip_path_consistency_screen_v1.json"
            ),
            "refiner_target": read(
                ROOT / "outputs/reports/chip_refiner_target_probe_v1.json"
            ),
            "inference_ensemble": read(
                ROOT / "outputs/reports/chip_inference_ensemble_screen_v1.json"
            ),
            "global_causality": {
                "baseline": read(
                    ROOT / "outputs/reports/chip_global_causality_probe_v1.json"
                ),
                "film": read(
                    ROOT / "outputs/reports/chip_global_causality_film_v1.json"
                ),
            },
        },
    )
    write(
        args.out_dir / "tables/efficiency.json",
        {
            "runtime": runtime,
            "graph_parameters": screen["parameter_match"]["graph"],
            "control_parameters": screen["parameter_match"]["control"],
            "refined_graph_sample_seconds_100_cases": read(
                ROOT
                / "outputs/chip_graph_rescue/screen_20k/"
                "graph_slot_seed19_20k/refiner_nfe4_ref400.json"
            )["results"]["nfe4_refine400"]["stage_seconds"]["sample"],
            "refined_control_sample_seconds_100_cases": read(
                ROOT
                / "outputs/chip_graph_rescue/screen_20k/"
                "global_matched198_seed19_20k/refiner_nfe4_ref400.json"
            )["results"]["nfe4_refine400"]["stage_seconds"]["sample"],
        },
    )
    write(args.out_dir / "submission_checklist.json", checklist)
    print(json.dumps({"status": checklist["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
