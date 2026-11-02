#!/usr/bin/env python3
"""Integrate FEVP-2 vector evidence into CVPR ledger v13."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/reports/fp_cvpr_evidence_v13.json"


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    evidence = copy.deepcopy(_read("outputs/reports/fp_cvpr_evidence_v12.json"))
    claim = _read("outputs/reports/cvpr_vector_claim_v1.json")
    benchmark = _read("outputs/reports/vector_edit_benchmark_v1.json")
    attacks = _read("outputs/reports/fevp2_vector_attacks_v1.json")
    results = _read("outputs/reports/vector_edit_results_v1.json")
    significance = _read("outputs/reports/vector_edit_significance_v1.json")
    baseline = _read("outputs/reports/vector_edit_baselines_v1.json")
    failure = _read("outputs/reports/vector_edit_failure_tail_v1.json")
    regression = _read("outputs/reports/vector_edit_regression_gate_v1.json")
    smoke = _read("outputs/reports/vector_primitive_smoke_v1.json")
    topology = _read("outputs/reports/vector_topology_factorial_v1.json")
    inference = _read(
        "outputs/reports/vector_residual_inference_multiseed_v1.json"
    )
    gates = {
        "vector_benchmark_complete": all(benchmark["checks"].values()),
        "vector_attacks_sensitive": all(attacks["checks"].values()),
        "vector_results_complete": all(results["checks"].values()),
        "vector_significance_complete": all(significance["checks"].values()),
        "vector_baseline_complete": baseline["status"] == "complete",
        "vector_failure_tail_locality": all(failure["checks"].values()),
        "vector_raster_regression": all(regression["checks"].values()),
        "vector_architecture_smoke": all(smoke["checks"].values()),
        "vector_topology_factorial_audited": topology["status"] == "complete",
        "vector_inference_residual_complete": all(
            inference["checks"].values()
        ),
    }
    evidence.update(
        {
            "status": "complete" if all(gates.values()) else "failed",
            "submission_status": "cvpr_fevp2_vector_experiment_evidence_ready_v18",
            "primary_narrative": claim["claim"],
            "release_boundary": " ".join(claim["boundaries"]),
            "vector_route": {
                "claim": claim,
                "benchmark": {
                    "cases": benchmark["cases"],
                    "summary": benchmark["summary"],
                },
                "results": results,
                "significance": significance,
                "baseline": baseline,
                "failure_tail": failure,
                "architecture_smoke": smoke,
                "topology_factorial": topology,
                "inference_residual": inference,
                "regression_gate": regression,
            },
            "package_target": "outputs/cvpr_package_fp_v18",
        }
    )
    evidence["gates"].update(gates)
    evidence["new_gate_checks"] = gates
    evidence["protocol_sources"].update(
        {
            "vector_claim": "outputs/reports/cvpr_vector_claim_v1.json",
            "vector_benchmark": "outputs/reports/vector_edit_benchmark_v1.json",
            "vector_results": "outputs/reports/vector_edit_results_v1.json",
            "vector_significance": (
                "outputs/reports/vector_edit_significance_v1.json"
            ),
            "vector_baseline": "outputs/reports/vector_edit_baselines_v1.json",
        }
    )
    failed = [name for name, value in gates.items() if not value]
    if failed:
        raise ValueError(f"vector evidence gates failed: {failed}")
    OUTPUT.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "gates": gates}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
