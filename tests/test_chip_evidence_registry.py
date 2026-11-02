from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module(script: str):
    path = ROOT / "scripts" / script
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_salvage_matrix_closes_failed_output_contracts() -> None:
    matrix = _module("audit_chip_architecture_salvage.py").build_matrix(ROOT)
    entries = {entry["family"]: entry for entry in matrix["entries"]}

    assert matrix["status"] == "pass_cpu_audit"
    assert matrix["gpu_experiments_run"] is False
    assert entries["M45 projector-displacement output head"]["decision"] == "closed"
    assert entries["M46 all-layer state-conditioned edge bias"]["decision"] == "closed"
    assert all(entry["rerun_allowed"] is False for entry in entries.values())


def test_postfix_registry_keeps_budget_and_sota_boundaries() -> None:
    registry = _module("build_chip_postfix_claim_registry.py").build_registry(ROOT)

    assert registry["status"] == "pass_cpu_registry"
    assert registry["sota_ready"] is False
    assert registry["matched_budget_anchor_found"] is False
    assert registry["gpu_experiments_run"] is False
    assert registry["metrics"]["paired_v_rel_relative"] < 0
    assert registry["metrics"]["b2b_relative"] > 0
    statuses = {claim["status"] for claim in registry["claims"]}
    assert {"supported_contract", "descriptive_higher_budget", "unsupported"} <= statuses


def test_committed_registry_artifacts_match_generators() -> None:
    salvage = _module("audit_chip_architecture_salvage.py").build_matrix(ROOT)
    registry = _module("build_chip_postfix_claim_registry.py").build_registry(ROOT)

    committed_salvage = json.loads(
        (ROOT / "outputs/reports/chip_architecture_salvage_matrix_v1.json").read_text()
    )
    committed_registry = json.loads(
        (
            ROOT / "outputs/reports/chip_current_evidence_gate_postfix_v1.json"
        ).read_text()
    )
    assert committed_salvage == salvage
    assert committed_registry == registry


def test_v2_execution_plan_preserves_order_budgets_and_gpu_block() -> None:
    module = _module("audit_chip_v2_execution_plan.py")
    plan = json.loads(
        (ROOT / "configs/chip_v2_execution_plan_v1.json").read_text()
    )
    architecture = json.loads(
        (ROOT / "configs/chip_architecture_completion_v2.json").read_text()
    )
    report = module.audit_plan(ROOT, plan, architecture)

    assert report["status"] == "pass_cpu_plan_only"
    assert report["execution_order"] == [
        "M50-COMMITTED-HIERARCHY-FLOW",
        "M51-POSITION-SHAPE-FACTORIZATION",
        "M53-REFINER-COMPATIBLE-FLOW",
        "M52-RELATION-SPECIFIC-FUSION",
        "M55-ROUTEABILITY-SPATIAL-STREAM",
        "M57-CONFIDENCE-GATED-COMMIT",
        "M58-SHARED-TRUNK-FACTORIZATION",
        "M59-LEGALIZER-CONSTRAINED-ENDPOINT",
        "M60-INCIDENCE-NORMALIZED-GRAPH",
        "M61A-ROUTE-ONLY",
        "M61B-SOURCE-ONLY",
        "M62-CONSTRAINED-COMPLEMENTARY-FUSION",
        "M64-SOURCE-ONLY-CLEAN",
        "M65-GRAPH-SOURCE-FUSION",
    ]
    assert report["gpu_experiments_run"] is False
    assert all(report["checks"].values())
