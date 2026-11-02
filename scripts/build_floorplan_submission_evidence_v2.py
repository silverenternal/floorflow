#!/usr/bin/env python3
"""Build the scoped CVPR evidence registry and derived statistics."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/floorplan_submission_source_registry_v2.json"
PROVENANCE_AUDIT = ROOT / "outputs/reports/floorplan_source_provenance_audit_v2.json"
ORACLE_STATS = ROOT / "outputs/reports/oracle_submission_statistics_v2.json"
OPERATION_EVIDENCE = ROOT / "outputs/reports/predicted_operation_evidence_v2.json"

SEEDS = (19, 20, 21, 22, 23)
OPERATIONS = (
    "room_type",
    "room_add",
    "room_remove",
    "opening_delete",
    "furniture_delete",
)
PRIMARY_METHOD = {
    "room_type": "typed_before_floorflow_after",
    "room_add": "typed_before_floorflow_after",
    "room_remove": "typed_before_floorflow_after",
    "opening_delete": "floorflow",
    "furniture_delete": "floorflow",
}
SOURCE_KEYS = {
    "edit": "outputs/reports/residential_edit_test503_v1.json",
    "edit_seed19": "outputs/reports/residential_edit_hybrid_typedall_seed19_v1.json",
    "trace": "outputs/reports/oracle_metric_trace_audit_20261102.json",
    "raster": "outputs/reports/fp_raster2seq_local_redraw_v1.json",
    "vector": "outputs/reports/residential_edit_vector_protocol_raw_v1.json",
    "hard_gate": "outputs/reports/fp_control_gated_hard_gate_v1.json",
    "fevp": "outputs/reports/fevp_fiveseed_stability_v2.json",
    "mechanism": "outputs/reports/fevp_mechanism_pareto_v3.json",
    "budget": "outputs/reports/fevp_budget_matched_control_v2.json",
    "attacks": "outputs/reports/fevp_attack_supplement_v1.json",
    "editing_aware": "outputs/reports/residential_editing_aware_baseline_v1.json",
    "matched": "outputs/reports/matched_local_editing_baselines_v2.json",
    "plausibility": "outputs/reports/predicted_control_plausibility_fiveseed_v12.json",
    "plausibility_seed19": "outputs/reports/predicted_control_plausibility_val491_final_v12.json",
    "fidelity": "outputs/reports/predicted_control_fidelity_val491_v3.json",
    "safety": "outputs/reports/residential_instruction_robustness_test503_final_v1.json",
    "semantic": "outputs/reports/predicted_operation_success_val191_v1.json",
    "object_delete": "outputs/reports/predicted_edit_object_delete_audit_v1.json",
    "ghost_state": "outputs/reports/predicted_edit_ghost_state_audit_v1.json",
    "identity": "outputs/reports/residential_identity_test509_v4_summary.json",
    "external": "outputs/reports/external_baseline_protocol_audit_v2.json",
    "ce2eplan": "outputs/reports/ce2eplan_tvcg_anchor_v2.json",
    "housemind_status": "outputs/reports/fp_housemind_reproduction_status_v1.json",
    "cvpr_evidence": "outputs/reports/fp_cvpr_evidence_v12.json",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pointer(document: Any, pointer: str) -> Any:
    if not pointer:
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer: {pointer}")
    value = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(token)]
        else:
            value = value[token]
    return value


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    value: str,
    *,
    iterations: int = 10000,
    seed: int = 20261102,
) -> dict[str, Any]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[str(row["name"])].append(float(row[value]))
    cluster_values = np.asarray(
        [np.mean(values) for values in clusters.values()],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = cluster_values[
        rng.integers(0, cluster_values.size, (iterations, cluster_values.size))
    ]
    means = sampled.mean(axis=1)
    return {
        "mean": float(cluster_values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "clusters": int(cluster_values.size),
        "rows": len(rows),
        "ci_method": "floorplan_cluster_percentile_bootstrap",
        "iterations": iterations,
        "rng_seed": seed,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (total - index) * float(p_value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def build_oracle_statistics(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    edit = sources["edit"]
    seed19 = sources["edit_seed19"]
    raster = sources["raster"]
    operations: dict[str, Any] = {}
    primary_p_values: dict[str, float] = {}
    for operation in OPERATIONS:
        method = PRIMARY_METHOD[operation]
        rows = seed19["operations"][operation]["rows"]
        if len(rows) != int(seed19["operations"][operation]["cases"]):
            raise ValueError(f"seed19 row count mismatch for {operation}")
        names = {str(row["name"]) for row in rows}
        frozen_names = {str(row["name"]) for row in raster["operations"][operation]["rows"]}
        if names != frozen_names:
            raise ValueError(f"frozen baseline name mismatch for {operation}")
        primary_rows = [
            {"name": row["name"], **row[method]}
            for row in rows
        ]
        seed_values = {
            str(seed): edit["operations"][operation]["per_seed"][str(seed)]["primary_edit"]
            for seed in SEEDS
        }
        identical = all(
            math.isclose(value["delta_iou"], seed_values["19"]["delta_iou"], abs_tol=0.0)
            and math.isclose(value["delta_mse"], seed_values["19"]["delta_mse"], abs_tol=0.0)
            for value in seed_values.values()
        )
        paired = edit["operations"][operation]["paired_vs_raster2seq_local"]
        primary_p_values[operation] = float(paired["p_value"])
        operations[operation] = {
            "cases": len(rows),
            "unique_floorplans": len(names),
            "primary_method": method,
            "delta_iou": cluster_bootstrap(primary_rows, "delta_iou"),
            "delta_mse": cluster_bootstrap(primary_rows, "delta_mse"),
            "outside_exact_ratio": cluster_bootstrap(primary_rows, "outside_exact_ratio"),
            "seed_variation": {
                "seeds": list(SEEDS),
                "delta_iou_values": [
                    float(seed_values[str(seed)]["delta_iou"]) for seed in SEEDS
                ],
                "delta_mse_values": [
                    float(seed_values[str(seed)]["delta_mse"]) for seed in SEEDS
                ],
                "primary_metric_identical_across_seeds": bool(identical),
                "unit": "repeated execution seed on the same selected rows",
            },
            "paired_vs_frozen_raster2seq": paired,
            "paired_wilcoxon_holm_p": None,
        }
    holm = holm_adjust(primary_p_values)
    for operation, p_value in holm.items():
        operations[operation]["paired_wilcoxon_holm_p"] = p_value
    return {
        "schema_version": "oracle-submission-statistics-v2",
        "status": "passed_scoped_release",
        "scope": (
            "CubiCasa5K test503 declared oracle controls; seed19 rows for "
            "cluster intervals; all five repeated-seed summaries retained."
        ),
        "population": {
            "edit_instances": sum(
                int(edit["operations"][operation]["per_seed"]["19"]["cases"])
                for operation in OPERATIONS
            ),
            "operations": len(OPERATIONS),
            "seeds": list(SEEDS),
            "cluster_key": "floorplan name including floor identifier",
        },
        "statistical_unit": {
            "intervals": (
                "Resample floorplans with replacement and average all edit "
                "instances inside a sampled floorplan."
            ),
            "seed_handling": (
                "Seed variation is reported separately; repeated seed rows are "
                "not treated as independent floorplans."
            ),
            "paired_baseline": (
                "One mean FloorFlow-minus-frozen-Raster2Seq difference per unique "
                "floorplan; Wilcoxon on unique floorplans with Holm adjustment "
                "across five operations."
            ),
        },
        "multiple_comparisons": {
            "primary_interval_adjustment": "none; 95% intervals are descriptive",
            "paired_baseline_adjustment": "Holm over five operations",
        },
        "trace_audit": {
            "source": SOURCE_KEYS["trace"],
            "scope": sources["trace"]["scope"],
            "seed19_rows_recomputed": 2479,
            "zero_iou_error_required": True,
        },
        "pooled_topology": {
            "source": SOURCE_KEYS["hard_gate"],
            "validity_min": sources["hard_gate"]["aggregate"]["validity_min"],
            "conditional_watertight": sources["hard_gate"]["aggregate"][
                "conditional_watertight"
            ],
            "boundary": (
                "Pooled across the fixed five-operation row set; not an "
                "operation-specific estimate."
            ),
        },
        "operations": operations,
    }


def build_operation_evidence(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    safety = sources["safety"]
    semantic = sources["semantic"]
    rows = [row for row in semantic["rows"] if row.get("executed")]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    semantic_operations = {}
    for category, category_rows in sorted(grouped.items()):
        failures: dict[str, int] = defaultdict(int)
        for row in category_rows:
            failures[str(row.get("failure_reason", "unknown"))] += 1
        semantic_operations[category] = {
            "prompts": len(category_rows),
            "successful_prompts": sum(bool(row.get("success")) for row in category_rows),
            "success_rate": (
                sum(bool(row.get("success")) for row in category_rows)
                / len(category_rows)
            ),
            "changed_prompts": sum(
                float(row.get("changed_pixels", 0)) > 0 for row in category_rows
            ),
            "source_present_prompts": sum(
                float(row.get("source_pixels_before", 0)) > 0
                for row in category_rows
            ),
            "source_drift_prompts": sum(
                float(row.get("source_removed_fraction", 0.0)) > 0
                for row in category_rows
            ),
            "failure_reasons": dict(sorted(failures.items())),
        }
    split_values = sorted({str(row["split"]) for row in safety["prompts"]})
    executed = sum(bool(row["executed"]) for row in safety["prompts"])
    abstained = sum(bool(row["abstained"]) for row in safety["prompts"])
    return {
        "schema_version": "predicted-operation-evidence-v2",
        "status": "passed_scoped_release_with_semantic_boundary",
        "safety": {
            "prompts": len(safety["prompts"]),
            "executed": executed,
            "abstained": abstained,
            "split_values_in_rows": split_values,
            "exposure": "development-exposed test prompts; not held-out",
            "outside_pixel_exact_min": min(
                float(row["outside_pixel_exact"]) for row in safety["prompts"]
            ),
            "outside_room_label_exact_min": min(
                float(row["outside_room_label_exact"]) for row in safety["prompts"]
            ),
            "validity_min": min(float(row["validity"]) for row in safety["prompts"]),
            "legacy_summary_fields": {
                "split": safety.get("split"),
                "test_opened": safety.get("test_opened"),
                "correction": "Prompt rows contain split=test; row-level values govern.",
            },
        },
        "semantic_diagnostics": {
            "source": SOURCE_KEYS["semantic"],
            "split": semantic.get("split"),
            "population": "supported val191 prompt rows from an older model run",
            "denominator": "executed prompts in each category",
            "operations": semantic_operations,
            "source_present_note": (
                "Counts are descriptive. The rows do not support a general "
                "source-present success-rate estimand for every operation."
            ),
        },
        "contract_checks": {
            "object_delete": {
                "source": SOURCE_KEYS["object_delete"],
                "passed": sources["object_delete"]["passed"],
                "checks": sources["object_delete"]["checks"],
                "interpretation": "Execution contract only; not semantic success.",
            },
            "ghost_state": {
                "source": SOURCE_KEYS["ghost_state"],
                "passed": sources["ghost_state"]["passed"],
                "checks": sources["ghost_state"]["checks"],
                "interpretation": "State contract only; not semantic success.",
            },
        },
        "boundary": (
            "Safety execution, contract checks, and semantic realization are "
            "reported separately. Abstentions and no-ops are never counted as "
            "semantic successes."
        ),
    }


def main() -> None:
    sources = {
        key: load(ROOT / path)
        for key, path in SOURCE_KEYS.items()
    }
    oracle = build_oracle_statistics(sources)
    operation = build_operation_evidence(sources)
    ORACLE_STATS.parent.mkdir(parents=True, exist_ok=True)
    ORACLE_STATS.write_text(json.dumps(oracle, indent=2) + "\n", encoding="utf-8")
    OPERATION_EVIDENCE.write_text(
        json.dumps(operation, indent=2) + "\n",
        encoding="utf-8",
    )

    source_files = {
        key: {
            "path": path,
            "sha256": sha256(ROOT / path),
            "bytes": (ROOT / path).stat().st_size,
        }
        for key, path in SOURCE_KEYS.items()
    }
    source_files["oracle_statistics"] = {
        "path": str(ORACLE_STATS.relative_to(ROOT)),
        "sha256": sha256(ORACLE_STATS),
        "bytes": ORACLE_STATS.stat().st_size,
    }
    source_files["operation_evidence"] = {
        "path": str(OPERATION_EVIDENCE.relative_to(ROOT)),
        "sha256": sha256(OPERATION_EVIDENCE),
        "bytes": OPERATION_EVIDENCE.stat().st_size,
    }

    entries: list[dict[str, Any]] = []

    def add(
        key: str,
        claim: str,
        source_key: str,
        pointer: str | None,
        *,
        population: str,
        unit: str,
        seeds: list[int] | str,
        evaluator: str,
        checkpoint_config: str,
        exposure: str,
        deployability: str,
        derivation: str | None = None,
        locator: str | None = None,
    ) -> None:
        expected = resolve_pointer(sources[source_key], pointer) if pointer else None
        entries.append(
            {
                "key": key,
                "claim": claim,
                "source_path": SOURCE_KEYS[source_key],
                "json_pointer": pointer,
                "locator": locator,
                "expected_value": expected,
                "derivation": derivation,
                "population": population,
                "unit": unit,
                "seeds": seeds,
                "evaluator": evaluator,
                "checkpoint_config": checkpoint_config,
                "exposure": exposure,
                "deployability": deployability,
                "provenance_gaps": [],
            }
        )

    oracle_population = "CubiCasa5K test503 eligible oracle-control edits"
    oracle_eval = "HouseMind-style Delta-IoU/Delta-MSE + strict outside comparison"
    oracle_ckpt = "selected existing completion/evaluator evidence; no new checkpoint"
    oracle_deploy = "not deployable; after-edit control is target-derived"
    for operation in OPERATIONS:
        op_population = f"{oracle_population}; operation={operation}"
        base = f"oracle.{operation}"
        add(
            f"{base}.cases",
            f"Number of eligible {operation} edits",
            "edit",
            f"/operations/{operation}/per_seed/19/cases",
            population=op_population,
            unit="edit instance",
            seeds=list(SEEDS),
            evaluator="selected-row count",
            checkpoint_config=oracle_ckpt,
            exposure="oracle target controls; development evaluation split",
            deployability=oracle_deploy,
        )
        for metric, label in (
            ("delta_iou", "Delta-IoU"),
            ("delta_mse", "Delta-MSE"),
            ("outside_exact_ratio", "outside pixel exactness"),
        ):
            add(
                f"{base}.{metric}",
                f"{label} for {operation}",
                "edit",
                f"/operations/{operation}/per_seed/19/primary_edit/{metric}",
                population=op_population,
                unit="mean over paired edits",
                seeds=list(SEEDS),
                evaluator=oracle_eval,
                checkpoint_config=oracle_ckpt,
                exposure="oracle target controls; development evaluation split",
                deployability=oracle_deploy,
            )
        for metric in (
            "outside_room_label_exact",
            "outside_room_polygon_exact",
            "outside_wall_exact",
            "outside_opening_exact",
            "outside_edge_exact",
            "validity",
        ):
            add(
                f"{base}.{metric}",
                f"{metric} for {operation}",
                "vector",
                f"/operations/{operation}/summary/{metric}/mean",
                population=op_population,
                unit="case mean",
                seeds=list(SEEDS),
                evaluator="deterministic strict-locality/vector audit",
                checkpoint_config="typed-splice deterministic consumer",
                exposure="oracle target controls; test503",
                deployability=oracle_deploy,
            )
        add(
            f"{base}.frozen_raster2seq_delta_iou",
            f"Frozen Raster2Seq Delta-IoU for {operation}",
            "raster",
            f"/operations/{operation}/rows/0/delta_iou",
            population=op_population,
            unit="first row pointer; operation mean stored in statistics v2",
            seeds="not seed-dependent",
            evaluator="frozen outside-redraw wrapper",
            checkpoint_config="Raster2Seq outside-frozen reconstruction wrapper",
            exposure="same rows and edit regions",
            deployability="non-native editing wrapper",
        )
        add(
            f"{base}.matched_editing_aware_iou",
            f"Editing-aware control Delta-IoU for {operation}",
            "editing_aware",
            f"/operations/{operation}/summary/control_delta_replace/delta_iou",
            population=op_population,
            unit="paired edit mean",
            seeds=[19],
            evaluator="local control-delta replacement",
            checkpoint_config="no learned checkpoint",
            exposure="same declared controls",
            deployability="weak deployable-style control consumer",
        )
        for metric, pointer, unit in (
            (
                "frozen_raster2seq_delta_iou_mean",
                "/raster2seq_delta_iou",
                "paired edit mean",
            ),
            (
                "frozen_raster2seq_outside_exact",
                "/raster2seq_outside_exact",
                "case mean",
            ),
            (
                "paired_vs_raster2seq_mean_delta",
                "/paired_vs_raster2seq/mean_delta",
                "paired floorplan mean difference",
            ),
            (
                "paired_vs_editing_aware_mean_delta",
                "/paired_vs_editing_aware/mean_delta",
                "paired floorplan mean difference",
            ),
        ):
            add(
                f"{base}.{metric}",
                f"Matched comparison {metric} for {operation}",
                "matched",
                f"/operations/{operation}{pointer}",
                population=op_population,
                unit=unit,
                seeds=[19],
                evaluator="same-case matched local comparison",
                checkpoint_config="frozen Raster2Seq outside wrapper; deterministic editing-aware control",
                exposure="oracle target controls; development evaluation split",
                deployability="baseline context only",
            )

    for operation in OPERATIONS:
        stats = oracle["operations"][operation]
        entries.append(
            {
                "key": f"oracle.{operation}.cluster_statistics",
                "claim": f"Floorplan-cluster interval and seed variation for {operation}",
                "source_path": str(ORACLE_STATS.relative_to(ROOT)),
                "json_pointer": f"/operations/{operation}",
                "locator": None,
                "expected_value": stats,
                "derivation": "floorplan-cluster bootstrap over seed19 rows",
                "population": f"{oracle_population}; operation={operation}",
                "unit": "floorplan cluster",
                "seeds": list(SEEDS),
                "evaluator": "scripts/build_floorplan_submission_evidence_v2.py",
                "checkpoint_config": oracle_ckpt,
                "exposure": "oracle target controls",
                "deployability": oracle_deploy,
                "provenance_gaps": [],
            }
        )

    predicted_metrics = {
        "raster_plausible": "raster plausibility",
        "plan_plausible": "plan plausibility",
        "room_coverage": "room coverage",
        "room_face_retention": "room-face retention",
        "validity": "validity",
        "watertight": "raw fpeval watertightness",
        "functional_watertight": "functional watertightness",
        "wall_continuity_rate": "wall continuity",
        "opening_wall_alignment": "opening-wall alignment",
        "furniture_room_containment": "furniture-room containment",
        "outside_footprint_fraction": "outside-footprint fraction",
        "room_overlap_ratio": "room overlap ratio",
        "room_macro_iou_diagnostic": "room macro IoU diagnostic",
    }
    for metric, label in predicted_metrics.items():
        add(
            f"predicted.val491.{metric}",
            label,
            "plausibility",
            f"/summary/{metric}/mean",
            population="CubiCasa val491, five repeated predicted-control seeds",
            unit="case mean; seed summary",
            seeds=list(SEEDS),
            evaluator="predicted-control plausibility evaluator",
            checkpoint_config="selected room/structure checkpoints listed in config v12",
            exposure="validation; target-free inference",
            deployability="deployable predicted-control path",
        )
    add(
        "predicted.val491.single_component_cases",
        "Single-component footprint cases passing raw watertightness",
        "plausibility_seed19",
        "/arms/union0_explicit/plan_failure_counts/multi_component_footprint_protocol_boundary",
        population="CubiCasa val491 seed19",
        unit="cases",
        seeds=[19],
        evaluator="component audit",
        checkpoint_config="selected room/structure checkpoints listed in config v12",
        exposure="validation; target-free inference",
        deployability="deployable predicted-control path",
        derivation="491 - multi_component_footprint_protocol_boundary",
    )
    add(
        "predicted.val491.multi_component_cases",
        "Multi-component footprint protocol-boundary cases",
        "plausibility_seed19",
        "/arms/union0_explicit/plan_failure_counts/multi_component_footprint_protocol_boundary",
        population="CubiCasa val491 seed19",
        unit="cases",
        seeds=[19],
        evaluator="component audit",
        checkpoint_config="selected room/structure checkpoints listed in config v12",
        exposure="validation; target-free inference",
        deployability="deployable predicted-control path",
    )
    add(
        "operation.safety.prompts",
        "Development operation prompts",
        "safety",
        "/prompt_count",
        population="CubiCasa test503 development-exposed operation prompts",
        unit="prompt",
        seeds=[19],
        evaluator="typed operation safety contract",
        checkpoint_config="selected predicted-control completion model",
        exposure="development-exposed test; not held-out",
        deployability="deployable prompt parser/abstention path",
    )
    for field in ("executed", "abstained"):
        add(
            f"operation.safety.{field}",
            f"Operation prompts {field}",
            "safety",
            f"/{field}",
            population="CubiCasa test503 development-exposed operation prompts",
            unit="prompt",
            seeds=[19],
            evaluator="typed operation safety contract",
            checkpoint_config="selected predicted-control completion model",
            exposure="development-exposed test; not held-out",
            deployability="deployable prompt parser/abstention path",
        )
    for category in (
        "room_addition",
        "room_type_change",
        "room_deletion",
        "opening_pressure",
        "furniture_low_salience",
    ):
        add(
            f"operation.semantic.{category}",
            f"Semantic success count and denominator for {category}",
            "semantic",
            f"/summary/{category}",
            population="supported val191 prompts from an older predicted model",
            unit="prompt",
            seeds=[19],
            evaluator="operation success predicates",
            checkpoint_config="older predicted model; not the selected release checkpoint",
            exposure="validation",
            deployability="diagnostic only",
        )
    add(
        "operation.object_delete.contract",
        "Object-delete execution contract",
        "object_delete",
        "/passed",
        population="contract smoke cases",
        unit="contract check",
        seeds=[19],
        evaluator="object-delete audit",
        checkpoint_config="selected predicted-control completion model",
        exposure="contract smoke",
        deployability="execution contract only",
    )
    add(
        "operation.ghost_state.contract",
        "Inactive-state ghost contract",
        "ghost_state",
        "/passed",
        population="contract smoke cases",
        unit="contract check",
        seeds=[19],
        evaluator="ghost-state audit",
        checkpoint_config="selected predicted-control completion model",
        exposure="contract smoke",
        deployability="state contract only",
    )

    for metric, pointer in (
        ("room_f1", "/test_metrics/room_f1"),
        ("room_iou", "/test_metrics/room_iou"),
        ("corner_f1", "/test_metrics/corner_f1"),
        ("opening_f1", "/test_metrics/window_door_f1"),
    ):
        add(
            f"identity.fallback.{metric}",
            f"Identity fallback {metric}",
            "identity",
            pointer,
            population="CubiCasa official test509 identity reconstruction",
            unit="case mean",
            seeds=[19],
            evaluator="official identity evaluator",
            checkpoint_config="validation-frozen eight-view consensus + polygon fusion",
            exposure="official test509",
            deployability="fallback reconstruction path",
        )

    add(
        "external.raster2seq.official_room_f1",
        "Raster2Seq public RoomF1 anchor",
        "external",
        "/raster2seq/official_room_f1",
        population="Raster2Seq published protocol",
        unit="RoomF1",
        seeds="published",
        evaluator="external paper",
        checkpoint_config="external model",
        exposure="public anchor",
        deployability="external context only",
    )
    add(
        "external.housemind.status",
        "HouseMind official inference code status",
        "housemind_status",
        "/official_code_status",
        population="HouseMind published assets",
        unit="status",
        seeds="published",
        evaluator="public artifact audit",
        checkpoint_config="public benchmark and weights; serialization unavailable",
        exposure="external context only",
        deployability="not reproduced",
    )
    add(
        "mechanism.typed_routing",
        "Typed routing mechanism evidence",
        "mechanism",
        "/audited_arms/typed_edit_graph",
        population="12,395 five-seed strict-locality rows",
        unit="row/seed aggregate",
        seeds=list(SEEDS),
        evaluator="FEVP mechanism audit",
        checkpoint_config="selected existing completion/evaluator evidence",
        exposure="oracle-control evaluation rows",
        deployability="oracle-control mechanism diagnostic",
    )
    for arm in (
        "dense_cross_attention",
        "no_edit_gating",
        "no_structural_graph",
    ):
        add(
            f"mechanism.{arm}",
            f"Composition-policy ablation arm {arm}",
            "mechanism",
            f"/audited_arms/{arm}",
            population="12,395 five-seed strict-locality rows",
            unit="row/seed aggregate",
            seeds=list(SEEDS),
            evaluator="FEVP mechanism audit",
            checkpoint_config="deterministic composition policy; not a trained network baseline",
            exposure="oracle-control evaluation rows",
            deployability="oracle-control mechanism diagnostic",
        )
    for key, pointer, unit in (
        (
            "mechanism.budget.typed_latency_ms",
            "/budget_matched_control/typed_latency_ms_mean",
            "milliseconds per row",
        ),
        (
            "mechanism.budget.no_graph_latency_ms",
            "/budget_matched_control/no_graph_latency_ms_mean",
            "milliseconds per row",
        ),
        (
            "mechanism.budget.typed_overhead_ms",
            "/budget_matched_control/typed_overhead_ms_mean",
            "milliseconds per row",
        ),
        (
            "mechanism.budget.delta_iou_gain",
            "/quality_deltas_typed_minus_budget_matched_no_graph/delta_iou/mean_delta",
            "paired row mean difference",
        ),
        (
            "mechanism.budget.ged_gain",
            "/quality_deltas_typed_minus_budget_matched_no_graph/ged/mean_delta",
            "paired row mean difference",
        ),
    ):
        add(
            key,
            "Post-hoc matched-compute routing audit",
            "budget",
            pointer,
            population="12,395 paired five-seed strict-locality rows",
            unit=unit,
            seeds=list(SEEDS),
            evaluator="FEVP matched-compute audit",
            checkpoint_config="same checkpoint/input/NFE; zero added learned parameters",
            exposure="oracle-control evaluation rows",
            deployability="post-hoc routing cost audit; not an efficiency claim",
        )
    add(
        "external.housemind_e.delta_iou",
        "Published HouseMind-E Delta-IoU context",
        "matched",
        "/external_context/HouseMind-E/delta_iou",
        population="HouseMind-E published protocol",
        unit="published metric",
        seeds="published",
        evaluator="external publication",
        checkpoint_config="external model",
        exposure="cross-dataset context only",
        deployability="not reproduced",
    )
    add(
        "external.housemind_e.delta_mse",
        "Published HouseMind-E Delta-MSE context",
        "matched",
        "/external_context/HouseMind-E/delta_mse",
        population="HouseMind-E published protocol",
        unit="published metric",
        seeds="published",
        evaluator="external publication",
        checkpoint_config="external model",
        exposure="cross-dataset context only",
        deployability="not reproduced",
    )
    for metric, pointer in (
        ("node_f1", "/house_mind_context/HouseMind-E/node_f1"),
        ("edge_overlap", "/house_mind_context/HouseMind-E/edge_overlap"),
        ("ged", "/house_mind_context/HouseMind-E/ged"),
    ):
        add(
            f"external.housemind_e.{metric}",
            f"Published HouseMind-E {metric} context",
            "mechanism",
            pointer,
            population="HouseMind-E published protocol",
            unit="published metric",
            seeds="published",
            evaluator="external publication",
            checkpoint_config="external model",
            exposure="cross-dataset context only",
            deployability="not reproduced",
        )
    for metric, pointer in (
        ("room_macro_iou", "/summary/room_macro_iou_mean"),
        ("node_f1", "/summary/node_f1_mean"),
        ("edge_overlap", "/summary/edge_overlap_mean"),
        ("ged", "/summary/ged_mean"),
    ):
        add(
            f"predicted.fidelity.{metric}",
            f"Predicted-control reference-fidelity diagnostic {metric}",
            "fidelity",
            pointer,
            population="CubiCasa val491 target-free predicted controls",
            unit="case mean",
            seeds=[19],
            evaluator="predicted-control fidelity evaluator",
            checkpoint_config="selected existing predicted-control completion pipeline",
            exposure="validation; target used for evaluation only",
            deployability="diagnostic; not part of the reasonableness gate",
        )
    add(
        "robustness.attack_suite",
        "Deterministic protocol attack suite",
        "attacks",
        "/checks",
        population="attack suite",
        unit="attack probe",
        seeds=[19],
        evaluator="FEVP attack supplement",
        checkpoint_config="frozen evaluator",
        exposure="evaluation-protocol audit",
        deployability="evaluator evidence",
    )

    registry = {
        "schema_version": "floorplan-submission-source-registry-v2",
        "status": "passed_scoped_release",
        "created_at": "2026-11-02",
        "scope": (
            "Selected existing oracle-control, predicted-control, operation-safety, "
            "identity-fallback, mechanism, and external-context evidence."
        ),
        "source_files": source_files,
        "entries": entries,
        "table_bindings": {
            "T1_oracle_anchor": [
                f"oracle.{operation}.{field}"
                for operation in OPERATIONS
                for field in ("cases", "delta_iou", "delta_mse", "outside_exact_ratio")
            ],
            "T2_predicted_reliability": [
                f"predicted.val491.{metric}" for metric in predicted_metrics
            ]
            + [
                "operation.safety.prompts",
                "operation.safety.executed",
                "operation.safety.abstained",
            ],
            "T3_matched_and_ablation": [
                f"oracle.{operation}.matched_editing_aware_iou" for operation in OPERATIONS
            ]
            + [
                f"oracle.{operation}.frozen_raster2seq_delta_iou_mean"
                for operation in OPERATIONS
            ]
            + [
                "mechanism.typed_routing",
                "mechanism.dense_cross_attention",
                "mechanism.no_edit_gating",
                "mechanism.no_structural_graph",
                "mechanism.budget.typed_latency_ms",
                "mechanism.budget.no_graph_latency_ms",
                "mechanism.budget.delta_iou_gain",
                "mechanism.budget.ged_gain",
                "robustness.attack_suite",
            ],
            "T4_external_context": [
                "external.raster2seq.official_room_f1",
                "external.housemind_e.delta_iou",
                "external.housemind_e.delta_mse",
                "external.housemind_e.node_f1",
                "external.housemind_e.edge_overlap",
                "external.housemind_e.ged",
                "predicted.fidelity.room_macro_iou",
                "predicted.fidelity.node_f1",
                "predicted.fidelity.edge_overlap",
                "predicted.fidelity.ged",
                "external.housemind.status",
            ],
        },
        "provenance_gaps": [
            (
                "The selected predicted-control report does not embed a single "
                "execution hash for every checkpoint; file hashes and checkpoint "
                "paths are frozen in the release configuration instead."
            ),
            (
                "The operation semantic audit uses an older predicted model and "
                "is retained only as a diagnostic boundary."
            ),
            (
                "Fresh held-out generalization is not evaluated and is excluded "
                "from the scoped claims."
            ),
            (
                "HouseMind-E anchors are published external context; the official "
                "inference serialization is unavailable and no matched ranking is claimed."
            ),
        ],
    }
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    errors: list[str] = []
    warnings: list[str] = []
    for entry in entries:
        path = ROOT / entry["source_path"]
        if not path.is_file():
            errors.append(f"missing source: {entry['source_path']}")
            continue
        if entry["json_pointer"] is not None:
            try:
                resolve_pointer(load(path), entry["json_pointer"])
            except (KeyError, IndexError, TypeError, ValueError) as error:
                errors.append(
                    f"{entry['key']}: pointer {entry['json_pointer']} failed: {error}"
                )
        if not entry["exposure"]:
            warnings.append(f"{entry['key']}: exposure not recorded")
        if not entry["deployability"]:
            warnings.append(f"{entry['key']}: deployability not recorded")
    audit = {
        "schema_version": "floorplan-source-provenance-audit-v2",
        "status": "passed" if not errors else "failed",
        "scope": "selected scoped-release evidence only",
        "checks": {
            "all_source_files_exist": not errors,
            "all_json_pointers_resolve": not errors,
            "all_entries_record_population": all(
                bool(entry["population"]) for entry in entries
            ),
            "all_entries_record_unit": all(bool(entry["unit"]) for entry in entries),
            "all_entries_record_seeds": all(bool(entry["seeds"]) for entry in entries),
            "all_entries_record_evaluator": all(
                bool(entry["evaluator"]) for entry in entries
            ),
            "all_entries_record_checkpoint_config": all(
                bool(entry["checkpoint_config"]) for entry in entries
            ),
            "all_entries_record_exposure": all(
                bool(entry["exposure"]) for entry in entries
            ),
            "all_entries_record_deployability": all(
                bool(entry["deployability"]) for entry in entries
            ),
            "provenance_gaps_are_explicit": bool(registry["provenance_gaps"]),
            "original_evidence_retained": all(
                (ROOT / path).is_file() for path in SOURCE_KEYS.values()
            ),
        },
        "entry_count": len(entries),
        "errors": errors,
        "warnings": warnings,
        "provenance_gaps": registry["provenance_gaps"],
    }
    PROVENANCE_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "registry": str(REGISTRY.relative_to(ROOT)),
                "entries": len(entries),
                "provenance_status": audit["status"],
                "oracle_statistics": str(ORACLE_STATS.relative_to(ROOT)),
                "operation_evidence": str(OPERATION_EVIDENCE.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
