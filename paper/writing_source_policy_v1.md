# CVPR Numerical Citation Policy v2

Status date: 2026-11-02

This policy governs the abstract, introduction, method, experiments, figure and
table captions, supplement, and reviewer replies. It replaces the v17 package
policy. A numerical statement enters the manuscript only when it resolves to the
selected scoped release below.

## Release Identity

- Evidence registry: `docs/floorplan_submission_source_registry_v2.json`.
- Provenance audit: `outputs/reports/floorplan_source_provenance_audit_v2.json`.
- Frozen release: `configs/floorplan_submission_evidence_freeze_v2.json`.
- Freeze audit: `outputs/reports/floorplan_evidence_freeze_audit_v2.json`.
- Figure cases: `outputs/reports/floorplan_figure_case_manifest_v1.json`.
- Artifact manifest: `outputs/floorplan_submission/artifact_manifest.json`.

The freeze and audit are regenerated after any registry, table, figure, or
manuscript-source change. A stale hash is a release failure, not a cosmetic
warning. Historical packages `outputs/cvpr_package_fp_v*` and historical evidence
reports are not manuscript sources.

## Canonical Table Bindings

| Table | Frozen source | Population |
|---|---|---|
| `paper/tables/main_oracle_anchor_v1` | `outputs/reports/oracle_submission_statistics_v2.json` and registry keys `oracle.*` | CubiCasa5K test503 declared oracle controls, 2,479 edits, five repeated execution seeds |
| `paper/tables/main_predicted_reliability_v1` | `outputs/reports/predicted_operation_evidence_v2.json` and registry keys `predicted.val491.*`, `operation.*` | val491 target-free five-seed plausibility; 738 development-exposed prompts; older val191 semantic diagnostic |
| `paper/tables/main_resource_and_ablation_v1` | registry keys `oracle.*matched*`, `oracle.*frozen_raster2seq*`, `mechanism.*`, `robustness.*` | Same-case oracle comparisons and 12,395 strict-locality composition-policy rows |
| `paper/tables/main_external_context_v1` | registry keys `external.*`, `predicted.fidelity.*` | Heterogeneous external context; no leaderboard ranking |

The Markdown tables are binding. LaTeX tables are generated from the same
script and must be regenerated rather than hand-edited.

## Numerical Rules

1. Report six decimals when the frozen source provides at least that precision.
2. Intervals resample floorplans, not repeated seed rows. Five seeds demonstrate
   reproducibility, not five independent plans.
3. Oracle, predicted, identity-fallback, generation, and free-text rows are never
   pooled. A result inherits the population and exposure of its source row.
4. Delta-IoU and Delta-MSE measure binary change support, not requested-class
   correctness or reference reconstruction fidelity.
5. Raw fpeval watertightness and functional/componentwise watertightness remain
   separate. A raw value of `0.816701` is never rounded or relabeled as `1.0`.
6. Supported execution, safe abstention, and deterministic object/ghost-state
   contracts are not semantic success.
7. The older val191 semantic diagnostic is labeled by model age and split. It is
   not attributed to the selected predicted-control checkpoint.
8. HouseMind-E is published cross-dataset context only. Raster2Seq is an
   outside-frozen redraw wrapper, not a native editing model.
9. Optional fresh generalization is `optional_generalization_not_evaluated`; no
   held-out number, execution receipt, or submission receipt is fabricated.

## Scope Lock

Allowed headline: a scoped oracle-control editing anchor plus a target-free
execution path with exact outside preservation, validity, structural
plausibility, and calibrated abstention. Predicted fidelity and semantic
operation realization are retained limitations.

Forbidden claims: complete HouseMind defeat, RPLAN or full-CAD leaderboard SOTA,
deployable free-text editing, high predicted room identity, clean held-out
generalization, or semantic success inferred from locality.

