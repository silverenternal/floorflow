# Floorplan Claim-Evidence Matrix v2

Status date: 2026-11-02. Status: `passed_scoped_release`.

| Claim | Authoritative evidence | Evaluation unit and population | Status | Required caveat |
|---|---|---|---|---|
| Strict outside preservation is auditable | Registry keys `oracle.*.outside_exact_ratio`, `operation.safety.*`; FEVP audit rows | Oracle test503 and development-exposed prompts | Supported | Contract metric, not semantic quality |
| Oracle-control edit support is strong | T1; `oracle_submission_statistics_v2.json` | 2,479 edits, five operations, five repeated seeds | Supported | Target-derived, non-deployable controls |
| Oracle outputs remain legal | Pooled FP-Eval validity and conditional watertightness | Five operations | Supported | Raw and functional watertightness are separate |
| Freezing alone does not explain the gap | T3; registry `frozen_raster2seq` and paired deltas | Same-case test503 controls | Supported | Raster2Seq is a redraw wrapper |
| Target-free outputs are structurally plausible | T2 Panel A | val491, five repeated seeds | Validation-supported | Raw watertightness is `0.816701` |
| Supported prompts execute and unsafe prompts abstain | T2 Panel B; `predicted_operation_evidence_v2.json` | 738 development-exposed test503 prompts | Development-supported, not held-out | Execution is not semantic success |
| Semantic operation realization is strong | T2 Panel B semantic diagnostic | 374 supported prompts from an older val191 model | Not established | Rates range from `0.00` to `0.92` |
| Predicted reference fidelity is high | T4; registry `predicted.fidelity.*` | val491 | Not established | Room macro IoU `0.269701`; graph fidelity weak |
| Typed composition policy improves matched controls | T3 ablation; registry `mechanism.budget.*` | 12,395 strict-locality rows | Supported diagnostic | Post hoc budget audit; no efficiency claim |
| Fresh held-out generalization is demonstrated | `outputs/reports/floorplan_heldout_availability_v1.json` | N/A | Optional, not evaluated | Does not block the scoped submission |
| Complete HouseMind/RPLAN/full-CAD SOTA | None | N/A | Forbidden | No matched executable evidence |

## Claim Language

Use:

> FloorFlow separates edit reliability from reference fidelity. Under declared
> oracle controls it provides a strong same-protocol editing anchor with exact
> outside preservation and legality. Under target-free predicted controls it
> preserves the safety contract and produces structurally plausible plans, while
> reference fidelity and semantic operation realization remain explicit limits.

Do not use:

- complete-method or uniform SOTA;
- deployable free-text editing;
- high predicted room identity or graph fidelity;
- semantic success inferred from exact locality, abstention, or no-op checks;
- development-exposed test prompts as held-out evidence;
- repeated seeds as independent plans.

## Source Rebuild Order

1. `uv run python scripts/build_floorplan_submission_evidence_v2.py`
2. `uv run python scripts/build_floorplan_submission_tables_v2.py`
3. `uv run python scripts/build_floorplan_submission_figures_v1.py`
4. `uv run python scripts/prepare_predicted_control_submission_freeze_v1.py --mode existing_evidence_scoped --output configs/floorplan_submission_evidence_freeze_v2.json`
5. `uv run python scripts/audit_floorplan_evidence_freeze_v2.py`
6. `uv run python scripts/build_floorplan_submission_artifact_v1.py`
7. `uv run python scripts/audit_floorplan_submission_integrity_v1.py`

