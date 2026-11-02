# Predicted-Control Operation Success Audit v1

Status date: 2026-11-02

## Purpose

This audit separates three properties that were previously conflated:

1. the edit was supported by the typed operation contract;
2. the edit executed without violating locality or validity;
3. the requested semantic change is visible in the resulting plan.

Only the third property is operation-specific semantic success.

## Evaluation Status

The predicted-control model is still under development. The current audit uses
the older val191 output rows and is therefore preliminary:

- model frozen: false;
- test used: false;
- semantic thresholds: proposed, not submission-frozen;
- result status: below the proposed `0.90` success gate for several operations.

## Preliminary Results

| Operation | Prompts | Semantic successes | Success rate | Inside change rate |
|---|---:|---:|---:|---:|
| room add | 100 | 92 | 0.92 | 0.96 |
| room type | 100 | 55 | 0.55 | 1.00 |
| room remove | 100 | 20 | 0.20 | 0.85 |
| opening delete | 59 | 30 | 0.5085 | 0.5085 |
| furniture delete | 15 | 0 | 0.00 | 0.00 |

Unsupported and ambiguous categories retain 100% abstention, exact outside
pixel/room-label preservation, and validity `1.0`.

The latest five-prompt smoke rows show the same pattern:

| Operation | Success rate |
|---|---:|
| room add | 0.80 |
| room type | 0.80 |
| room remove | 0.20 |
| opening delete | 0.40 |
| furniture delete | 0.00 |

These smoke rates are directional only because each category has five prompts.

## Interpretation

The current evidence supports:

- safe execution;
- safe abstention;
- exact locality;
- structural validity;
- structural plausibility.

It does not yet support a claim that all supported operations realize their
requested semantic change at a high rate.

The main failure families are:

- room removal retains the source class or merges into multiple classes;
- room type realizes only part of the requested conversion;
- opening deletion is often a no-op because the source family is absent from
  the current plan;
- furniture deletion has no visible target-family change in the current val491
  completion raster.

## Old Control-Source Policy

Using the pre-edit control source is a valid editing input. CAD editors also
operate from the old drawing state. This audit therefore does not penalize
source dependence itself.

It separately measures:

- whether the source exists in the current plan;
- whether the requested target is realized;
- whether the source family/class is removed when required;
- whether the edit changes the current plan at all.

This distinguishes a valid source-conditioned request from a request that
cannot be synchronized with the current output state.

## Metadata Correction

The latest 738-prompt report contains prompt rows with `split=test`. A legacy
evaluator bug wrote `split=val191` and `test_opened=false` into its summary.
`scripts/evaluate_residential_predicted_edit_v1.py` now writes those fields from
the actual run arguments.

The affected run remains development exposure and cannot be used as the final
held-out claim.

## Required Follow-Up

1. Re-run this audit after the predicted model and control contract are frozen.
2. Apply the `0.90` semantic-success threshold to every supported operation.
3. Report source-present and source-drifting subsets separately.
4. Keep semantic success, safety, and reference fidelity in separate tables.
5. Use a fresh unopened held-out split for final submission.

## Sources

- `configs/predicted_operation_success_contract_v1.json`
- `src/floorflow/evaluation/predicted_operation_success.py`
- `scripts/evaluate_predicted_operation_success_v1.py`
- `outputs/reports/predicted_operation_success_val191_v1.json`
- `outputs/reports/predicted_operation_success_smoke5_v12.json`
- `outputs/reports/residential_instruction_robustness_test503_final_v1.json`
