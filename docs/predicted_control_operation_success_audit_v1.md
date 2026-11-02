# Predicted-Control Operation Success Audit v1

Status date: 2026-11-02

## Purpose

This audit separates three properties:

1. the request is supported by the typed operation contract;
2. the edit executes with exact locality and structural validity;
3. the requested semantic change is visible in the resulting plan.

Only the third property is operation-specific semantic success. Safety,
abstention, object-delete contracts, ghost-state contracts, and semantic
realization are reported as separate evidence axes.

## Evidence Populations

- Safety: 738 development-exposed test503 prompts; 438 supported prompts
  executed and 300 unsupported or ambiguous prompts abstained.
- Semantic diagnostic: older val191 rows from a predicted model that is not the
  selected release checkpoint. This is a diagnostic boundary, not a held-out
  result and not a release gate.
- Contract checks: deterministic object-delete and inactive-ghost probes.

Every safety prompt row reports outside pixel exactness `1.0`, outside room-label
exactness `1.0`, and validity `1.0`. These are scoped observed values, not a
general guarantee for unseen inputs.

## Semantic Diagnostic

| Operation | Prompts | Semantic successes | Success rate | Changed prompts | Source present | Source drift |
|---|---:|---:|---:|---:|---:|---:|
| room add | 100 | 92 | 0.9200 | 96 | 68 | 63 |
| room type | 100 | 55 | 0.5500 | 100 | 47 | 47 |
| room remove | 100 | 20 | 0.2000 | 85 | 96 | 29 |
| opening delete | 59 | 30 | 0.5085 | 30 | 30 | 30 |
| furniture delete | 15 | 0 | 0.0000 | 0 | 0 | 0 |

The `0.90` semantic target is retained only as a historical exploratory target.
It is not a prerequisite for the scoped release. Low semantic success is visible
in the main paper rather than reclassified as safety failure.

## Contract Evidence

The object-delete audit checks family deletion inside the edit region, room-pixel
preservation, and exactness outside the region. The ghost-state audit checks that
inactive instance ids, tokens, classes, adjacency, masks, and assignments are
masked. Both pass their listed deterministic checks. Neither is semantic
success evidence.

## Exposure Correction

The 738 prompt rows contain `split=test`. A legacy evaluator summary wrote
`split=val191` and `test_opened=false`. The row-level values govern the report.
The run is development-exposed and must not be described as held-out.

## Interpretation

Supported claims:

- every supported prompt in the declared development set executed through the
  typed path;
- every unsupported or ambiguous prompt abstained;
- observed outside exactness and validity are `1.0` on those prompts;
- structural reasonableness is supported separately on val491.

Not supported:

- uniformly high semantic realization;
- a fresh held-out generalization claim;
- a claim that object-delete or ghost-state contracts establish semantic
  correctness.

## Sources

- `configs/predicted_operation_success_contract_v1.json`
- `src/floorflow/evaluation/predicted_operation_success.py`
- `scripts/evaluate_predicted_operation_success_v1.py`
- `outputs/reports/predicted_operation_evidence_v2.json`
- `outputs/reports/predicted_operation_success_val191_v1.json`
- `outputs/reports/residential_instruction_robustness_test503_final_v1.json`
- `outputs/reports/predicted_edit_object_delete_audit_v1.json`
- `outputs/reports/predicted_edit_ghost_state_audit_v1.json`
