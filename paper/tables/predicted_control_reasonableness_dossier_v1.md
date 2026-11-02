# Predicted-Control Reasonableness Dossier v2

Status date: 2026-11-02. Status: `frozen_scoped_release_with_semantic_boundary`.

This dossier separates target-free structural reasonableness and execution
safety from reference-relative fidelity and semantic operation realization.

## Scoped Claim

> With target-free predicted controls, FloorFlow preserves exact outside content,
> produces valid and structurally plausible plans across five repeated val491
> seeds, executes every supported typed prompt in the development-exposed audit,
> and abstains safely on unsupported or ambiguous requests. Raw topology and
> reference fidelity remain explicit limitations.

This is not a fresh held-out result, a free-text editing claim, a matched
HouseMind ranking, or evidence of high room-instance fidelity.

## Structural Reasonableness

CubiCasa5K val491, 491 target-free cases per seed, five repeated seeds:

| Metric | Five-seed mean | Interpretation |
|---|---:|---|
| Raster plausible | 0.984929 | Raster agreement with the frozen plausibility contract |
| Plan plausible | 0.986558 | Composite structure gate |
| Room coverage | 1.000000 | Interior partition coverage |
| Room-face retention | 0.995655 | Explicit room survives CAD readout |
| FP-Eval validity | 1.000000 | Mandatory legality |
| Raw fpeval watertightness | 0.816701 | Raw single-region protocol boundary |
| Functional watertightness | 0.997963 | Functional component-aware topology |
| Wall continuity | 0.995427 | Connected wall structure |
| Opening-wall alignment | 0.993530 | Openings remain attached to walls |
| Furniture-room containment | 1.000000 | Furniture remains room-local |
| Outside-footprint fraction | 0.000000 | No exterior leakage |
| Room overlap ratio | 0.000000 | No pairwise room overlap |
| Room macro IoU diagnostic | 0.269701 | Weak reference fidelity; not a gate |

All five seeds pass the frozen reasonableness contract. Repeated seeds demonstrate
reproducibility, not independent floorplans.

## Watertightness Boundary

The raw value `0.816701` is not rounded or relabeled as `1.0`. Of 491 footprints,
401 are single-component and pass both raw and componentwise checks. The remaining
90 are multi-component footprints that exceed the standard single-region FP-Eval
boundary. Their componentwise watertightness is a declared diagnostic, not a
redefinition of the raw metric.

## Execution Safety

The operation audit contains 738 development-exposed test503 prompts. Row-level
metadata declares `split=test`; legacy summary fields claiming val191 are ignored.

| Evidence | Result |
|---|---:|
| Total typed prompts | 738 |
| Supported prompts | 438 |
| Executed supported prompts | 438 |
| Unsupported or ambiguous prompts | 300 |
| Safe abstentions | 300 |
| Outside pixel exact minimum | 1.000000 |
| Outside room-label exact minimum | 1.000000 |
| FP-Eval validity minimum | 1.000000 |
| Supported execution rate | 1.000000 |
| Unsupported/ambiguous abstention rate | 1.000000 |

The supported categories are room-type change, room addition, room deletion,
opening pressure, and low-salience furniture. The abstention categories are
unsupported wall move, ambiguous global scope, and conflicting instruction.
Execution and abstention are safety/contract outcomes, not semantic success.

Object-delete and ghost-state audits pass deterministic contracts. They verify
family removal or inactive-state cleanliness and do not count as successful
semantic edits.

## Semantic Boundary

An older val191 model provides the only supported-operation semantic diagnostic.
It is not the selected predicted-control checkpoint and is reported only as a
boundary:

| Operation | Successes / prompts | Rate | Failure mode |
|---|---:|---:|---|
| Room add | 92 / 100 | 0.9200 | Missing or insufficient requested class |
| Room type | 55 / 100 | 0.5500 | Requested class does not dominate change |
| Room remove | 20 / 100 | 0.2000 | Source class retained |
| Opening delete | 30 / 59 | 0.5085 | Required family deletion or source drift |
| Furniture delete | 0 / 15 | 0.0000 | Family retained or absent |

The diagnostic does not satisfy a uniform `0.90` semantic gate. That gate is not
a submission prerequisite for the scoped claim.

## Reference Fidelity

Predicted-control fidelity remains substantially below oracle-control fidelity.

| Metric | val491 value | Interpretation |
|---|---:|---|
| Room macro IoU | 0.269701 | Weak room-instance identity |
| Node F1 | 0.660202 | Partial graph topology |
| Edge overlap | 0.264622 | Weak canonical edges |
| GED | 11.600815 | High graph distance; lower is better |

The failure is dominated by room-instance identity, count, type, and graph
readout rather than invalid output. Predicted and oracle rows are never pooled.

## Reviewer-Facing Interpretation

The defensible contribution is:

> FloorFlow decouples edit reliability from reference fidelity. Under imperfect
> predicted controls, the system knows which typed requests it can execute,
> applies them without outside leakage, preserves legality, and abstains on
> unsupported requests, while target-relative fidelity and semantics remain
> measurable limitations.

Do not state that predicted reference fidelity is high, that Node/Edge/GED SOTA
is reached, that abstention is a successful edit, that val191 is held out, or
that oracle-control metrics measure predicted-control quality.

## Source Bindings

- `docs/floorplan_submission_source_registry_v2.json`
- `outputs/reports/predicted_operation_evidence_v2.json`
- `outputs/reports/predicted_control_plausibility_fiveseed_v12.json`
- `outputs/reports/predicted_control_fidelity_val491_v3.json`
- `outputs/reports/predicted_operation_success_val191_v1.json`
- `outputs/reports/predicted_edit_object_delete_audit_v1.json`
- `outputs/reports/predicted_edit_ghost_state_audit_v1.json`
- `configs/floorplan_submission_evidence_freeze_v2.json`
- `outputs/reports/floorplan_evidence_freeze_audit_v2.json`

