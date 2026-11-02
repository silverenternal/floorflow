# Predicted-Control Reasonableness Dossier v1

Status date: 2026-11-02

This dossier separates target-free operation reliability from
target-comparative reconstruction fidelity. The predicted-control model is still
being improved, so these values are a submission candidate baseline, not a
final held-out result.

## Claim

> Under target-free predicted controls, FloorFlow executes every supported typed
> operation in the declared validation set, safely abstains from unsupported or
> ambiguous requests, preserves the outside region exactly, and produces
> structurally plausible plans. These properties hold despite limited
> reference-relative room and graph fidelity.

This claim does not assert high semantic reconstruction accuracy, free-text
editing, or a matched HouseMind ranking.

## Predicted-Control Plausibility

CubiCasa5K val491, five seeds:

| Metric | Five-seed mean | Interpretation |
|---|---:|---|
| Raster plausible | 0.98493 | High structural plausibility |
| Room coverage | 1.00000 | No missing interior partition |
| Room-face retention | 0.99565 | Explicit rooms survive CAD readout |
| Validity | 1.00000 | Mandatory legality passes |
| Functional watertightness | 0.99796 | Closed functional output |
| Wall continuity | 0.99543 | Connected structural graph |
| Opening-wall alignment | 0.99353 | Openings remain attached |
| Furniture-room containment | 1.00000 | Furniture remains room-local |
| Plan plausibility | 0.98656 | Frozen composite gate passes |
| Outside-footprint fraction | 0.00000 | No exterior leakage |
| Room overlap ratio | 0.00000 | No pairwise room overlap |

Raw fpeval watertightness is `0.81670` and must be reported separately from
functional watertightness. The difference is a footprint-component boundary:

- 401/491 single-component footprints have raw and componentwise watertightness
  `1.0`;
- 90 multi-component footprints fail the standard single-region rule;
- componentwise watertightness for those 90 inputs is `0.9889`.

This is not a redefinition of fpeval. It is a declared diagnostic boundary.

## Operation Reliability

The latest 738-prompt report was produced from a development exposure of the
test manifest. Its embedded prompt rows declare `split=test`; a legacy evaluator
bug wrote `split=val191` and `test_opened=false` into the summary. The metadata
writer is fixed, but this run cannot serve as the final held-out result.

| Evidence | Result |
|---|---:|
| Total typed prompts | 738 |
| Supported prompts | 438 |
| Executed supported prompts | 438 |
| Unsupported or ambiguous prompts | 300 |
| Safe abstentions | 300 |
| Outside pixel exact | 1.00000 |
| Outside room-label exact | 1.00000 |
| FP-Eval validity | 1.00000 |
| Supported-operation execution rate | 1.00000 |
| Unsupported/ambiguous abstention rate | 1.00000 |

The five supported categories are room-type change, room addition, room
deletion, opening pressure, and low-salience furniture. The three abstention
categories are unsupported wall move, ambiguous global scope, and conflicting
instruction. Every category preserves outside pixel and room-label content
exactly and reports validity `1.0`.

This evidence supports safe execution and abstention. It does not establish
high target-relative semantic fidelity, and it is not an unopened test result.

Object-delete execution checks also verify that:

- opening deletion removes the opening family and preserves room labels;
- furniture deletion removes the furniture family and preserves room labels;
- both operations leave the outside region exactly unchanged.

Inactive count-gated room queries are separately audited to have no ghost
tokens, classes, masks, instance IDs, or adjacency.

## Relative Refinement Evidence

Across the 438 executed prompts:

| Metric | Paired mean delta | 95% CI | Interpretation |
|---|---:|---:|---|
| Node F1 | +0.077893 | [0.065764, 0.090077] | Graph readout improves |
| Edge overlap | +0.001353 | [0.000269, 0.002892] | Small positive change |
| GED | +4.842466 | [3.616381, 6.107420] | Negative diagnostic |

The GED regression is retained. Relative graph-readout improvement does not
imply high absolute graph quality.

## Operation-Specific Semantic Success

The preliminary semantic audit is reported separately from safety. It does not
pass the proposed `0.90` gate for most operations:

| Operation | Current val191 success rate |
|---|---:|
| room add | 0.92 |
| room type | 0.55 |
| room remove | 0.20 |
| opening delete | 0.5085 |
| furniture delete | 0.00 |

These are preliminary stale-model results. They demonstrate why safe execution
and semantic success must not be merged into one claim.

## Fidelity Boundary

The predicted-control baseline remains far below oracle-control fidelity:

| Metric | Val491 predicted control |
|---|---:|
| Room macro IoU | 0.26972 |
| Room-support IoU | 0.77268 |
| Instance F1 @0.5 | 0.23065 |
| Instance type accuracy | 0.58825 |
| fpeval wall F1 | 0.44194 |
| fpeval opening F1 | 0.08295 |
| fpeval Room F1 | 0.36116 |
| Node F1 | 0.66020 |
| Edge overlap | 0.26462 |
| GED | 11.60081 |

The failure is dominated by room-instance identity, count, and type, not by
invalid output. Predicted and oracle rows are never pooled.

## Reviewer-Facing Interpretation

The defensible contribution is:

> FloorFlow decouples edit reliability from reference fidelity. Under imperfect
> predicted controls, the system knows which typed edits it can execute, applies
> them locally, preserves legality and outside content, and abstains when the
> request is unsupported.

The paper must not state:

- that predicted-control reference fidelity is high;
- that Node/Edge/GED SOTA is reached;
- that abstention is a successful semantic edit;
- that val191 is the held-out test;
- that oracle-control metrics measure predicted-control quality.

## Evidence Sources

- `docs/predicted_control_plausibility_v2.md`
- `docs/predicted_control_reasonableness_results_v1.md`
- `docs/predicted_control_fidelity_baseline_v3.md`
- `configs/predicted_control_reasonableness_contract_v1.json`
- `outputs/reports/predicted_control_plausibility_fiveseed_v12.json`
- `outputs/reports/predicted_control_fidelity_val491_v3.json`
- `outputs/reports/residential_predicted_edit_test503_final_v1.json`
- `outputs/reports/predicted_edit_object_delete_audit_v1.json`
- `outputs/reports/predicted_edit_ghost_state_audit_v1.json`
