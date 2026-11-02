# Oracle-Control CVPR Metric Dossier v2

Status date: 2026-11-02. Status: `frozen_scoped_release`.

This is the canonical claim-evidence map for the oracle-control editing track.
It is an internal submission dossier, not manuscript prose.

## Scoped Claim

> Under one frozen CubiCasa5K test503 oracle-control protocol, FloorFlow reaches
> Delta-IoU 0.956703-0.999801 and Delta-MSE 0.000019-0.000096 across five typed
> edits. Outside content and FP-Eval validity remain exact under the declared
> contract. Same-case wrapper and editing-aware comparisons, a deterministic
> composition-policy ablation, and protocol attacks separate edit support from
> locality and freezing alone.

Oracle controls are target-derived and non-deployable. The result is not a
HouseMind RPLAN leaderboard rank, a predicted-control result, or complete-method
SOTA.

## Main Result

CubiCasa5K test503, 2,479 paired edits, five operations, five repeated execution
seeds. Delta-IoU intervals resample floorplans, not duplicated seed rows.

| Operation | N | Delta-IoU [95% CI] | Delta-MSE | Outside exact | FP-Eval | FEVP 5-check |
|---|---:|---:|---:|---:|---:|---:|
| Room type | 487 | 0.999199 [0.998420, 0.999747] | 0.000069 | 1.000000 | 1.000000 | 0.9569 |
| Room add | 495 | 0.999801 [0.999545, 0.999990] | 0.000030 | 1.000000 | 1.000000 | 0.9556 |
| Room remove | 500 | 0.999756 [0.999479, 0.999966] | 0.000019 | 1.000000 | 1.000000 | 0.9540 |
| Opening delete | 502 | 0.956703 [0.941316, 0.970424] | 0.000096 | 1.000000 | 1.000000 | 0.9124 |
| Furniture delete | 495 | 0.987812 [0.982565, 0.992375] | 0.000083 | 1.000000 | 1.000000 | 0.9515 |

The seed19 trace audit recomputes all 2,479 rows with zero IoU error. Other seeds
are retained as reported reproducibility checks rather than independent plans.
Pooled conditional FP-Eval watertightness is `0.957875-0.967491`; it is not an
operation-specific estimate.

## Metric Boundary

Delta-IoU and Delta-MSE measure binary change support. They do not prove that the
requested room class is correct or that the reference plan is reconstructed.
Those questions are isolated in the predicted-control semantic diagnostic and
fidelity table.

Outside exact covers the declared pixel, room-label, mask, polygon, wall,
opening, and unaffected-edge contract. A change outside the valid domain is a
contract failure, not a small quality loss.

## Matched Comparisons

| Operation | N | FloorFlow Delta-IoU | Raster2Seq wrapper | Editing-aware | Paired gain vs wrapper | Paired gain vs editing-aware |
|---|---:|---:|---:|---:|---:|---:|
| Room type | 487 | 0.999199 | 0.137160 | 0.909800 | 0.862039 | 0.089486 |
| Room add | 495 | 0.999801 | 0.166756 | 0.995985 | 0.833044 | 0.004015 |
| Room remove | 500 | 0.999756 | 0.177266 | 0.998508 | 0.822490 | 0.001340 |
| Opening delete | 502 | 0.956703 | 0.189491 | 0.731151 | 0.767212 | 0.225552 |
| Furniture delete | 495 | 0.987812 | 0.145367 | 0.742469 | 0.842445 | 0.245343 |

The Raster2Seq row is an outside-frozen redraw wrapper, not a native editing
model. Its outside exactness is a wrapper property. The editing-aware runner
recomputes its own paired FloorFlow row, so its absolute value can differ
slightly from the main table; paired differences remain within that run.

## Mechanism And Robustness

The 12,395-row composition-policy audit compares deterministic policies, not
separately trained networks:

| Arm | Outside pixel | Outside room | Violation | Delta-IoU | Node F1 | Edge F1 | GED |
|---|---:|---:|---:|---:|---:|---:|---:|
| Typed routing | 1.000000 | 1.000000 | 0.000000 | 0.937473 | 0.955068 | 0.890007 | 2.120855 |
| Dense routing | 0.999545 | 0.999470 | 0.000455 | 0.887741 | 0.953797 | 0.887080 | 2.185478 |
| No edit gating | 0.986115 | 0.983599 | 0.013885 | 0.853715 | 0.949057 | 0.874121 | 2.448649 |
| No structural graph | 1.000000 | 1.000000 | 0.000000 | 0.888817 | 0.953819 | 0.887200 | 2.182251 |

At matched observable routing budget, typed routing adds Delta-IoU
`+0.048656` and changes GED by `-0.061396` with zero added learned parameters.
It consumes more routing time (`3.1925` vs `0.2666` ms mean); efficiency is not
claimed. All 26 deterministic protocol attacks are sensitive, with observed
false-pass and false-fail rates `0.0`.

## Negative Controls

- Removing edit gating lowers room-level exactness to `0.983599` and raises the
  violation rate to `0.013885`.
- Wall-adjacent, opening-covering, and exterior-boundary stress suites retain
  gated outside exactness `1.0`.
- Single-room negative controls retain the same declared outside contract.
- Graph quality is reported as a diagnostic; it is not pooled with the typed
  operation-delta metric.

## Source Bindings

- `docs/floorplan_submission_source_registry_v2.json`
- `outputs/reports/floorplan_source_provenance_audit_v2.json`
- `outputs/reports/oracle_submission_statistics_v2.json`
- `outputs/reports/oracle_metric_trace_audit_20261102.json`
- `outputs/reports/matched_local_editing_baselines_v2.json`
- `outputs/reports/fevp_mechanism_pareto_v2.json`
- `outputs/reports/fevp_protocol_attacks_v2.json`
- `configs/floorplan_submission_evidence_freeze_v2.json`
- `outputs/reports/floorplan_evidence_freeze_audit_v2.json`

## Reviewer Defense

The empirical claim is deliberately narrower than complete-method SOTA:

> Under one frozen oracle-control protocol, FloorFlow provides a strong
> same-protocol editing anchor with exact outside preservation, legality, matched
> controls, and attack-tested auditing, while its graph and predicted-control
> fidelity remain limited.

This wording names the evaluated unit, split, metric definitions, matched
comparisons, and limitation. It supports a CVPR paper about verifiable local
editing without overstating dataset or benchmark equivalence.

