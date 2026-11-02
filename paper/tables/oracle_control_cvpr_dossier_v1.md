# Oracle-Control CVPR Metric Dossier v1

Status date: 2026-11-02

This is the canonical one-page claim-evidence map for the residential
oracle-control editing track. It is an internal submission dossier, not
manuscript prose.

## Claim

> Under one frozen CubiCasa test503 oracle-control protocol, FloorFlow reaches
> Delta-IoU 0.956703-0.999801 and Delta-MSE 0.000019-0.000096 across five typed
> edits. These values exceed the published HouseMind-E metric anchors 0.608 and
> 0.0019 while exact outside preservation, FP-Eval validity, and controlled
> matched comparisons separate edit fidelity from locality.

The claim is protocol-scoped. Oracle controls are target-derived and
non-deployable; the result is not a HouseMind RPLAN leaderboard rank or
complete-method SOTA.

## Main Result

CubiCasa5K test503, 2,479 paired edits, five operations, five repeated seeds:

| Operation | N | Delta-IoU | Delta-MSE | Outside exact | FP-Eval | FEVP 5-check | Watertight |
|---|---:|---:|---:|---:|---:|---:|---:|
| room type | 487 | 0.999199 | 0.000069 | 1.000000 | 1.000000 | 0.9569 | 0.9579-0.9675 |
| room add | 495 | 0.999801 | 0.000030 | 1.000000 | 1.000000 | 0.9556 | 0.9579-0.9675 |
| room remove | 500 | 0.999756 | 0.000019 | 1.000000 | 1.000000 | 0.9540 | 0.9579-0.9675 |
| opening delete | 502 | 0.956703 | 0.000096 | 1.000000 | 1.000000 | 0.9124 | 0.9579-0.9675 |
| furniture delete | 495 | 0.987812 | 0.000083 | 1.000000 | 1.000000 | 0.9515 | 0.9579-0.9675 |

Outside exact covers pixel, room label, room mask, polygon, wall, opening, and
unaffected edge content. The primary local-delta path is deterministic on fixed
rows; repeated seeds demonstrate reproducibility rather than population-level
seed variance.

## Context And Matched Controls

HouseMind-E is cross-dataset metric context. Raster2Seq is a matched
reconstruction wrapper; the editing-aware baseline consumes declared after-edit
controls.

| Evidence | Delta-IoU | Delta-MSE | Comparison |
|---|---:|---:|---|
| FloorFlow current range | 0.956703-0.999801 | 0.000019-0.000096 | Positive headline |
| HouseMind-E published anchor | 0.608 | 0.0019 | Different data and inputs |
| Raster2Seq local-redraw | 0.035690-0.164930 | 0.088200-0.173900 | Reconstruction wrapper |
| Editing-aware control | 0.731151-0.998508 | 0.000086-0.011078 | Same declared controls |

FloorFlow exceeds the Raster2Seq wrapper for every operation; paired Wilcoxon
`p <= 1.60e-74`, with rank-biserial effect `0.99987-1.0`. Raster2Seq outside
exactness ranges from `0.8773` to `0.9573`, versus `1.0` for FloorFlow.

FloorFlow minus the editing-aware control:

| Operation | Delta-IoU gain | Paired 95% CI |
|---|---:|---:|
| room type | +0.089486 | [0.065883, 0.115083] |
| room add | +0.004015 | [0.000000, 0.010048] |
| room remove | +0.001340 | [0.000000, 0.004021] |
| opening delete | +0.225552 | [0.204670, 0.245953] |
| furniture delete | +0.245343 | [0.223465, 0.268625] |

## Mechanism And Robustness

FEVP aggregate mechanism audit, 12,395 five-seed rows:

| Arm | Outside pixel | Outside room | Violation | Delta-IoU | Node F1 | Edge F1 | GED |
|---|---:|---:|---:|---:|---:|---:|---:|
| Typed routing | 1.000000 | 1.000000 | 0.000000 | 0.937473 | 0.955068 | 0.890007 | 2.120855 |
| Dense cross-attention | 0.999545 | 0.999470 | 0.000455 | 0.887741 | 0.953797 | 0.887080 | 2.185478 |
| No edit gating | 0.986115 | 0.486325 | 0.013885 | 0.853715 | 0.949057 | 0.874121 | 2.448649 |
| No structural graph | 1.000000 | 1.000000 | 0.000000 | 0.888817 | 0.953819 | 0.887200 | 2.182251 |

At matched observable compute with zero added learned parameters, typed routing
improves Delta-IoU by `+0.048656` (`[0.045549, 0.051844]`) and lowers GED by
`0.061396` (`[-0.070432, -0.052844]`). Typed routing consumes more latency and
is not presented as an efficiency contribution.

Robustness evidence:

- Strict arm: five-layer outside exactness `1.0`, violation rate `0.0`, across
  12,395 rows and 7,345 single-room controls.
- Removing edit gating lowers room-level exactness to `0.486325` and raises
  violation rate to `0.013885`, demonstrating evaluator sensitivity.
- All 26 deterministic protocol attacks are sensitive, with observed
  false-pass and false-fail rates `0.0`.
- Wall-adjacent, opening-covering, and exterior-boundary stress suites retain
  gated outside exactness `1.0`; mean conditional watertightness is `0.969-0.972`.

## Structural And Slice Boundaries

Aligned HouseMind-style structure reaches Node F1 `0.982250`, edge overlap
`0.929475`, and GED `0.869544`, below HouseMind-E's published `0.998`, `0.934`,
and `0.467`. Graph metrics are therefore secondary diagnostics, not SOTA claims.

The main risk slices are:

| Slice | Delta-IoU | Watertight | Node F1 | GED | Interpretation |
|---|---:|---:|---:|---:|---|
| room count 0-4 | 0.977269 | 0.930435 | 0.959899 | 0.266221 | Lowest watertight stratum |
| no wall/opening interface | 0.997134 | 0.857143 | 0.713043 | 3.448980 | Weak topology evidence |
| large edit fraction | 0.994355 | 0.966385 | 0.868032 | 3.239178 | Graph tail |

Exact outside preservation and FP-Eval validity remain `1.0` in every complexity
stratum. Independent FEVP validity and conditional watertightness remain
separate metrics.

## Evidence Index

| Claim | Authoritative source |
|---|---|
| Current main metrics | `outputs/reports/residential_edit_test503_v1.json` |
| HouseMind-style structure | `outputs/reports/residential_structural_hardened_v2.json` |
| Matched local baseline | `outputs/reports/residential_editing_aware_baseline_v1.json` |
| FEVP mechanism | `outputs/reports/fevp_mechanism_pareto_v2.json` |
| FEVP attacks | `outputs/reports/fevp_protocol_attacks_v2.json` |
| FEVP stress/locality | `outputs/reports/fp_strict_locality_stress_fiveseed_v1.json` |
| Complexity strata | `outputs/reports/fp_complexity_strata_fiveseed_v1.json` |
| Claim gate | `outputs/reports/residential_sota_claim_gate_v5.json` |

## Locked Boundaries

- No complete-method HouseMind, RPLAN, full-CAD, or all-protocol SOTA.
- No claim that aligned Node/Edge/GED exceed HouseMind-E.
- No deployable predicted-control or free-text claim from oracle controls.
- No move-wall, resize, sketch, or global-recomposition claim.
- No pooling of aligned, canonical-instance, and typed graph protocols.
- No use of missing external metrics as zeros or failures.
