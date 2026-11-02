# CVPR Experiment Status

Status: historical snapshot. The current submission-facing status is
`docs/cvpr_experiment_status_20261102.md`; the current oracle-control protocol
summary is `docs/fp_oracle_control_protocol_status_v1.md`.

Status date: 2026-10-26. This document tracks experiment evidence, claim
boundaries, and reproducible commands. It is not a manuscript draft.

## Selected Direction

Primary route: legal boundary-conditioned generation plus identifiability
analysis on CubiCasa. The dynamic hierarchical chip arm was tested as route B
and failed its promotion gate, so it is not the primary claim. The existing
three-seed global-slot transfer remains a secondary cross-domain result.

Do not claim exact interior reconstruction, 0.95 room IoU, style control,
FloorSet SOTA, or raw placement feasibility without legalization.

## Frozen Residential Evidence

The FP-C01 control contract is frozen in
`configs/fp_control_contract_v1.json`. Oracle controls are serialized under
`outputs/control_contracts/oracle_v1/` with per-file hashes and recoverable
information content. On test503, the control alone reaches room macro IoU
`0.76188`, room-support IoU `0.83109`, wall IoU `0.66083`, opening IoU
`0.66113`, and furniture IoU `0.86478`. These controls are target-derived and
therefore remain an upper-bound diagnostic only.

The target-free CubiCasa test503 pipeline supports the following measured
full-pipeline statement:

- schema validity: `1.0000`
- conditional watertightness: `0.98649`
- room target recall: `0.99602`
- pixels outside footprint: `0`
- room macro IoU: `0.17034`
- furniture IoU: `0.06027`
- fpeval opening F1: `0.11962`

The legality numbers are the headline evidence; IoU is a limitation. The
identifiability audit finds a `0.00596` strict-0.95 train-neighbour fraction and
mean nearest-neighbour room IoU of `0.10044`, which frames exact boundary-to-
interior reconstruction as ill-posed.

Sources:

- `outputs/reports/final_gated_targetfree_cad_test503_v1.json`
- `outputs/reports/d0_identifiability_exact_ambiguity_test503_v2.json`
- `outputs/reports/metric_hierarchy_test503_v5.json`

## Distributional Generation Audit

The contour-only model was evaluated on 501 test conditions with four samples per
condition. Unique non-diagonal diversity is `10.8504`, compared with target
cross-plan diversity of `13.4410` (ratio `0.8073`).

Coverage and density are both `0.0`. This is a retained limitation: sample
differences exist, but they do not cover the target distribution at the
pre-registered radius. Do not present diversity alone as successful distribution
matching.

Sources:

- `outputs/reports/distributional_generation_test503_contour_only_v4_multisample.json`
- `outputs/reports/distributional_generation_gate_v1.json`

## Constraint Attribution

Paired test503 stages using one checkpoint:

| Stage | Room IoU | Validity | Conditional watertight |
|---|---:|---:|---:|
| raw flow | 0.11009 | n/a | n/a |
| structured room readout | 0.16551 | n/a | n/a |
| overlay without legalization | 0.17034 | 1.00000 | 0.98649 |
| overlay with fixed legalization | 0.17034 | 1.00000 | 0.98649 |

The original legalizer deleted off-wall opening labels and reduced conditional
watertightness to `0.13514`. The fix preserves footprint-valid openings and
filters only impossible furniture. Val64 radius sweeps `2/4/6` confirmed that
wall-proximity opening deletion, not the radius, caused the regression. The
fixed legalizer is safe but adds no quality gain, so it is not a headline claim.

Sources:

- `outputs/reports/constraint_attribution_test503_v1.json`
- `outputs/reports/final_attribution_no_legalize_test503_v1.json`
- `outputs/reports/final_legalized_fixed_attribution_test503_v2.json`

## Route B Substrate

FloorSet-Lite has 50,000 packed train cases and 12,000 packed dev cases with the
five native constraints: fixed, preplaced, MIB group, cluster, and boundary.
Physics parity is `1.0` for boundary, MIB, and grouping; Trinity official
scorer/legalizer integration is complete.

Source: `outputs/reports/chip_route_b_substrate_v1.json`.

The existing global-slot transfer has three positive seeds:

- mean legalized hard-cost relative delta: `-0.33664`
- mean V_rel relative delta: `-0.50177`
- hard cost and V_rel improve on all three seeds

This is a transfer result, not yet the full dynamic-hierarchy arm. The
hierarchical arm reuses `soft_slot_decompose`, sparse token attention,
global-latent CFM, exact-area decode, and FloorSet physics.

The matched 20k hierarchy promotion test failed. On official FloorSet test100,
hierarchical slot reached legalized hard cost `2.40915` and V_rel `0.14385`,
versus `2.28065` and `0.12755` for global slot. The hierarchy also uses 7.32M
parameters versus 5.49M. Its paired hard-cost interval crosses zero and V_rel
worsens with a positive interval. This negative result must remain in the
ablation table; it is not the main claim.

## Contemporary Baseline Availability

GSDiff's official code and boundary entrypoint are present at commit
`a02af021b80eba984228f92f7cf5b6f296cf3725`, but its RPLAN tree and official
weights are absent. RPLAN access is gated by the authors' Google Form. FMLM and
TLC-Plan do not currently expose runnable code/checkpoints on their public
project pages. No native or cross-dataset baseline number has been fabricated.

Source: `outputs/reports/contemporary_boundary_baseline_audit_v1.json`.

## Completion Status

The available experiment package is complete. All 460 tests pass, all 646 Python
files are below 600 lines, and the final evidence package is at
`outputs/cvpr_package_v8_reframe/submission_checklist.json`.

Known limitations are explicit: contour-only distribution coverage is zero,
style control is excluded, the dynamic hierarchy promotion failed, and a
matched contemporary RPLAN baseline requires external data authorization.

Current detailed ordering and status: `todo.json`.
