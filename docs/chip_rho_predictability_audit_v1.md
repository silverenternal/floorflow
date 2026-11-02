# Chip Rho Predictability Audit v1

Date: 2026-11-01

## Question

Can the existing four legalized candidates be reranked or locally reshaped to
close the legalized `rho_mae` gap without target-derived inference features?

## Candidate-Level Signal

Across the fresh 500-case four-view cohort, the best-rho view is:

- candidate 0: 190 cases;
- candidate 1: 83;
- candidate 2: 98;
- candidate 3: 129.

Target-free features have substantial global correlation with legalized rho:

| Feature | Spearman vs rho MAE |
| --- | ---: |
| shape displacement | +0.727 |
| legalization displacement | +0.716 |
| outline area | +0.494 |
| legalizer reshaped | +0.440 |

On the 408/92 case-disjoint split, Ridge candidate ranking reaches:

- pair accuracy `71.38%`;
- top-1 accuracy `46.74%`;
- mean within-case Spearman `0.483`;
- selected rho `0.22816`, candidate0 `0.24076`, oracle `0.21323`.

The rho ordering is therefore learnable, but reranking cannot create a
candidate outside the existing pool.

## Why Reranking Is Not Deployable

The rho-only selector versus Trinity NFE16/ref1600 gives:

- cost `-0.23%`, interval crossing zero;
- V_rel `+18.83%`;
- grouping `+28.57%`;
- rho `+2.89%`;
- joint/strict/safe wins `36/19/6` out of 92.

The existing four-view pool has only `23/92` cost-V_rel-grouping-rho oracle
cases, `6/92` all-metric oracle cases and `15/92` constraint-safe oracle
cases. Better ranking cannot reach the `35%` coverage gate.

## Shape Variant Screen

All variants preserve centers, exact areas and fixed/preplaced shapes:

- square rho: no oracle gain;
- aspect flip: no oracle gain;
- MIB median: no oracle gain;
- global median: no oracle gain;
- cluster median: joint `62→63`, strict `37→40`, safe `15→16`.

Applying cluster-median shape to all four candidates yields:

- cost-V_rel-grouping-rho oracle `24/92`;
- all-metric oracle `7/92`;
- constraint-safe oracle `16/92`.

Deterministic shape priors are insufficient.

## Residual Shape Probe

A per-block ExtraTrees residual head, trained only on target-free input and
candidate0 rho, improves raw block rho MAE from `0.23305` to `0.22377` on the
92-case validation split. After the exact-area projector and legalizer,
however, legalized rho regresses `+13.06%`; the shape candidate adds only one
four-metric oracle case and does not increase safe coverage.

A constraint-safe classifier trained on 408 candidate-vs-candidate0 safe
labels also fails to clear its probability thresholds and chooses candidate0
on all 92 validation cases.

## Direct Rho-Loss Probe

A single-variable `rho_weight=0.25` run from the retained backbone used the
frozen 1k subset and 300 steps. On the 32-case quick screen:

- rho MAE changed from `0.124162` to `0.124036` (-0.10%);
- grouping regressed from `0.318092` to `0.330235` (+3.81%);
- total objective regressed from `0.016796` to `0.016930` (+0.80%).

The rho gain is negligible and the structural/legality tradeoff is
unacceptable, so the run was not extended to 2k.

## Decision

Fail closed. Do not deploy a rho-only selector, residual shape head, deterministic
aspect rule or constraint-safe classifier. The next useful mechanism must
generate a structurally different shape candidate and evaluate legality after
the legalizer; further selector tuning is forbidden by the candidate-pool
ceiling.

## Artifacts

- `src/floorflow/chip_flow/rho_predictability.py`
- `scripts/audit_chip_rho_predictability.py`
- `tests/test_chip_rho_predictability.py`
- `outputs/reports/chip_rho_predictability_val92_v1.json`
- `outputs/reports/chip_rho_cluster_median_all4_val92_v1.json`
- `outputs/reports/chip_rho_residual_shape_val92_v1.json`
