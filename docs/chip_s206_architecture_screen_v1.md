# Chip SOTA Architecture Screen v1

Date: 2026-11-01

## S1: Group-Moment Families

Three opt-in variants were implemented and tested before GPU screening:

1. a single pooled moment residual;
2. typed cluster/MIB/boundary moment paths with an early-explore/late-commit schedule;
3. a native 7D structural-moment hierarchy state.

All variants preserve M27 identity when disabled. The first variant produced a
positive 32-case signal, but failed the 500-case expansion: V_rel `+3.24%`,
legalized boundary `+3.55%`, grouping `+6.08%`, and b2b `+0.58%`.

The typed and native variants failed earlier at the matched 32-case screen.
Native moments added grouping `+3.47%` and total `+0.84%`. The family is paused;
more moment-vector tuning is not justified.

## S2: Legalization Survival

Across three M48 seeds, 2,891 defined raw/legalized candidate pairs have:

- pairwise Legalization Survival Rate: `61.54%`;
- top-choice retention: `61.10%`;
- boundary survival: `44.44%`;
- grouping survival: `52.54%`;
- V_rel survival: `46.27%`.

Raw candidate rankings therefore cannot select checkpoints.

A sidecar-supervised legalizer-response endpoint with a bounded final action was
also rejected: at 2k updates it regressed total `+8.29%`, boundary `+87.46%`,
area `+4.33%`, and grouping `+3.79%` versus the matched head-only control.

## S3: Candidate Consensus

Four real candidate views show strong complementarity on 500 dev cases:
M48-s19 baseline, M48-s20 expanded, M61B projector32, and M65 baseline.

Whole-layout averaging and learned sparse fusion are rejected. The first
learned 500-case selector was also invalidated because four feature columns
(legalized V_rel, grouping, boundary and MIB) required ground truth.

After removing those four columns, the corrected feature vector contains seven
geometry/complexity values plus legalizer rung, solves, reshape count and
displacement. On a 408/92 case split, a policy selected only on train,
`min_b2b`, improved the fixed reference on the 92-case holdout:

- cost `-7.26%`;
- V_rel `-17.90%`;
- area gap `-50.78%`;
- b2b gap `-65.50%`;
- grouping `-28.28%`;
- boundary `-9.78%`.

V_rel, area, b2b and grouping paired intervals exclude zero; cost improves in
mean but its interval crosses zero. The policy wins 46/92 constraint-safe joint
comparisons against the reference.

This candidate is frozen and must be tested once on a new 500-case holdout with
zero overlap against policy-selection cases.

## S4: Route Transfer Inputs

FloorSet exposes pairwise b2b and p2b incidence, not true multi-pin training
hyperedges. External DEF graphs can be converted through the existing
`parse_def_graph` and `external_netlist_hypergraph` paths.

Current ORFS artifacts contain 14 complete routed metric files over six unique
metric designs and 130 final DEFs. The completed external PPA cohort has eight
designs and 18 seed pairs, with zero DRC but negative mean routed WL and
congestion.

This is enough for calibration, not route-transfer training or publication-level
confirmation. A fresh 15-20 design cohort and a provenance manifest keyed by
design, proposal, seed, placement hash and route recipe are required.

## Current Decision

The only new positive candidate is the leakage-corrected target-free `min_b2b`
selector. S1 group-moment variants and the predicted legalizer-action endpoint
are rejected. Route transfer remains blocked on fresh external data.
