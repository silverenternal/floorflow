# CVPR Chip Experiment Package v1

Date: 2026-11-01

## Terminal Class

Venue gate: `mechanism_ready`.

Package class:
`absolute_budget_mechanism_with_negative_external_transfer`.

This is a complete experiment package for a mechanism-focused CVPR submission.
It is not a FloorSet or external SOTA package.

## Finalist

`m75_seed_diverse_guarded`

- target-free legality-guarded geometry consensus;
- M48 inference seeds 19/20/21;
- deterministic M61B/M65 candidates;
- 10 trajectories;
- total NFE 14;
- refiner steps 16,000;
- 15 legalizer calls;
- 6,797,217 parameters, 16.06% above Trinity d192;
- feasibility 1.0;
- zero solver escalation.

## Primary Evidence

Against three Trinity d192 NFE4/refiner3200 anchors:

- cost mean better against all three anchors;
- V_rel mean better against all three anchors;
- case-cluster cost `-1.57%`, 95% CI `[-2.99%, -0.43%]`;
- case-cluster V_rel `-51.24%`, 95% CI `[-1.43%, -0.63%]`.

The case-cluster interval is the release CI because the same candidate is paired
with three anchors. The naive 300-pair interval is retained only as a diagnostic.

## Structural Evidence

- grouping improves with a pooled interval excluding zero;
- boundary improves with a pooled interval excluding zero;
- p2b HPWL gap improves with a pooled interval excluding zero;
- the 110-block cost tail is substantially rescued;
- area still regresses significantly;
- b2b mean remains worse despite an interval crossing zero.

## Negative Evidence That Must Stay in the Paper

1. Equal-inference and equal-quality efficiency claims fail.
2. M76 candidate reranking fails grouped case-out validation.
3. M56 legal-parent reflow preserves locality but loses to the retained M48
   portfolio on Official-100.
4. The external 8-design/18-pair cohort is DRC-clean but routed wirelength
   regresses `+0.656%` and congestion regresses `+2.466%`.
5. The external cohort is not a new 15-20 design confirmation.
6. Training-seed stability is not established by inference-seed replication.

## v196-v203 Rescue Audit

The post-freeze rescue screens do not replace the M75 finalist:

- isolated area and symmetric/degree b2b heads regress boundary or grouping;
- port consensus regresses grouping;
- the trained boundary anchor improves raw boundary but legalization inverts the
  gain on the 500-case dev curve;
- expanded, shape-preserving and low-trust legalizer policies all fail the joint
  cost/V_rel/area/b2b gate;
- state-conditioned dynamic graph input improves WL, area, boundary and overlap
  at 300 updates, but grouping regresses and a 2k update extension collapses
  total by 5.85%;
- channel/fusion staged training is diagnostic infrastructure only: all routes
  regress quality and the cached b64 route is slower end to end.

These are mechanism falsifications, not tuning failures to hide. They motivate a
future paper on grouping-preserving graph routing and legalizer-compatible
endpoints; they do not strengthen the current SOTA claim.

## Correct Claim

> In a shared CFM plus exact-area decoder, moving graph conditioning into
> per-block trajectories improves grouping, b2b-connected structure, p2b
> geometry and V_rel, with the strongest gains in the cost tail and structural
> relation metrics. Under a fully disclosed multi-trajectory absolute budget,
> target-free candidate consensus beats three Trinity d192 anchors in mean cost
> and V_rel while preserving legality. The method does not establish
> same-inference efficiency, area dominance, training-seed stability or
> routed-PPA transfer.

## Unsupported Claims

- FloorSet SOTA;
- external routed-PPA SOTA;
- same-inference or same-compute superiority;
- universal per-case or per-seed gains;
- area or b2b superiority;
- MIB or opening-specific improvement;
- surrogate HPWL-to-PPA transfer.

## Reproducibility

- full pytest: `1571 passed`;
- scoped Ruff: pass;
- first-party Python: every file below 600 lines;
- exact artifact hashes are in the venue gate report;
- no generated checkpoint, dataset or report is intended for git commit.

## Artifact Index

- `outputs/reports/chip_cvpr_venue_gate_v1.json`
- `outputs/reports/chip_m74_cross_checkpoint_seed_report_v1.json`
- `outputs/reports/chip_m75_decision_v1.json`
- `outputs/reports/chip_s1_named_budget_certification_v1.json`
- `outputs/reports/chip_s1_equal_quality_pareto_v1.json`
- `outputs/reports/chip_m76_dev_selection_v1.json`
- `outputs/reports/chip_m76_dev_selection_v2.json`
- `outputs/reports/chip_m56_reflow2_decision_official100_v1.json`
- `outputs/reports/chip_e1_external_holdout_v1.json`
- `outputs/reports/chip_operator_profile_v2.json`
- `outputs/reports/chip_operator_equivalence_v2.json`

## Stop Condition

The experiment package is complete for the mechanism claim. Further work should
start a new research goal focused on the area gap or a fresh external cohort; it
should not relabel the current package as SOTA.
