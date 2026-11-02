# Candidate Consensus Scale Gate v1

## Pilot Status

The 32-case pilot is a wiring test, not promotion evidence.

- 4 candidate views: M48-s19 baseline, M48-s20 expanded, M61B projector32, M65 baseline.
- 24 disjoint training cases and 8 held-out cases.
- 4 candidates on every case.
- Best-single winner counts: `[16, 4, 3, 9]`.
- Reference wins on 50% of cases.
- Mean pairwise candidate latent L2 distance: `2.429750`; p95: `4.256138`.

The full path now runs as:

```text
legalized candidate states
  -> learned consensus
  -> exact-area decode
  -> hard-constraint projection
  -> scale_pack legalization
  -> official case metrics
```

## Scale Gate

A 500-case cohort may be promoted only if all conditions hold:

1. At least 4 valid candidate views per case.
2. Candidate-state export contains `gt_boxes`, `areas`, `constraints`, `b2b`, `p2b`, `pins`, `raw_xywh`, and `legalized_xywh`.
3. Training and evaluation case indices are disjoint.
4. At least two distinct candidates produce the best single legalized cost on the cohort.
5. No target-derived feature or official cost is used at inference.
6. Consensus is compared with both the fixed reference and the best legalized single candidate.
7. Cost and V_rel are reported jointly with case-level paired intervals.
8. K <= 4 consensus must beat the matched single-trajectory budget after all projection and legalizer calls are counted.

The 8-case held-out result is retained as `wiring_only` and must not be cited as a
quality result.

## Held-out Wiring Result

- reference mean cost / V_rel: `1.16685 / 0.04814`;
- best-single mean cost / V_rel: `1.12975 / 0.04021`;
- learned consensus mean cost / V_rel: `1.34643 / 0.05376`;
- feasibility is `1.0` for all three;
- reference/best-single/consensus mean legalizer rungs are `2.125/1.875/2.5`.

This confirms the full path executes, but it rejects the current 24-case teacher
as evidence of consensus quality. The next scale run must use at least 500 cases
and a case-disjoint fit, with no promotion from this pilot.

## Sparse Selection Redesign

Whole-layout averaging is disabled for the next pilot. The replacement starts
from a hard top-k candidate selected by
`CandidateLegalizabilityReranker`, uses straight-through gradients for listwise
legalized ranking, and adds a residual bounded to
`abs(delta) <= residual_limit` around the selected state. Candidate 0 remains a
valid reference/fallback. At initialization the residual gate is zero, so the
model exactly returns a real valid candidate rather than a synthetic average.

The redesigned selector must pass the same 8-case disjoint pilot before any
500-case cohort generation.

## Sparse Pilot Result

With `top_k=1`, `residual_limit=0.05` and the same 24/8 split:

- reference mean cost / V_rel: `1.16685 / 0.04814`;
- best-single mean cost / V_rel: `1.12975 / 0.04021`;
- sparse consensus mean cost / V_rel: `1.14377 / 0.04500`;
- reference/best-single/sparse-consensus legalizer rungs:
  `2.125 / 1.875 / 2.0`;
- all layouts remain feasible.

The redesign removes the catastrophic averaging regression and beats the fixed
reference, but it has not matched the per-case oracle single candidate and only
8 held-out cases are available. It is a scale candidate, not a quality result.

## Leakage Repair and Target-Free Selector

The first 500-case candidate feature set contained legalized V_rel, grouping,
boundary and MIB values. Those metrics require ground truth and are not
deployable. The four target-derived columns were removed; the corrected feature
vector has 11 columns: seven geometry/complexity features plus legalizer rung,
solves, reshape count and raw-to-legalized displacement.

On a deterministic 408/92 case split, a policy chosen only on the 408 training
cases, `min_b2b`, improved the fixed reference on the 92 held-out cases:

- cost `-7.26%`;
- V_rel `-17.90%`;
- area gap `-50.78%`;
- b2b HPWL gap `-65.50%`;
- grouping `-28.28%`;
- boundary `-9.78%`.

V_rel, area, b2b and grouping paired intervals exclude zero. Cost improves in
mean but its interval crosses zero. The selector achieves 46 constraint-safe
joint wins out of 92 cases versus the reference.

This is a retained mechanism candidate, not a promoted result. The same exact
policy is now being tested once on a fresh 500-case holdout with zero overlap
against the split used for policy selection. The frozen 92-case holdout must not
be reused for tuning.

## 500-Case Learned Selector Result

The initial 15-dimensional learned selector result is invalid because four of
its features use ground truth. After removing those columns, the corrected
11-dimensional learned selector fails the same 92-case holdout:

- consensus cost: `1.16592`;
- fixed reference cost: `1.19590`;
- fixed M61B cost: `1.14968`;
- legalizer rung: `2.05`;
- oracle best-single agreement: `5.43%`.

The learned selector remains closed on this cohort.

## Fresh 500-Case Confirmation

The frozen target-free `min_b2b` policy was evaluated once on a new 500-case dev
holdout with zero overlap against the policy-selection cases:

- cost `-7.05%` versus the fixed reference and `-6.94%` versus the best fixed view;
- V_rel `-13.90%` and `-14.24%`;
- area gap `-48.28%` and `-51.55%`;
- b2b gap `-64.90%` and `-68.32%`;
- p2b gap `-83.33%` and `-84.69%`;
- boundary `-18.78%` and `-10.59%`.

Paired intervals exclude zero favorably for cost, V_rel, area, b2b, p2b,
boundary, legalizer rung and legalizer solves against both fixed baselines.
Grouping excludes zero against the best fixed view; its interval crosses zero
against the reference.

Joint success is `399/500` versus the reference and `403/500` versus the best
fixed view. Constraint-safe joint wins are `249/500` and `209/500`.

This is a confirmed internal mechanism candidate. Trinity anchor comparison,
training-seed replication and fresh external PPA remain required.

## Trinity Anchor Comparison

On the same fresh 500 cases:

| Budget | Candidate cost | Trinity cost | Candidate V_rel | Trinity V_rel |
| --- | ---: | ---: | ---: | ---: |
| portfolio NFE16/ref1600 vs Trinity NFE4/ref1600 | 1.12190 | 1.15615 | 0.03347 | 0.03162 |
| portfolio NFE16/ref1600 vs Trinity NFE16/ref1600 | 1.12190 | 1.14814 | 0.03347 | 0.03027 |

The selector improves cost `-2.96%` and `-2.29%`, area, b2b and p2b, but
regresses V_rel by `+5.86%` and `+10.57%`, grouping by `+20.85%` and `+16.79%`,
and rho MAE against both anchors. Only `217/500` cases pass the joint primary
gate at the equal total NFE/refiner budget, with `48/500` constraint-safe wins.

A target-free safety guard reduces rho error but further reduces cost gain and
still regresses V_rel/grouping. The selector is therefore a structural-transfer
mechanism, not a SOTA candidate.
