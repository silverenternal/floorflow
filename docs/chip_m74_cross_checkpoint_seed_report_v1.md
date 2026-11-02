# M74 Cross-Checkpoint Portfolio Seed Report

Date: 2026-11-01

## Decision

M74 passes the mechanism gate but fails the stable-cost/SOTA gate:

- feasibility is `1.0` in all three inference seeds;
- the joint primary direction repeats in two of three seeds;
- pooled V_rel improves `50.50%` with a paired CI excluding zero;
- pooled cost improves `0.62%`, but its paired CI crosses zero;
- area and b2b remain significantly worse than the matched Trinity d192 anchor.

The supported claim is therefore:

> In the same CFM and exact-area decoder, moving graph conditioning into
> per-block trajectories is associated with stable grouping, boundary, p2b
> and total V_rel gains. The multi-checkpoint portfolio also lowers mean cost,
> but cost stability is not established.

Do not claim SOTA, same-inference superiority or a universal cost win.

## Per-Seed Results

Negative is better. Intervals are paired percentile bootstrap intervals
against each seed's matched Trinity d192 NFE4/refiner3200 anchor.

| Seed | Candidate cost | Trinity cost | Cost change | Cost 95% CI | V_rel change | V_rel 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 19 | 1.050223 | 1.052969 | -0.26% | [-2.69%, +2.94%] | -52.38% | [-1.62%, -0.55%] |
| 20 | 1.050354 | 1.045964 | +0.42% | [-1.89%, +3.52%] | -41.23% | [-1.28%, -0.16%] |
| 21 | 1.049441 | 1.070648 | -1.98% | [-4.54%, -0.11%] | -56.32% | [-1.68%, -0.77%] |
| Pooled | 1.050006 | 1.056527 | -0.62% | [-2.08%, +0.92%] | -50.50% | [-1.31%, -0.72%] |

Seed 21 passes the paired cost interval. Seeds 19 and 20 do not. This is
heterogeneity, not evidence that can be removed by averaging.

## Structural Result

Pooled paired deltas versus Trinity:

| Metric | Mean delta | 95% CI | Reading |
|---|---:|---:|---|
| Grouping | -0.3433 | [-0.4233, -0.2600] | stable improvement |
| Boundary | -0.1033 | [-0.1833, -0.0233] | stable mean improvement |
| p2b HPWL gap | -0.04065 | [-0.07045, -0.02007] | stable improvement |
| Rho/aspect MAE | -0.02499 | [-0.03197, -0.01789] | stable improvement |
| Area gap | +0.03007 | [+0.02066, +0.04186] | stable regression |
| b2b HPWL gap | +0.00995 | [+0.00481, +0.01599] | stable regression |

The mechanism evidence is structural and robust. It is not a complete PPA win.

## Failure Tail

The cost tail is driven by legalization failures on a small number of
high-complexity cases:

- seed 19 and seed 20 both have a catastrophic case index `89` with 110 blocks;
- seed 20's selected candidate has cost `2.2070` versus Trinity `1.0385`;
- no available M48 seed20 candidate rescues this case;
- seed 21's M48 projector32 candidate reaches cost `1.4506` on the same case;
- seed 21 instead has a different worst case, index `0`, while its mean cost
  remains better than Trinity.

The 81-120 block strata are not the source of the V_rel gain. Area gap remains
worse than Trinity in every 20-block stratum. Any claim must keep the robust
V_rel/grouping/p2b mechanism separate from the unresolved area/b2b/cost-tail
problem.

## Budget

The displayed result uses six trajectories, nine legalizer calls, total NFE
`10` and total refiner steps `9600`. It is not a same-inference comparison with
the one-trajectory Trinity anchor.

## Required Repairs

1. Add a tail-safe target-free selector using candidate geometry consensus,
   not target boxes or post-route outcomes.
2. Test seed-diverse M48 trajectories on the catastrophic tail; seed 21 shows
   that a good legalizable basin exists for case 89.
3. If selection cannot rescue the tail, add a high-complexity area/b2b repair
   path before any SOTA claim.
4. Keep area/b2b regression and the pooled cost interval across zero in every
   paper-facing limitation.

## Artifacts

- `outputs/reports/chip_m74_cross_checkpoint_seed_report_v1.json`
- `outputs/reports/chip_m74_tail_seed19_v1.json`
- `outputs/reports/chip_m74_tail_seed20_v1.json`
- `outputs/reports/chip_m74_tail_seed21_v1.json`
- `outputs/reports/chip_m74_cross_checkpoint_seed20_official100_v1.json`
- `outputs/reports/chip_m74_cross_checkpoint_seed21_official100_v1.json`
