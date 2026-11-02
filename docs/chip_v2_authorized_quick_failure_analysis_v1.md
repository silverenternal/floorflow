# Chip v2 Authorized Quick-Screen Failure Analysis

Date: 2026-11-01

## Scope

The six candidates in this report use cuda:0, the frozen 5k training subset
(M54 uses its 32-case response subset), seed 21/data seed 21, 2,000 steps,
micro-batch 128, NFE4 and a 400-step refiner on the frozen 32-case legalized
screen. Seed 19 is the inference seed. All candidates and controls have
feasible fraction `1.0`.

The shared control is M27 resumed with structured heads enabled but hierarchy
commit and factorized geometry disabled. Its NFE4/ref400 metrics are:

| Metric | Control |
|---|---:|
| Hard cost | 1.192676 |
| V_rel | 0.044204 |
| Area gap | 0.103328 |
| b2b HPWL gap | 0.038302 |
| p2b HPWL gap | 0.129165 |
| Grouping | 0.812500 |
| Boundary | 1.562500 |
| Rho MAE | 0.243661 |

## Primary results

Relative changes are candidate/control - 1. Negative is better.

| Candidate | Raw total | Cost | V_rel | Area | b2b | p2b | Grouping | Boundary | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M50 commit | +1.67% | +2.17% | -0.51% | +4.12% | +27.03% | +172.59% | +3.85% | -2.00% | reject |
| M51 factorized | +0.71% | +0.20% | +5.04% | +0.70% | +10.95% | +485.75% | +11.54% | +2.00% | reject |
| M53 endpoint | -2.17% | +3.35% | -5.93% | +36.15% | +69.63% | +779.64% | 0.00% | -6.00% | reject |
| M52 typed graph | -0.08% | +6.52% | +10.24% | +39.29% | +69.16% | +858.15% | +19.23% | +4.00% | reject |
| M55 route/source | +1.01% | -1.78% | +3.66% | -27.80% | -21.70% | -67.59% | +19.23% | 0.00% | reject, split |
| M54 legalizer | +22.80% | +0.91% | +11.31% | +8.73% | +11.93% | +479.09% | +38.46% | +4.00% | reject, overfit |

M54 improved over its own 32-case control (`cost -7.77%`, `V_rel -8.44%`,
`b2b -46.23%`, `area -29.40%`) but both arms were much worse than the shared
5k control. This is evidence of subset overfit, not a deployable gain.

## Structural diagnosis

### M50: hard coarse commitment is unsafe

`coarse_commit_fraction` freezes all L0/L1 states after a fixed trajectory
fraction. It has no confidence estimate, no cross-step stability check, and no
trust radius. A wrong cluster or boundary decision becomes irreversible and
drives p2b from 0.129 to 0.352.

Repair:

1. Replace hard freezing with a per-cluster bounded anchor.
2. Commit only slots whose predicted membership and geometry are stable over
   the previous two or three ODE predictions.
3. Gate commitment by relation-head confidence and active pin pressure.
4. Preserve a late-phase rescue path for each committed cluster.

### M51: factorized heads drift independently

The center and rho MLPs share encoders but have no cross-consistency target.
Rho remains approximately neutral while p2b degrades by 4.86x. Shape changes
therefore move pin geometry without improving the joint legalized layout.

Repair:

1. Use a shared trunk with low-rank center and rho residual branches.
2. Add an exact-area/cross-head consistency loss and a connected-pin gradient.
3. Constrain rho updates with a trust radius tied to current block occupancy.

### M53: endpoint objective optimizes the wrong compatibility

The current endpoint objective uses raw/projected physics and a short
differentiable refiner trajectory. The hard projection itself is detached, and
the objective does not directly model scale_pack rung, legalized area gap or
legalized p2b. The raw total improves by 2.17%, but legalized cost rises 3.35%
and p2b rises 7.8x.

Repair:

1. Add rung/solve prediction and post-legalization displacement to checkpoint
   selection and the training objective.
2. Add legalized area-gap and p2b proxies, not only refiner energy.
3. Use a straight-through/trust-region bridge through a few projector steps.
4. Reject an endpoint if it worsens legalized cost or p2b, regardless of raw
   validation total.

### M52: relation paths amplify port incidence

The relation-specific path adds each family with an independent scalar gate but
does not normalize by active incidence or constrain each family's residual.
Port edges can dominate, consistent with the `+858%` p2b and broad geometric
regression.

Repair:

1. Normalize relation attention and residuals by active incidence.
2. Add per-family trust radii and a relation-gate entropy floor.
3. Allow port evidence only for block queries, never global/structural queries.
4. Require a matched port-off variant to improve cost/V_rel before promotion.

### M55: route field and source schedule are confounded

M55 combines the route-aware spatial field, mixed source annealing, dihedral
augmentation and backbone unfreeze. It improves legalized cost `-1.78%`, b2b
`-21.7%`, area `-27.8%` and p2b `-67.6%`, but worsens V_rel `+3.66%`,
grouping `+19.2%` and rho `+3.2%`. The route field is also only an input
feature; `routing_proxy_weight` remains zero, so nothing directly penalizes
congestion in training.

Repair:

1. Split route-only and source-only matched arms.
2. Add a differentiable route proxy or test-time displacement objective.
3. Preserve V_rel/grouping with a structural constraint before combining route
   and endpoint-compatible training.

### M54: 32-case sidecar overfits

The response sidecar is high quality but covers only 32 cases. Repeating this
subset for 2,000 steps overfits severely and changes the deployment manifold.
The response loss is also global rather than post-legalization outcome-aware.

Repair:

1. Build a stratified CPU sidecar with at least 1,000 cases.
2. Hold out sidecar cases from response-head validation.
3. Add rung calibration, displacement-tail weighting and response-head decay.
4. Apply the legalizer action only at the final sample step and compare against
   the same 5k-trained control.

## Repair-stage follow-up

The repair program was screened under the same 2k/NFE4/ref400 protocol. M57
through M61 and M65 were rejected; M64 is the only candidate promoted.

| Candidate | Cost | V_rel | Area | b2b | p2b | Grouping | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| M57 confidence anchor | +7.71% | +3.50% | +35.39% | +66.40% | +537.82% | +11.54% | reject |
| M58 shared geometry | +0.66% | +2.20% | +10.16% | +17.25% | +494.46% | +15.38% | reject |
| M59 endpoint constraints | +5.08% | -1.92% | +39.11% | +61.40% | +252.56% | +7.69% | reject |
| M60 normalized graph | +3.45% | -2.11% | +28.72% | +50.50% | +736.69% | 0.00% | reject |
| M61A route only | +3.44% | +27.02% | +6.68% | +11.53% | +604.80% | +26.92% | reject |
| M61B confounded source | -2.92% | +2.99% | -32.26% | -35.95% | -118.22% | +23.08% | reject |
| M64 clean source | -4.56% | -10.14% | -38.31% | -39.91% | -100.37% | 0.00% | **confirm** |
| M65 graph + source | -1.21% | +6.42% | -20.17% | -6.03% | -88.67% | +15.38% | reject |

### M64 is the new confirmation parent

M64 removes the route field, routing proxy, dihedral augmentation, backbone
unfreeze and graph fusion from M61B. It retains only the mixed/derived source
schedule. The legalized screen is a joint primary win and preserves grouping:

- hard cost `1.138233` versus `1.192676` control;
- V_rel `0.039720` versus `0.044204`;
- area gap `-38.31%`, b2b `-39.91%`, p2b `-100.37%`;
- grouping unchanged and boundary `-18.00%`;
- feasibility `1.0`.

The 2k run does not reach the declared final shelf probability at step 4,000.
It is therefore exploratory evidence, not a publishable final result. The next
gate is a 5,000-step matched control/candidate confirmation followed by the
declared NFE4/refiner1600 endpoint.

### What the failed repairs establish

- Incidence normalization alone does not make port-heavy graph evidence safe.
- Endpoint and route proxies still optimize quantities that reverse after
  legalization.
- Graph/source fusion is not complementary at the current training budget:
  M65 loses M64's V_rel gain and erases most of its cost gain.
- The strong M61B area/b2b/p2b signal is real enough to preserve, but its
  V_rel/grouping failure means the source manifold, not the route field, is
  the useful isolated ingredient.

M62 remains blocked because its endpoint and route parents did not pass. It
must not be run as a blind all-module fusion.

## M64 confirmation failure

The formal matched confirmation used 5,000 steps for both arms with the same
5k training subset, effective batch, parameters, seed/data seed, NFE4/refiner400
and legalizer protocol. Feasibility remained `1.0` in both arms.

| Metric | 5k control | M64 5k candidate | Relative |
|---|---:|---:|---:|
| Hard cost | 1.185559 | 1.327701 | +11.99% |
| V_rel | 0.042405 | 0.056851 | +34.07% |
| Area gap | 0.098783 | 0.167100 | +69.16% |
| b2b HPWL gap | 0.036468 | 0.115134 | +215.71% |
| p2b HPWL gap | 0.143135 | 0.713260 | +398.31% |
| Grouping | 0.718750 | 1.312500 | +82.61% |
| Boundary | 1.593750 | 1.531250 | -3.92% |
| Rho MAE | 0.272381 | 0.319794 | +17.41% |

The raw 100-case validation total for M64 rose from `0.01686` at 2k to
`0.03197` at 5k, while the control rose from `0.01653` at 2k to `0.01808`
at 5k. The failure is therefore not legality or refiner instability: it is
model quality after source adaptation and long training. Refiner1600 is
blocked.

### Source-fraction by budget 2x2

The existing evidence occupies two diagonal cells. The missing cells isolate
the failure mechanism before any new architecture is attempted.

| Source policy | 2k | 5k |
|---|---|---|
| mixed/derived p=.25 to .50 | M64 pass at ref400 | not run |
| mixed/derived p=.25 to .75 | not run | M64 confirmation fail |

The next two probes are fixed `p=.75` at 2k and fixed `p=.50` at 5k. If both
fail while p=.50/2k passes, the source-adaptation family is not robust to a
usable training budget and should be closed rather than tuned.

### Completed 2x2 result

| Cell | Cost delta | V_rel delta | Raw-total delta | Decision |
|---|---:|---:|---:|---|
| 2k, p=.50 | -4.56% | -10.14% | +1.93% | pass |
| 2k, p=.75 | +0.30% | +10.07% | +7.53% | fail |
| 5k, p=.50 | +12.77% | +50.02% | +79.55% | fail |
| 5k, p=.75 | +11.99% | +34.07% | +76.77% | fail |

The interaction is decisive. Increasing shelf probability is harmful even at
2k, and both source candidates collapse after 5k while the matched random-source
control remains comparatively stable. Source scaling on this 5k training subset
is therefore closed. The only surviving mechanism is the 2k/p50 cell, pending
inference-seed and then training-seed certification.

## Seed certification

The frozen 2k/p50 M64 candidate improved both primary metrics over its matched
control at inference seeds 19, 20 and 21:

| Inference seed | Cost | V_rel |
|---|---:|---:|
| 19 | -4.56% | -10.14% |
| 20 | -2.01% | -14.64% |
| 21 | -5.77% | -18.71% |

The deployment source is deterministic, so the candidate output itself is
identical across inference seeds; the random-source control varies and remains
worse on both primaries.

Training-seed replication passes the preregistered 2-of-3 gate but is
heterogeneous:

| Training seed | Cost | V_rel | Result |
|---|---:|---:|---|
| 19 | -0.13% | +1.18% | fail; p2b/grouping also regress |
| 20 | -4.98% | -0.91% | pass; grouping regresses |
| 21 | -4.56% | -10.14% | pass; grouping preserved |

Seed21 is the strongest finalist for Official-100. Seed19 is retained as a
negative replication; universal training-seed stability is not claimed.

## Official-100 reversal

The seed21 M64 candidate failed its first out-of-screen test:

| Arm | Cost | V_rel | Feasible |
|---|---:|---:|---:|
| M64 2k/p50 candidate | 1.218301 | 0.042937 | 1.0 |
| Matched control | 1.195814 | 0.038636 | 1.0 |
| Trinity d192, NFE4/ref400 | 1.170112 | 0.037008 | 1.0 |

Against control, M64 regressed cost `+1.88%` and V_rel `+11.13%`. Against the
same-budget Trinity anchor it regressed `+4.12%` and `+16.02%`. The paired
intervals cross zero, but both means reject the source candidate and 56/100
cases worsened in cost. Refiner1600 was blocked.

The important conclusion is screen validity, not a new architecture failure:
the 32-case response-selected screen can reverse on Official-100. Future model
decisions must use Official-100 before promotion, even when the small screen
contains three inference seeds and two positive training seeds.

### Complexity strata

The reversal is strongly complexity-dependent:

| Blocks | Cases | Cost vs control | V_rel vs control |
|---|---:|---:|---:|
| 21-40 | 20 | -1.30% | -14.22% |
| 41-60 | 20 | +4.29% | +24.94% |
| 61-80 | 20 | +8.72% | +37.85% |
| 81-100 | 20 | -1.45% | +5.15% |
| 101-120 | 20 | +0.58% | +8.74% |

The source mechanism helps the smallest layouts but catastrophically regresses
mid-complexity layouts. This rules out a simple global source policy. A future
source/control router would need a target-free complexity rule and must be
certified on Official-100, not selected post hoc on these cases.

## Matched neural-budget M48 check

Two exact block shards were merged for the M27 parent and Trinity d192 at
NFE4/refiner3200. The M48 portfolio was then compared on the same cases and
seed. M48 uses two NFE1/refiner1600 trajectories, so this is an absolute-budget
comparison with higher refiner and legalizer work, not an equal-neural-budget
comparison.

| Arm | Cost | V_rel | Feasible |
|---|---:|---:|---:|
| Trinity d192, NFE4/ref3200 | 1.052969 | 0.020741 | 1.0 |
| M27 parent | 1.108727 | 0.026446 | 1.0 |
| M48 portfolio | 1.080161 | 0.016695 | 1.0 |

Against Trinity, the M27 parent is `+5.30%` in cost and `+27.51%` in V_rel.
The portfolio improves the parent but remains `+2.58%` in cost while gaining
`-19.51%` in V_rel. Both portfolio paired intervals cross zero.

The cost gap is structural: M48 area is `0.05599` versus `0.00464`, b2b is
`0.01799` versus `-0.00224`, and boundary is `0.52` versus `0.41`. In contrast,
M48 improves p2b (`-0.00231` vs `0.01101`), grouping (`0.26` vs `0.49`) and rho
(`0.24881` vs `0.26461`). M48 is therefore a mixed mechanism diagnostic, not a
SOTA finalist.

## Structural checkpoint complementarity

Running M61B or M65 alone through the M48 legalizer portfolio does not pass:

| Checkpoint | Cost vs M48 | V_rel vs M48 | Area | b2b | p2b |
|---|---:|---:|---:|---:|---:|
| M61B | -1.28% | +9.80% | -1.61% | -4.81% | -77.64% |
| M65 | -1.43% | +22.71% | -10.56% | -13.58% | -71.64% |

However, selecting target-free over all already-generated candidates from
M48, M61B and M65 gives a much stronger dev32 result versus post-fix M48:

| Metric | Relative |
|---|---:|
| Cost | -5.64% |
| V_rel | -29.20% |
| Area gap | -23.40% |
| b2b | -25.95% |
| p2b | -86.62% |
| Grouping | -33.33% |
| Boundary | -29.73% |

All three checkpoints contribute to the selected set. This is a genuine
complementarity signal, but the diagnostic uses six trajectories and nine
legalizer calls. It is being confirmed on Official-100 before any promotion.

### Official-100 cross-checkpoint result

The same target-free selector was confirmed on Official-100. M48 contributes
26 selected cases, M61B 38 and M65 36.

| Metric | Cross portfolio | Trinity d192 |
|---|---:|---:|
| Cost | 1.050223 | 1.052969 |
| V_rel | 0.009877 | 0.020741 |
| Area gap | 0.038623 | 0.004640 |
| b2b HPWL gap | 0.009304 | -0.002238 |
| p2b HPWL gap | -0.020805 | 0.011011 |
| Grouping | 0.110 | 0.490 |
| Boundary | 0.340 | 0.410 |
| Rho MAE | 0.243358 | 0.264608 |

Mean cost improves by `-0.26%`, but the paired 95% CI `[-0.0270, +0.0296]`
crosses zero. V_rel improves by `-52.38%` with CI
`[-0.01627, -0.00546]`. The budget is two NFE1/refiner1600 M48 trajectories
plus two NFE2/refiner1600 trajectories from each of M61B/M65: total NFE10,
refiner steps9600 and nine legalizer calls. This is an absolute upper-budget
mechanism win, not a same-inference result.

Machine-readable results:
`outputs/reports/chip_v2_authorized_quick_summary_v1.json`.
