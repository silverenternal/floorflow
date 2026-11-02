# Chip CVPR Experiment Status

Status date: 2026-10-27. This file tracks experimental evidence and claim
boundaries, not manuscript text.

> Post-audit lineage correction (2026-10-31): the canonical executable lineage is
> M27 typed hierarchy initialized from Trinity d192. The graph-token results in
> this historical note are retained as provenance and must not be combined with
> M27 parameter, budget or seed tables. The canonical checkpoint has 6,797,217
> parameters and 640k training exposures, not the older 5.84M / 2.56M facts.
> See `docs/chip_architecture_completion_v2.md` and
> `outputs/reports/chip_architecture_lineage_audit_v1.json`.

Chip-specific terminology and formulas are defined in
`docs/chip_floorplanning_concepts.md`.

## Current Decision

The retained candidate is now the same-parameter graph-token velocity flow with
exact-area decoding. It directly wraps the pinned Trinity denoiser and uses
per-block flow state, spectral graph positional encoding, qk-normalized SDPA,
SwiGLU/RMSNorm blocks, graph message mixing, and token time conditioning. It
beats the implemented graph-conditioned invariant-global-slot decoder on both
official endpoints across three seeds after the common refiner.

Do not claim FloorSet SOTA. The same-protocol Trinity d192 endpoint remains
better than the new candidate by 5.76% hard cost and 30.45% V_rel.

## Core Evidence

| Experiment | Hard cost | V_rel | Status |
|---|---:|---:|---|
| 20k matched global control, raw NFE8 | 2.31784 | 0.13208 | control |
| 20k graph slot, raw NFE8 | 1.61711 | 0.10062 | -30.23% cost |
| 20k matched global, NFE4 + 400 refiner | 1.86012 | 0.09128 | control |
| 20k graph slot, NFE4 + 400 refiner | 1.30942 | 0.05425 | -29.61% cost |
| 20k graph slot, EMA-refined diagnostic | 1.25510 | 0.05040 | not post-hoc selected |
| 20k graph token, NFE4 + 400, seed mean | 1.26280 | 0.05119 | -5.33% cost / -13.60% V_rel vs graph |
| Trinity d192, same protocol | 1.19400 | 0.03924 | external comparator |

All graph-versus-control paired intervals exclude zero. The graph result is
currently 9.67% above Trinity on refined hard cost and must be reported as such.

## Frozen-Runtime Three-Seed Replication

The final candidate and parameter-matched control were retrained from scratch
under one frozen runtime with saved batch RNG state.

| Seed | Graph refined cost | Control refined cost | Graph V_rel | Control V_rel |
|---:|---:|---:|---:|---:|
| 19 | 1.35263 | 1.89656 | 0.05854 | 0.09056 |
| 20 | 1.34296 | 1.93707 | 0.06221 | 0.09661 |
| 21 | 1.30698 | 1.86184 | 0.05729 | 0.09247 |
| Mean | 1.33419 | 1.89849 | 0.05935 | 0.09322 |

Every seed passes both refined primary metrics with paired intervals excluding
zero. Seed-mean relative deltas are `-33.05%` raw cost, `-30.05%` raw V_rel,
`-29.72%` refined cost, and `-36.34%` refined V_rel. The final refined mean
remains 11.74% above Trinity and 51.23% above Trinity on V_rel, so no SOTA claim
is permitted.

Sources:

- `outputs/reports/chip_graph_slot_20k_v1.json`
- `outputs/reports/trinity_d192_matched_protocol_v2.json`

## Mechanism Attribution

At the same 5k budget, runtime, data order, objective, and evaluator:

| Arm | Parameters | Dev total | Refined cost | Refined V_rel |
|---|---:|---:|---:|---:|
| Full graph | 5,871,566 | 0.023940 | 1.32011 | 0.06289 |
| Features only | 5,499,075 | 0.029780 | 1.81613 | 0.12062 |
| Topology only | 5,866,382 | 0.029125 | 1.61517 | 0.10803 |

Topology-only is within 0.09% of full parameter count, and every paired interval
prefers full. This supports a joint mechanism claim rather than a feature-only
or graph-bias-only claim.

Source: `outputs/reports/chip_graph_mechanism_ablation_v1.json`.

## Graph-Token Flow Promotion

The 5k token-flow screen failed: refined cost/V_rel changed by
`+5.85%/+8.42%` relative to the 5k graph decoder. A separately labeled 20k
optimization-budget rescue then passed. The final three-seed official results
are:

| Seed | Token refined cost | Graph refined cost | Token V_rel | Graph V_rel |
|---:|---:|---:|---:|---:|
| 19 | 1.27569 | 1.35263 | 0.05174 | 0.05854 |
| 20 | 1.25136 | 1.34296 | 0.05019 | 0.06221 |
| 21 | 1.26134 | 1.30698 | 0.05164 | 0.05729 |
| Mean | 1.26280 | 1.33419 | 0.05119 | 0.05935 |

Every seed wins both refined metrics. The seed-mean relative improvements are
`-5.33%` cost and `-13.60%` V_rel after the common refiner; raw improvements are
`-3.89%` and `-10.31%`. Paired case intervals cross zero on seeds 19 and 21 for
the refined metrics, so the supported claim is a seed-mean/architecture result,
not a per-case guarantee.

Sources:

- `outputs/reports/chip_trinity_token_three_seed_v1.json`
- `outputs/reports/chip_trinity_claim_gate_v1.json`
- `outputs/reports/chip_trinity_parity_gap_v1.json`
- `outputs/reports/chip_trinity_token_bridge_cpu_v1.json`

## Mechanism, Tail, and Budget Diagnostics

The official `V_rel` is exactly
`(cluster split + MIB shape mismatch + boundary violation) / N_soft`.
FloorSet has no floorplan-opening channel, so boundary and p2b/port metrics are
reported as structural proxies and are not relabeled as openings.

Matched pooled refined diagnostics show:

| Component | Baseline | Token flow | Relative improvement |
|---|---:|---:|---:|
| b2b connectivity HPWL gap | 0.07968 | 0.05784 | -27.41% |
| p2b/port HPWL gap | 0.48826 | 0.28202 | -42.24% |
| Area/outline gap | 0.15717 | 0.12509 | -20.41% |
| Grouping `V_rel` | 0.02644 | 0.02219 | -16.08% |
| Boundary `V_rel` | 0.03147 | 0.02884 | -8.36% |
| Rho/aspect MAE | 0.31684 | 0.30484 | -3.79% |
| MIB `V_rel` | 0 | 0 | no official signal |

The b2b HPWL, area/outline, grouping, and rho improvements have pooled paired
intervals below zero. p2b and boundary have large mean gains but intervals that
cross zero.

Cost-tail improvement increases from `2.59%` at p50 to `14.46%` at p90 and
`14.39%` at p95, supporting a structural-failure reduction rather than a
uniform shift of all cases. Complexity is heterogeneous rather than monotonic:
high block-count and boundary-coded strata gain, while high-b2b and high-pin
strata show little benefit. This limitation must be reported.

The equal-quality budget diagnostic finds:

- Token-flow NFE1 already beats baseline NFE8 raw on both pooled mean
  thresholds, saving seven NFE steps.
- On seed 19, token flow at 100 refiner steps reaches and exceeds baseline
  400-step quality, saving 300 refiner steps.
- The NFE4 plus 400-step setting remains the preregistered primary endpoint;
  the budget result is a separate inference diagnostic.

Sources:

- `outputs/reports/chip_token_flow_component_rows_v2.json`
- `outputs/reports/chip_token_flow_mechanism_diagnostics_v1.json`
- `outputs/reports/chip_token_flow_refiner_curve_v1.json`
- `outputs/reports/chip_token_flow_budget_curve_v1.json`
- `outputs/reports/chip_token_flow_claim_gate_v3.json`

## Trinity Gap and Budget Pareto

The predeclared legacy Trinity anchor remains `1.19400/0.03924`, giving candidate
gaps of `+5.76%/+30.45%`. It was produced without a serialized sampling seed.
Three fixed-seed replications produce:

| Seed | Cost | V_rel |
|---:|---:|---:|
| 19 | 1.23861 | 0.04387 |
| 20 | 1.21378 | 0.04573 |
| 21 | 1.19730 | 0.04118 |
| Mean | 1.21656 | 0.04360 |

The legacy anchor lies below the observed three-seed ranges, so it is retained
only as an appendix/footnote boundary and is explicitly marked non-reproduced.
The primary external table uses the reproducible seed mean:

| Comparison | Cost | V_rel |
|---|---:|---:|
| Trinity d192, unseeded legacy | 1.19400 | 0.03924 |
| Trinity d192, 3-seed reproducible | 1.21656 | 0.04360 |
| Candidate | 1.26280 | 0.05119 |
| Gap vs legacy | +5.76% | +30.45% |
| **Gap vs reproducible** | **+3.80%** | **+17.42%** |

Component decomposition identifies the remaining structural gap: b2b HPWL
`+41.5%`, grouping `+33.2%`, area/outline `+24.8%`, and rho/aspect `+10.9%`.
The candidate is already better on p2b/port HPWL by `22.3%`. Because Trinity
d192's decoder, training protocol, and sampling settings are not fully public,
the component gaps describe structural differences only and are not attributed
to any single factor among exact-area decoder, training data, or sampling.

The budget-quality Pareto figure is available as SVG/PDF with NFE and refiner
panels. It directly marks candidate NFE1 against baseline NFE8 and seed-19
candidate 100 against baseline 400 refiner steps.

The figure also keeps the 50-step point visible and marks that it has lower
cost but insufficient V_rel, making the extreme-budget quality trade-off
explicit rather than hiding a failed shortcut.

Sources:

- `outputs/reports/chip_trinity_d192_component_gap_v1.json`
- `outputs/reports/chip_token_flow_pareto_v1.json`
- `outputs/figures/chip_token_flow_pareto_v1.svg`
- `outputs/figures/chip_token_flow_pareto_v1.pdf`

Trinity d192 is positioned as the legacy anchor under the current protocol; the
reported external gap uses the reproducible three-seed anchor and makes no
cross-protocol SOTA claim. Graph2Plan and FreeScene remain related-work context,
not numeric comparisons.

## Retained Negative Results

- Absolute slot transport failed its matched screen.
- Residual slot transport failed its CPU probe: residual norm was 7.59x the
  absolute slot norm and residual prediction was worse than nearest absolute.
- x0, conditioning dropout, and their combination did not improve both refined
  primary metrics by 5%.
- A proper 200k warm restart improved raw aggregate and 61-120 metrics, but
  regressed refined cost by 3.65% and refined V_rel by 15.35%.
- The fixed hierarchy remains a negative control.
- Learned condition slots failed the L1 gate: residual eta-squared was `0.0134`,
  residual error improved only `0.39%`, and one slot was dead.
- The graph-token flow at 5k did not meet the 5% refined gate; only the 20k
  optimization-budget rescue is promoted.

## Bottleneck Review and Rescue Attempts

The single sampled global latent is not carrying the layout information implied
by the architecture name. Across 640 cases, its cross-case standard deviation is
`3.28e-5` with effective rank `6.33 / 192`, while the condition latent has
standard deviation `0.07875`.

A causal intervention on the final seed-19 checkpoint confirms that the decoder
mostly ignores the sampled global state: shuffling it changes dev total geometry
loss by only `0.85%`, zeroing it by `2.91%`, and even the oracle target global
changes the result by only `0.20%`. The evidence therefore supports
graph-conditioned decoder quality, not rich information flow through the single
global vector.

Three rescue directions were implemented and evaluated without changing the
established model:

- Raw-statistics global skip: recovered target std by `48.46x`, but refined cost
  regressed `3.56%` and V_rel improved only `1.23%`.
- Global FiLM bottleneck: shuffle sensitivity increased only to `1.51%`;
  refined cost regressed `2.13%` and V_rel improved `2.37%`.
- Two-step path-consistency loss: parameters/data/order were identical, but
  refined cost changed by `+0.28%` and V_rel by `-1.00%`.

Two additional diagnostics close obvious alternatives:

- Applying the shared refiner to 80 training targets degraded legalized cost
  from `1.01722` to `1.02611`; refiner-output target distillation is invalid.
- An equal ensemble of velocity and x0+dropout changed refined cost by `+1.36%`
  and V_rel by `-4.48%`; it does not improve both metrics.

The next credible strengthening direction is not more global-path capacity.
The local-slot follow-up was tested and failed its predictability gate. The
working rescue instead removes the single global flow state in favor of
per-block graph-token transport.

Sources:

- `outputs/reports/chip_slot_predictability_probe_v1.json`
- `outputs/reports/chip_output_parameterization_and_dropout_v1.json`
- `outputs/reports/chip_range_warm_restart_v1.json`
- `outputs/reports/two_domain_dynamic_hierarchy_v1.json`

## Runtime

The measured bottleneck was Python projection synchronization. Fixed-batch
values match and gradients differ by `1.5e-11`; projection is `37.08x` faster,
fused AdamW `22.46x`, and foreach EMA `2.93x`. The training entrypoint is below
250 lines and all changed Python files remain below 600 lines.

Source: `outputs/reports/chip_training_framework_benchmark_v1.json`.

## Completion

The final claim gate and evidence package are complete:

- `outputs/reports/chip_token_flow_claim_gate_v3.json`
- `outputs/cvpr_package_v13_chips/submission_checklist.json`

The engineering suite passes (`545 passed`) and every first-party Python file is
strictly below 600 lines. Repository-wide Ruff also scans vendored third-party
and legacy sources; the changed first-party scope passes Ruff.
