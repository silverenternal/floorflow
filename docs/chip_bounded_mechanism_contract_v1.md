# Chip Bounded-Mechanism Contract v1

Status date: 2026-10-28. This document defines the scope of the chip claim. It
does not claim a global optimum, lower routed PPA, or a theorem about FloorSet
cost.

## Objects

- `A = (a_i)` is the positive target-area vector.
- `z_i = (cx_i / s, cy_i / s, rho_i)` is the latent block state, with
  `s = sqrt(sum_i a_i)`.
- `D_A(z)` is the exact-area decoder.
- `L` is the audited legalization/projection operator, ending in `scale_pack`.
- `C` is the feasibility set accepted by the official checker.
- `V_raw(x)` is a vector of pre-legalization hard-constraint residuals.
- `Delta(x)` is normalized legalization displacement.

## Proposition 1: exact-area decoding

For `a_i > 0`, the decoder computes

```text
w_i = sqrt(a_i * exp(rho_i))
h_i = a_i / w_i
```

Therefore `w_i h_i = a_i`, up to finite floating-point tolerance, for every
finite `rho_i`. Clamping `rho_i` changes the aspect ratio but not the decoded
area. This is an invariant of the current decoder, not a learned property.

## Proposition 2: certified feasibility

If `q = L(x)` is accepted by the official checker, then `q` has no overlap,
fixed/preplaced violation, area/shape violation, MIB violation when applicable
or boundary violation under the checker's tolerance.

This proposition is conditional on `q` being checker-accepted. The legalizer is
allowed to fail or fall back; an unaccepted output is reported and removed from
the scored population. The proposition does not state that every arbitrary
input becomes feasible.

## Mechanism hypothesis

The empirical hypothesis is:

```text
For matched decoder, refiner, legalizer and budget, typed hierarchical flow
reduces the upper quantiles of pre-legalization residual V_raw and normalized
displacement Delta relative to the graph-token control.
```

This hypothesis is not proved by Proposition 1 or 2. It must be tested with
paired case/seed statistics.

## Empirical envelope

For an arm `m`, constraint `k`, and statistic `tau`, report:

```text
E_{m,k,tau} = quantile_tau(V_raw[k] or Delta)
```

Every envelope includes its case/seed denominator. Primary envelope points are
p50, p90, p95 and max. A claim is rejected if the candidate has a worse upper
envelope with a crossed interval or if the denominator changes between arms.

The current stored evidence includes raw overlap-pair envelopes and exact
post-legalization checks. Displacement, boundary-conflict and iteration
envelopes are completed by `MECH-TRACE`.

## What is not claimed

- No theorem that the learned hierarchy reaches a global FloorSet optimum.
- No theorem that learned proposals reduce routed wirelength or improve PPA.
- No theorem that every seed or every case improves.
- No claim that exact-area decoding implies structural or routing quality.
- No substitution of b2b/p2b HPWL for routed wirelength.

## Falsification

The empirical bounded-mechanism claim fails if:

- any scored output violates the checker feasibility invariant;
- candidate and control use different legalizer or budget denominators;
- upper empirical residual/displacement envelopes do not improve;
- the structural gain does not transmit to a routed PPA, congestion or timing
  endpoint on the frozen external subset.

