# HouseMind-style GED Deterministic Evaluator v2

Date: 2026-11-01

## Problem

The previous evaluator passed a `0.5s` timeout to NetworkX. Once the timeout
fired, the returned graph-edit distance depended on process scheduling and the
time consumed by unrelated batches. Two reports with identical graph counts and
intersections could therefore disagree. A wall-clock result is not a fixed
evaluator and cannot support paired significance or SOTA claims.

## Frozen Definition

The canonical graph is the unique room-type adjacency graph. Node substitution
costs `0` for the same type and `1` otherwise. Node deletion, node insertion,
edge deletion, and edge insertion each cost `1`. Edge substitution costs `0`,
matching the NetworkX defaults used by the original readout.

For a node product at most `36`, NetworkX computes an exact GED with no wall-clock
timeout. Node and edge iteration are sorted first. For larger graphs, the
evaluator evaluates exact-label, label-Hungarian, and label-degree-Hungarian
seeds, then performs deterministic one-swap/add/remove local search. It reports
`exact=false` and `fallback_reason=deterministic_bounded_matching`; bounded rows
must never be described as exact GED.

The machine-readable contract is
`configs/housemind_style_ged_deterministic_v2.json`.

## Failure Prevented

An earlier multiscale-opening comparison found two cases with identical node and
edge counts/intersections but GED changes of `8->12` and `10->11`. Both rows had
legacy `timeout_returned_best_bound`. The deterministic evaluator removes that
failure mode.

## Use Rules

- All arms in one paired table must use this evaluator or an explicitly named
  successor.
- Keep `ged_exact` and `ged_fallback_reason` in case-level evidence.
- Report the bounded-search share alongside mean GED.
- Do not compare absolute GED values produced by the legacy timed evaluator.
