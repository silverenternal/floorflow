# Sampling and best-of-N policy

The exterior contour `c` is a condition, not a unique answer. The target is a
conditional distribution `p(x|c)` over interiors `x`. The learned model should
approximate that distribution with `q_theta(x|c)`.

## Two different quantities

For a structural score `g(x,c)`, the single-sample quality is
`E_{x~q_theta(.|c)}[g(x,c)]`. A seed-controlled deterministic sample estimates
the behaviour used by a user who requests one design.

Best-of-N estimates a different functional:

```text
B_N(c) = E[max_{1 <= i <= N} g(X_i,c)]
```

For fixed independent candidates, `B_N(c)` is non-decreasing in `N`, is
generally larger than the single-sample expectation, and cannot be compared
across methods that use different budgets. It measures the upper envelope of a
proposal distribution, not the quality of that distribution.

Selecting a candidate with a learned score `s(x,c)` defines a different target:

```text
q_tilted(x|c) = q_theta(x|c) exp(beta s(x,c)) / Z(c,beta)
```

That is a legitimate reward-tilted sampler only when the scorer, budget,
calibration, and inference cost are declared and evaluated. Using the test
target or target-derived metrics to select a sample is oracle cherry-picking
and cannot support a generative-model claim.

## Project policy

- Production and primary reconstruction metrics use one seed-controlled sample.
- Distribution claims use every generated sample. Reported quantities are the
  mean, median, standard error, MMD/energy, and bounded coverage/density.
- Features are standardized with train statistics only. MMD uses a pooled
  median-distance bandwidth so the kernel does not collapse to a constant.
- Best-of-N is restricted to a predeclared diagnostic budget, appears in an
  appendix/ceiling report, and is forbidden from the publication gate.
- Structural repair is a guarded one-sample mapping, not candidate selection.
  It must be reported separately with before/after paired tests and rejection
  controls.
- Every report carries a `sampling_policy` manifest. The gate rejects reports
  with `best_of_n_used=true`, an implicit selector, or target-informed
  selection.

## Recommended final reporting

| Claim | Estimator | Role |
|---|---|---|
| Single-design reconstruction | one fixed-seed sample | primary |
| Conditional generative quality | expected metric over all samples | primary |
| Mode coverage / precision | bounded coverage and density | primary |
| Diversity | pairwise distance and count/area/adjacency diversity | primary |
| Proposal upper envelope | fixed-budget best-of-N curve | diagnostic only |
| Optional reward selection | tilt-normalized scorer with cost accounting | separate claim |
