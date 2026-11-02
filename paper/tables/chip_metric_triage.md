# Chip Metric Triage

| Metric | Decision | Evidence | Allowed statement |
|---|---|---|---|
| MIB V_rel | No claim | Baseline and candidate both zero; interval `[0,0]` | No MIB improvement claimed |
| Boundary V_rel | Trend only | Mean `-8.36%`; CI crosses zero | Possible boundary signal, not stable |
| p2b/port HPWL | Structural signal | Mean `-42.24%`; CI crosses zero | Structural proxy only |
| High-b2b / high-pin strata | Limitation | Little stable benefit | Gains are heterogeneous |
| Cost tail | Mechanism | p50 `2.59%`; p90/p95 about `14.4%` | Mainly reduces structural-failure tails |
| V_rel tail | Mechanism | p95 `4.76%`, non-monotonic | Moderate/high quantiles improve more than extreme tail |
| Equal-quality budget | Diagnostic | NFE8→1; seed19 refiner 400→100 | Fewer steps at declared quality; primary endpoint unchanged |
| Trinity d192 | External gap | Cost `+5.76%`, V_rel `+30.45%` | Candidate is not SOTA |
| FloorSet/CAD SOTA | Unsupported | Same-protocol comparator remains better | Do not claim |

The allowed language is deliberately narrower than the measured means whenever
the matched interval or official panel does not support a stable claim.

