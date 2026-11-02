# Matched-Compute Component Gap v1

This report decomposes the released normal-weight matched-compute anchor
comparison across candidate training seeds 19, 20, and 21. It is descriptive
and does not replace the exact group=2 anchor control.

| Metric | Mean relative gap | Seed direction |
|---|---:|---|
| Cost | -1.56% | improves 2/3, regresses 1/3 |
| V_rel | -10.53% | improves 3/3 |
| Grouping | -17.62% | improves 3/3 |
| Boundary | -5.23% | mixed, 1 improve / 2 regress |
| B2B HPWL gap | +32.11% | regresses 2/3 |
| P2B HPWL gap | -44.25% | improves 2/3 |
| Area gap | +10.18% | regresses 2/3 |
| Rho/aspect MAE | +11.15% | regresses 3/3 |
| MIB | no discrimination | both means are zero |

## Consequences

- V_rel and grouping are the only structural metrics with consistent positive
  direction across all three candidate seeds in this comparison.
- P2B is promising but seed-mixed.
- B2B and area do not support a general improvement claim in this anchor
  comparison.
- Aspect MAE consistently regresses and must remain an explicit limitation.
- MIB cannot be used as evidence because the metric has no discriminative power.
- The exact group=2 Trinity anchor still rejects the candidate on both aggregate
  metrics; this decomposition does not change that gate.

The machine-readable report is
`outputs/reports/chip_v45_matched_component_gap_v1.json`.
