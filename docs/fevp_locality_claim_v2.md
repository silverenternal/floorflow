# Locality as a Controllable Constraint

Across 12,395 five-seed cases and 7,345 single-room negative controls, the typed strict-locality arm produced no auditable outside leakage: pixel, room-instance, label, polygon, and edge exactness were 1.0 and locality violation rate was 0.0. Removing edit gating reduced room-level exactness to 0.4863 and raised violation rate to 0.0139. The leaking arm is positive sensitivity evidence that the evaluator is not a constant-pass metric.

## Claim Boundary

- The no-gate decrease is evaluator-sensitivity evidence, not a baseline ranking.
- Strict outside preservation is measured on the declared FEVP domain.
- Single-room controls do not cover global redesign instructions.
