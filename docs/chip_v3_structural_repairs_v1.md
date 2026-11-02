# Chip v3 Structural Repair Implementation

Date: 2026-11-01

This CPU-side implementation pass addresses the six failed authorized quick
screens without promoting any failed checkpoint.

## Implemented repairs

- **M57 bounded coarse anchor:** stability masks identify low-change global and
  structural slots. Stable slots receive a bounded anchored velocity instead of
  exact freezing, while unstable slots remain trainable through the trajectory.
- **M58 shared-trunk factorization:** the factorized block head can route center
  and rho evidence through one shared trunk. The center/rho output heads retain
  zero-initialized final projections and rho supports a trust radius.
- **M59 legalizer-constrained endpoint:** endpoint compatibility now exposes and
  optimizes projected area, p2b and per-block scale-pack displacement targets
  from the response sidecar.
- **M60 incidence-normalized graph:** relation contributions are scaled by the
  inverse square root of active incidence, preventing port-heavy cases from
  increasing relation residual magnitude by edge count alone.
- **M61 route/source decomposition:** route-only and source-only matched arms are
  frozen in the execution plan.
- **M62 constrained fusion:** endpoint and route residuals have a separate
  fusion recipe that remains blocked until M59 and M61 independently pass.
- **M63 response sidecar:** deterministic complexity-stratified selection builds
  a 1,000-case CPU scale-pack response sidecar.

## M63 sidecar

The sidecar contains 1,000 unique cases selected across complexity strata with
seed `20261101`:

| Rung | Cases |
|---|---:|
| 2 | 625 |
| 3 | 309 |
| 4 | 64 |
| 5 | 2 |

Displacement L2 mean is `0.6358`; p95 is `1.3607`.

Artifacts:

- `outputs/cache/chip_m63_response_screen1k_v1.npz`
- `configs/chip_m63_response_indices1k_v1.json`
- `outputs/reports/chip_m63_response_sidecar_audit_v1.json`

## Protocol

The new stages remain frozen to the 5k exploratory subset except M59/M62, which
use the 1k response subset. All comparisons use the shared matched control,
NFE4, refiner400, seed 19 inference and feasibility `1.0`.

The next GPU run starts with M57 and M58 independent quick screens. M63 is
already complete on CPU. M59 and M62 remain blocked until their prerequisite
probes pass.
