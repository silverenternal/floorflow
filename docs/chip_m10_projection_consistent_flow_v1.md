# M10 Projection-Consistent Per-Block Flow

Date: 2026-10-30

Status: rejected at confirmation. The quick mean gain reverses at refiner1600,
so the M10 family is closed without full data or multiseed.

## Motivation

The DAC'26 FlowPlace paper describes per-macro flow matching followed by a
hard-constraint sampling step that:

1. extrapolates a final layout from the current velocity;
2. projects that layout onto hard constraints;
3. updates the velocity to the projected endpoint; and
4. reinterpolates the next trajectory state.

Primary source: <https://arxiv.org/abs/2604.23658>.

The official implementation is
<https://github.com/lamda-bbo/flowplace>. No repository license was present on
2026-10-30. This project therefore uses the paper-level idea only and implements
it clean-room through existing first-party operators.

## Current Versus M10 Trajectory

The existing typed hierarchy already has:

- per-block flow trajectories;
- exact-area latent decoding;
- fixed/preplaced/MIB/boundary constraints;
- a differentiable hard projector applied after each Euler update.

The missing M10 operation is endpoint consistency. The default path advances
from the current state and projects the advanced state. M10 instead projects the
model's predicted endpoint and reinterpolates the block state from the source:

```text
projected_endpoint = Projector(predicted_endpoint)
next_state = (1 - t_next) * source_state + t_next * projected_endpoint
```

Global, structural and port states retain the ordinary hierarchy update. M10
does not add parameters or alter the trained checkpoint.

## Frozen Arms

The screen compares three modes from the same checkpoint and source noise:

| Mode | Intermediate hard projection | Projected endpoint reinterpolation |
|---|---|---|
| `post_step` | yes | no |
| `projection_consistent` | yes | yes |
| `final_only` | no | terminal projection only |

All arms use the same NFE, refiner snapshots, projector iterations, legalizer,
case manifest and inference seed. The quick stage uses NFE 1/4, refiner 0/400
and all three modes. Confirmation extends to NFE 1/2/4/8 and refiner 0/400/1600.

## Evidence Status

CPU/unit evidence:

- default path is bit-identical to the previous `post_step` path;
- source-to-projected-endpoint interpolation has an exact regression test;
- `final_only` equals one terminal projector call after an unprojected rollout;
- the complete runner passes a real-cache smoke;
- one NFE4/refiner0 case showed cost `+0.50%` and V_rel unchanged for
  `projection_consistent`, which is retained as a smoke-level negative
  observation, not a mechanism result.
- On the full 32-case quick screen, `final_only` improved NFE4/refiner400 cost
  by `1.51%` and V_rel by `2.98%`, while `projection_consistent` failed the
  primary point.
- Confirmation at NFE4/refiner1600 reversed the result: `final_only` regressed
  cost by `3.33%` and V_rel by `0.59%`, with median cost also worse. Cost
  p90/p95 improved, but the mean regression and primary equal-weight gate
  dominate; tail-only movement is insufficient for promotion.
- M10 is therefore closed as a confirmation-negative. The queued
  new-checkpoint replication was cancelled, and only the M6 response repair
  remains serialized behind full-50k.

## Promotion Gate

At one matched NFE/refiner point, the candidate mode must improve both mean
legalized cost and V_rel while preserving feasibility 1.0, exact area, fixed
shapes, preplaced positions, MIB sharing and boundary codes. Paired intervals
and block-count strata must be retained. A failed quick stage blocks
confirmation; a failed confirmation blocks full-data or architecture
registration.

## Reuse

- `src/floorflow/chip_flow/hierarchical_flow_runtime.py`
- `src/floorflow/chip_flow/hard_constraint_projector.py`
- `src/floorflow/chip_flow/flow_adapter.py`
- `scripts/evaluate_chip_refiner.py`
- `scripts/run_chip_m10_projection_flow_screen.py`

The screen does not create a parallel model, sampler, projector or metric stack.

