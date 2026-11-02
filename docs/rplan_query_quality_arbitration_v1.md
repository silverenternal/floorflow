# RPLAN Query Quality Arbitration v1

## Implementation

The arbitration path adds target-free per-query features and reuses
`RoomGraphTrustGate` as a lightweight scorer:

- class confidence, margin, and entropy;
- predicted query existence;
- mask mass, peak, and area;
- polygon area and polygon-mask Dice;
- predicted count rank;
- adjacency degree;
- routing entropy.

Training labels come only from Hungarian-matched target masks and classes.
Inference features contain no target data.

## Result

At matched 256/64:

| Arm | Room IoU | Instance F1 | Type accuracy | Count exact |
| --- | ---: | ---: | ---: | ---: |
| Class-agnostic separation | **0.4493** | **0.1842** | 0.5299 | 0.2031 |
| Query-quality arbitration | 0.4140 | 0.1704 | **0.5404** | **0.4219** |

The quality head only improves count exactness and a small type gain. It reduces
Room IoU and F1.

Error decomposition:

- mask IoU `0.4894`;
- rendered macro IoU `0.2400`;
- matched class accuracy `0.5261`.

This is worse than the frozen separation candidate in mask coverage and instance
matching, so arbitration is rejected.

## Engineering Audit

The first checkpoint exposed a real persistence bug: command-line model overrides
were applied in memory but the checkpoint config retained the old defaults. The
checkpoint now stores the effective model config. The analyzer also reconstructs
opt-in heads from state keys so older diagnostic checkpoints remain loadable.

## Decision

- Reject the query-quality head.
- Freeze class-agnostic separation as the current architecture frontier.
- Stop adding loss/arbitration modules.
- Move to larger, budget-audited training and parameter-matched controls.
