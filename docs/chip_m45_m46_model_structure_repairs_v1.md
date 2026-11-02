# M45-M46 Model-Structure Repairs

Date: 2026-10-31

## Scope

Both arms warm-start from the canonical M27 checkpoint, use the frozen 5k-case
training subset for 2,000 steps, and evaluate on the frozen 32-case legalized dev
manifest at NFE4/refiner400. The M36 resumed control is the reference.

## M45 Projector-Residual Head

M45 preserved the raw M27 x0 and added a zero-initialized head supervised by the
detached hard-projector displacement.

| Metric | Relative change |
|---|---:|
| Cost | +7.69% |
| V_rel | +16.72% |
| Area gap | +62.30% |
| b2b HPWL | +107.31% |
| p2b HPWL | +52.27x |
| Grouping | +36.36% |
| Boundary | 0.00% |

The cost paired interval excludes zero in the harmful direction. Legal feasibility
remains `1.0`, so the failure is endpoint quality rather than a legality crash.
Close the projector-displacement output family.

## M46 State-Conditioned Edge Bias

M46 tested a materially different dynamic-graph injection from the rejected U1
output residual. A sparse encoder maps current b2b center/aspect deltas, area
ratios and net weights to a symmetric adjacency residual. The residual changes
the existing graph-mix and attention bias inside every backbone block.

The first unnormalized implementation overflowed two 108/119-block cases. On the
same trained checkpoint, disabling the dynamic path restored finite rollouts.
The corrected implementation uses the mean active-edge score per row. All 500
dev-curve rollouts are finite, and the training-curve total improves slightly
from `0.017198` to `0.017185`.

| Metric | Relative change |
|---|---:|
| Cost | +7.76% |
| V_rel | +14.45% |
| Area gap | +57.45% |
| b2b HPWL | +109.65% |
| p2b HPWL | +55.42x |
| Grouping | +9.09% |
| Boundary | +14.29% |

The proxy improvement does not survive legalization. This closes all-layer
state-conditioned graph modulation; application depth, bottleneck width and
residual scale are not swept after a failed joint primary gate.

## Engineering Repair

For the hierarchical model, `decode_for_batch` is the identity. The generic
`clean` and `clean_physics` terms were therefore a zero gradient plus a constant.
The training objective now skips those terms for the identity hierarchy decoder.
This does not change the optimized gradient but removes redundant target-side
physics computation and misleading nonzero loss logging.

## Artifacts

- `src/floorflow/chip_flow/projector_residual_head.py`
- `src/floorflow/chip_flow/state_conditioned_edge_bias.py`
- `tests/test_chip_state_conditioned_edge_bias.py`
- `configs/chip_m45_projector_residual_head_v1.json`
- `configs/chip_m46_state_conditioned_edge_bias_v2.json`
- `outputs/reports/chip_m45_projector_residual_quick_refiner400_dev32.json`
- `outputs/reports/chip_m46b_state_edge_normalized_quick_refiner400_dev32.json`
