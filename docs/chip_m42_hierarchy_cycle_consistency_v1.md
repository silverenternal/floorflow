# M42 Hierarchy Cycle Consistency

Date: 2026-10-31

## Structural Finding

Canonical M27 predicts L0 global geometry, L1 cluster/MIB/boundary states and
L3 port offsets, but no loss requires the decoded endpoint to reproduce those
predictions. The heads are trained only against dataset targets.

M42 adds an endpoint-to-head cycle loss:

- decoded boxes recompute L0 bbox state;
- endpoint latents are pooled through the same structural masks for L1;
- port offsets are recomputed through the same p2b incidence for L3.

The term adds no parameters or inference operators and is disabled by default.

## Result

With cycle weight `0.1`, the 500-case training curve is nearly flat versus the
matched M36 control. The legalized dev32 endpoint reverses:

| Metric | Relative |
|---|---:|
| Cost | +7.52% |
| V_rel | +18.83% |
| Area gap | +56.37% |
| b2b HPWL gap | +76.64% |
| p2b HPWL gap | +52.87x |
| Grouping | +22.73% |

Cost and area paired intervals exclude zero in the harmful direction.

## Decision

Reject M42 and do not sweep cycle weight. Internal agreement can produce a
layout that is coherent with the model's own heads but harder for the
legalizer/refiner to repair.

The next model repair must supervise the endpoint in the deployment
legalization space, not merely against other model predictions.

## Artifacts

- `src/floorflow/chip_flow/hierarchy_consistency.py`
- `tests/test_chip_hierarchy_consistency.py`
- `configs/chip_m42_hierarchy_cycle_consistency_v1.json`
- `outputs/reports/chip_m42_cycle01_quick_refiner400_dev32.json`
