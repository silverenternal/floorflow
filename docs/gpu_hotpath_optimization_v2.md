# GPU Hot-Path Optimization v2

Date: 2026-11-01

## Scope

This pass changes residential FloorFlow execution mechanics only. It does not
claim a model-quality improvement, and it does not touch chips/FloorSet code.
All GPU measurements use `cuda:0`.

## Exact-Preserving Changes

- Room, wall, and opening exact assignments now share one reusable batch
  helper. Equal-shaped cost matrices use one stacked host transfer; ragged
  matrices retain the measured-safe sequential exact fallback.
- Variable-length match indices are packed once and moved to the device once,
  replacing repeated per-sample `numpy -> device` index transfers.
- Wall, opening, and room-quality losses consume packed indices with batched
  tensor operations while preserving the prior sample-mean semantics.
- Room target counts are materialized once per room target, then shared by
  polygon, identity, and auxiliary losses instead of synchronizing separately.
- Invalid polygon-point collapse is fully vectorized. It preserves valid
  vertex order, repeats the original final valid vertex, and retains the
  fallback for queries with fewer than three valid points.
- Scheduled token prefix corruption is vectorized across the batch. It no
  longer branches on a CUDA scalar or extracts a random start position per row.
- Device metrics are packed before their single epoch-end host transfer.
- CUDA capability lookup used by bf16 autocast is cached.

## Exact Equivalence

`tests/test_gpu_assignment.py`, `tests/test_unified_floorplan_generator.py`,
and the focused loss suites compare the optimized paths with the previous
semantics. The benchmark script also asserts:

- room assignments match SciPy one-by-one;
- wall and opening losses match the legacy scalar semantics within `2e-7`;
- polygon-point collapse matches the previous nested-loop output exactly.

Evidence:

- `outputs/reports/floorflow_gpu_hotpaths_v1.json`
- `scripts/benchmark_floorflow_gpu_hotpaths_v1.py`
- `tests/test_gpu_assignment.py`
- `tests/test_runtime_acceleration.py`

## Isolated `cuda:0` Timings

The table reports median wall time over 100 measured iterations after 10 warmup
iterations. Ratios are microbenchmarks for the changed paths, not end-to-end
epoch speedups.

| Path | Legacy | Optimized | Speedup |
|---|---:|---:|---:|
| Exact room assignment | 0.1758 ms | 0.1493 ms | 1.18x |
| Wall endpoint/type loss | 1.2633 ms | 1.1522 ms | 1.10x |
| Opening center/width/host loss | 1.2763 ms | 1.0409 ms | 1.23x |
| Polygon point collapse | 13.6848 ms | 0.3090 ms | 44.28x |
| Token prefix corruption | 2.3603 ms | 0.4424 ms | 5.34x |

## End-to-End Contract Check

A one-epoch, 32-train/8-validation `cuda:0` run completes training, validation,
gradient checks, checkpointing, and report generation:

- `outputs/train/floorflow_runtime_profile_32_v4/report.json`
- `outputs/profile/floorflow_runtime_profile_32_v4/summary.json`

This run is not a paired end-to-end speed claim because the earlier profile did
not include the current token consumer workload. The new profile is retained
for future matched-budget comparisons and hot-path diagnosis.

## Remaining CPU Boundaries

- Sentence-level token generation and contract verification remain sequential.
- Connected-component room metrics and canonical CAD validation still use
  SciPy/NumPy reference implementations.
- Variable-shape Hungarian assignment remains an exact CPU solver boundary.

Triton remains enabled for the existing VQ and junction-incidence kernels.
Small reductions that are already fused well by PyTorch are not replaced with
forward-only Triton kernels.
