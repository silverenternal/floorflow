# Chip GPU Hot-Path Acceleration v1

Status date: 2026-10-31. This note records engineering changes and wall-time
measurements only. It makes no model-quality or SOTA claim.

## Scope

The maintained entrypoint is `scripts/train_chip_slot.py`. The change moves
data, geometry and validation work to `cuda:0` when exact equivalence is
available, while retaining CPU solvers where an exact CUDA replacement is not
available in the project or dependency stack.

## Implemented

- Removed the duplicate `model_config` implementation from `train_cli.py`; the
  CLI now reuses `model_config.py`.
- Added a bounded GPU-resident training cache. The 50k active arrays require
  `0.61 GiB`; the 200k range cache requires `6.25 GiB`. A 16 GiB default budget
  and free-memory guard fall back to pinned CPU prefetch rather than risking OOM.
- Reused `DeviceCaseStore.batch` for all fixed and ragged arrays. Ragged edge
  lengths are computed from CPU metadata, so batch shapes remain identical to
  `PackedCaseStore`.
- Made geometry augmentation out-of-place for mutable cache fields. A resident
  cache can no longer be corrupted by reflection augmentation.
- Prepared grouping gap matrices on device and transferred them once per batch.
  The exact SciPy sparse MST remains the CPU boundary because its dense-zero
  semantics define the historical metric; replacing them would change results.
- Vectorized projector boundary and MIB counts and the MIB joint-energy
  reduction. Intermediate ODE projections can skip diagnostic materialization;
  the final projection records diagnostics.
- Accumulated evaluation metrics and sparse diagnostics on device and performs
  one host transfer at the end of `evaluate`.
- Batched gradient/parameter finiteness checks and grouped clipping into one
  synchronization. Normal optimizer steps defer clipping-stat materialization.
- Kept the exact Hungarian assignment solver on CPU. GPU cost construction plus
  transfer to SciPy was slower than keeping the exact path on CPU.
- Reused the response sidecar through device-resident label arrays rather than
  gathering NumPy labels on every microbatch.
- Disabled the token-OT cache in the maintained training evaluator, where source
  trajectories change across checkpoints and exact reuse is not expected.

## Evidence

Source: `outputs/reports/chip_gpu_hotpath_benchmark_v1.json`, cuda:0,
batch 128, 47 blocks.

| Path | Before | After | Result |
|---|---:|---:|---:|
| CPU gather + H2D | 3.353 ms | 0.885 ms resident index | 3.79x |
| Grouping aggregation | 123.648 ms | 103.784 ms | 1.19x, max error 0 |
| Projector with diagnostics | 28.083 ms | 22.587 ms latent-only | 1.24x, output error 0 |
| Exact token OT | 42.940 ms CPU tensors | 49.356 ms GPU tensors | CPU boundary retained |

The isolated pinned-async copy was not faster than the unpinned path
(`0.976x`) at this size. It remains a fallback for overlap with the compute
stream, not a claimed standalone speedup.

`outputs/reports/chip_gpu_hotpath_benchmark_range20k_v1.json` repeats the data
and exact-equivalence checks with the 6.25 GiB range cache.

## Correctness

- Grouping CPU/GPU outputs exactly match the historical SciPy reference.
- GPU-resident batches exactly match `PackedCaseStore.batch`, including ragged
  `b2b`, `p2b` and pin padding.
- Geometry augmentation leaves resident cache arrays unchanged.
- Projector counts and joint energy match CPU.
- Grouped clipping matches CPU.
- The added equivalence suite is `tests/test_chip_gpu_acceleration.py`.

## Remaining exact boundaries

- SciPy sparse MST for historical grouping semantics.
- SciPy Hungarian assignment for exact token matching.
- A single finite-check synchronization before backward and optimizer update.
- Sequential ODE/refiner control flow and final checkpoint/report serialization.

These boundaries are explicit and measured rather than approximated. A future
kernel can replace them only with an exact contract test at realistic block
counts.
