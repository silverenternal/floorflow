# GPU Hot-Path Optimization v3

Date: 2026-11-01

## Scope

This pass is residential FloorFlow only. It does not modify or benchmark the
chips/FloorSet implementation. All changes preserve exact Hungarian assignments
and loss semantics; none are model-quality changes.

## Profile Basis

The latest profiler capture contains three training steps:

- CPU self time across captured operators: about 371 ms.
- CUDA self time across captured operators: about 1.51 s.
- The dominant cost is model computation, not CPU preprocessing.

This rules out an indiscriminate CPU-to-GPU rewrite. The useful target is
synchronization and transfer overhead around exact matching.

## Exact-Preserving Changes

- RPLAN collation materializes room, wall, and opening target counts in the CPU
  worker. Training no longer performs three separate CUDA reductions followed
  by host copies for the same batch.
- Candidate wall matching no longer copies the complete candidate-valid mask to
  CPU. It selects valid rows with a device-side index when needed and transfers
  only those indices.
- Conflict-aware backward replaces per-pair host predicates for
  `dot(primary, conflict) < 0` with a tensor `where`. It packs cosine,
  projection, and final norm diagnostics into one host transfer.

The exact room, wall, and opening match solvers remain unchanged.

## Why Not More Triton Yet

Triton remains enabled for vector quantization and junction incidence, where it
replaces launch-heavy or materialized-distance Torch paths. The profile does not
show the remaining reductions or elementwise operators as the limiting factor.
Replacing them with custom kernels would add compilation and correctness risk
without measured end-to-end benefit. A Triton candidate should be introduced
only with an exact fallback, gradient coverage when needed, and a paired profile.

## Verification

- `tests/test_unified_floorplan_generator.py`
- `tests/test_token_downstream_v2.py`
- `tests/test_gradient_health.py`
- `tests/test_training_utils.py`
- Non-chip suite: 936 passed.
- Ruff on changed files: passed.
- `uv build`: passed.
- Non-chip Python line audit: every file is below 600 lines.

No end-to-end speed ratio is claimed from this pass. A paired `cuda:0` profile
must run without concurrent chip workloads before recording a wall-clock gain.

