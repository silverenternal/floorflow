# GPU Hot-Path Optimization v4

Date: 2026-11-01

## Scope

This pass changes residential FloorFlow execution mechanics only. It does not
touch chips/FloorSet code and does not change model outputs, losses, or exact
assignment semantics.

## Audit Decision

The current model is not quality-converged. The room head remains the dominant
gate: calibration Room IoU is `0.25873` while the executable Raster2Seq
reconstruction anchor is `0.44982` on its val64 audit. Engineering work is
therefore limited to reducing iteration cost; it is not presented as a model
improvement.

The existing profiles show that CUDA model computation dominates. The useful
engineering targets are exact-matching host boundaries, repeated D2H transfers,
and loader starvation, not an indiscriminate CPU-to-GPU rewrite.

## Implemented

- Training loss and component statistics now accumulate as packed device
  vectors and synchronize once at epoch end.
- Gradient-balancer component norms are packed into one D2H transfer per update.
- Ragged teacher matching is padded to square cost matrices, copied once, and
  solved with the same exact SciPy Hungarian solver. Padding cannot alter the
  rectangular real-to-real optimum because every dummy edge has a common
  high cost.
- Teacher proposal files use a bounded LRU cache, removing repeated archive
  reads across epochs while retaining lazy loading for large future manifests.
- Raster training and validation use pinned loader memory and non-blocking H2D
  transfers. Prefetch depth is configurable and only applied when workers exist.
- Raster validation transfers targets and footprints once per batch, rather than
  once per sample and blend.
- Teacher reports now state that class-head distillation is active instead of
  incorrectly recording `class_labels_copied: false`.

Existing Triton VQ and junction-incidence kernels remain the selected custom
kernel boundary. No new Triton kernel is justified by the current profile;
small reductions are already fused by PyTorch, and forward-only replacements
would lose backward semantics or add compile/correctness risk.

## Evidence

Focused tests:

```text
tests/test_gpu_assignment.py
tests/test_proposal_distillation.py
tests/test_runtime_acceleration.py
tests/test_gradient_balancer.py
```

`cuda:0` microbenchmark on a contended GPU, batch 8, 24 queries, 128x128 masks:

| Teacher matching path | Median | Exact |
|---|---:|---|
| Per-sample D2H Hungarian | 3.8087 ms | reference |
| One packed D2H + padded Hungarian | 1.8753 ms | equal |

The ratio is `2.03x` for this isolated path. It is not an end-to-end training
speed claim.

A synthetic 64-batch, 10-scalar epoch accumulator changed D2H synchronization
counts from `640` to `1`, but host microbenchmark wall time was noise-level
(`0.995x`). The synchronization reduction is retained for real model workloads;
no wall-clock gain is claimed from that microbenchmark.

An eight-train/eight-validation, one-epoch `cuda:0` teacher-distillation contract
run completed training, validation, checkpointing, and reporting:

`outputs/train/floorflow_gpu_hotpath_v4_contract/report.json`

## Remaining CPU Boundaries

- Connected-component and canonical CAD validation still require exact
  SciPy/NumPy reference implementations.
- Variable-size Hungarian search remains a CPU solver; only transfers and
  padding are batched.
- Token generation and grammar verification remain ordered CPU boundaries.

A paired end-to-end profile on an uncontended `cuda:0` is still required before
claiming training throughput improvement.
