# GPU Hot-Path Optimization v5

Date: 2026-11-01

## Scope

This pass covers the residential raster-instance trainer only. It does not
touch chips or FloorSet code and does not change model architecture, losses, or
the promoted room readout. Engineering changes are separated from model-quality
claims.

## Findings

The projector profile shows that CUDA model computation dominates, so replacing
large model regions with custom kernels is not the first optimization. The
remaining actionable gaps were:

- the raster-instance entry point did not expose the shared autocast, scaler,
  channels-last, compile, TF32, or Triton controls;
- validation recomputed dense and query softmax for every blend;
- validation copied labels, targets, and supports in separate blocking passes;
- teacher batches always transferred 64 padded proposal masks even when a batch
  contained far fewer rooms;
- gradient balancing concatenated every parameter gradient before taking a
  norm;
- mixed-precision logits reached fp32 class weights outside autocast.

## Implemented

- Raster training now supports `fp32`, `fp16`, and `bf16`, scaled backward,
  channels-last, `torch.compile`, TF32 control, and the existing Triton flag.
- Floating model outputs are materialized to fp32 at the loss boundary while
  retaining mixed-precision forward computation.
- Dense/query softmax endpoints are computed once per validation batch and
  reused for all five readout blends.
- Labels for every blend, the target, and the footprint are packed into one D2H
  transfer per validation batch.
- Teacher proposal tensors are compacted to the actual maximum room count in
  the batch. The first eight train64 cases have counts
  `[9,7,9,7,5,10,4,7]`, so the 256x256 mask transfer falls from
  `33.55 MB` to `5.24 MB`, a `6.4x` reduction for that batch.
- Gradient balancing now uses foreach norms without materializing a concatenated
  gradient buffer.
- Room decoder, CAD topology repair, and opening-host routing now normalize
  fully padded attention rows with device-side tensor operations instead of a
  host-visible `bool(empty.any())` check.
- Room-identity, room-partition, and room-group losses use masked device
  reductions instead of branching on `valid.any()` or `present.any()`.
- Room-instance targets are now extracted from the augmented CPU target inside
  DataLoader workers and delivered through the pinned collate path. The old
  order copied target labels H2D, back to CPU for connected components, then
  copied all padded masks H2D again.
- Worker-produced instance masks are serialized as binary `uint8` and converted
  to float32 after H2D, reducing pinned-host storage and mask transfer volume by
  `4x`.
- Per-sample valid room counts are carried as CPU metadata and passed directly
  to the room-mask loss, removing its previous CUDA count reduction and D2H.

## Verification

Focused tests pass, including teacher compaction, loss-boundary casting,
readout equivalence, exact assignment, gradient balancing, and runtime flags.
The full non-chip suite passes `964` tests; Ruff, the package build, JSON
validation, and the `<600`-line audit also pass.

Two `cuda:0` contracts completed training, validation, checkpointing, and report
generation on the real raster path:

- `outputs/train/room_runtime_smoke_v5/report.json`: bf16, channels-last,
  Triton enabled, compile disabled.
- `outputs/train/room_runtime_compile_smoke_v5/report.json`: bf16,
  channels-last, `torch.compile(mode="reduce-overhead")`.
- `outputs/train/room_runtime_dynamic_balance_smoke_v5/report.json`: bf16,
  channels-last, two updates, and dynamic loss balancing with bounded finite
  gradient-norm EMAs.
- `outputs/train/room_worker_uint8_contract_cpu_v154/report.json`: real-data
  worker-target, offset, and uint8-mask contract.
- `outputs/train/room_worker_counts_contract_cpu_v155/report.json`: worker
  counts, uint8 masks, and full training-path contract.

## Claim Boundary

These are execution-contract and transfer-reduction results. No end-to-end
wall-clock speedup is claimed until a matched multi-step profile runs without a
concurrent chip job. No model-quality conclusion changes in this pass.

## Triton Policy

The existing Triton VQ and junction-incidence kernels remain the selected custom
kernel boundary. The current hot paths do not justify another small Triton
kernel: the dominant cost is model computation, exact Hungarian matching remains
on the CPU by design, and a forward-only replacement for reductions or losses
would add compilation and gradient risk without measured benefit.
