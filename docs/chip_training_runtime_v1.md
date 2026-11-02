# Chip Training Runtime v1

Status date: 2026-10-27. This note records measured runtime changes; it makes
no quality claim.

## Decision

The original chip loop spent most of each step in Python synchronization rather
than GPU kernels. `project_conditioned_latent` looped over cases and MIB groups
twice per step. On a fixed 128-case train batch, replacing it with a vectorized
masked lower-median operation reduced measured gradient projection time from
`55.11 ms` to `1.49 ms` (`37.08x`) while preserving output exactly and changing
the maximum gradient by `1.5e-11`.

Fused AdamW reduced the optimizer step from `6.61 ms` to `0.29 ms` (`22.46x`).
A foreach EMA update reduced `0.73 ms` to `0.25 ms` (`2.93x`). Data gathering
was `1.80 ms`, so neither framework migration nor a larger model should be used
to explain or fix the original throughput problem.

PyTorch Lightning is installed, but the `lightning` namespace and Accelerate
are not. Trinity's Lightning trainer is useful as a reference for manual
optimization, callbacks, EMA, and compile configuration, but adopting the
framework would not remove the measured Python bottleneck. DeepSpeed and FSDP
are also unjustified for a 5.9M-parameter single-GPU model. A thin Lightning
adapter remains conditional on a documented checkpoint or callback reliability
gain.

## Organization

- `src/floorflow/chip_flow/train_cli.py`: CLI and model configuration.
- `src/floorflow/chip_flow/train_objective.py`: flow objective and schedule.
- `src/floorflow/chip_flow/train_evaluation.py`: curve selection and dev evaluation.
- `src/floorflow/chip_flow/training_runtime.py`: optimizer, EMA, and runtime helpers.
- `scripts/train_chip_slot.py`: maintained training/evaluation orchestration entrypoint.

All files remain below 600 lines. Checkpoints include a runtime manifest, and
new final comparisons must use one frozen runtime for both candidate and matched
control.

## Full-data reliability contract

The 2026-10-30 full-chain review fixed the following conditions:

- `evaluate()` restores the caller's train/eval mode, so a validation callback no
  longer leaves subsequent training steps in eval mode.
- checkpoint selection uses EMA by default because official evaluation loads
  `--use-ema`; `--eval-state raw` remains available for diagnostics.
- checkpoints save the CUDA generator, NumPy/Python/torch generator, batch RNG,
  and balanced-stream sampler queue/cursor state.
- checkpoint and curriculum report writes use same-directory temporary files and
  atomic replacement.
- exploratory training requires a frozen `--train-indices` subset. Full-data
  training requires the explicit `--allow-full-data` override.
- train/dev overlap is checked by `(source path, source layout)` before training.
- evaluation case rows retain raw and legalized feasibility flags.
- refiner evaluation uses a stable per-batch seed instead of replaying one noise
  sequence for every batch.
- paired statistics align unique case indices and expose `lower_is_better` so
  cost/gap improvements are not counted backwards.

## Evidence

- `outputs/reports/chip_training_framework_benchmark_v1.json`
- `tests/test_chip_projections.py`
- `tests/test_chip_training_runtime.py`
- `scripts/benchmark_chip_training_runtime.py`
