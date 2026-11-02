# GPU Training Optimization v1

Date: 2026-10-31

## Scope

This pass changes execution mechanics only. It does not claim a model-quality
improvement. Residential FloorFlow code is in scope; chips and FloorSet are not.

## Implemented

- Online Shapely/raster verifier work no longer blocks every optimizer step.
  `VerifierRewardReplay` keys rewards by a SHA-256-like target fingerprint,
  vocabulary/tokenization namespace, and refresh age. Missing or expired
  rewards are masked out instead of being silently replaced with stale values.
- Replay entries can be loaded from and written to JSON. Asynchronous entries use
  a background CPU worker; the main step only reads completed rewards.
- Grammar masks, token kind ids, and entity counts are constructed by the data
  path instead of rebuilt in every training loss.
- Device-resident metric accumulation removes per-component `.item()` calls from
  each batch. Metrics are copied to the host once per epoch.
- Loss and gradient finite checks use asynchronous CUDA assertions. Static OOV
  checks run on the CPU batch before pinned host-to-device transfer.
- Conflict-aware backward now performs one full objective backward plus
  gradients for the conflict pair terms only. The previous implementation
  recomputed every registered loss term.
- The gradient balancer and conflict projection operate on configurable parameter
  scopes and intervals. The active config restricts both to token heads.
- Train and validation loaders use shared runtime options, pinned memory,
  persistent workers, and explicit prefetch only when workers are enabled.
- bf16/fp16 autocast, fp16 gradient scaling, TF32 controls, optional
  `torch.compile`, channels-last, and checkpointed scaler state are available
  through `runtime`.
- Beam search expands all candidates in one batched incremental step. KV caches
  are reordered with `index_select` instead of cloned once per candidate.
- Optional Triton kernels accelerate VQ nearest-code lookup and junction
  endpoint incidence. Torch fallbacks remain authoritative on CPU and when the
  feature flag is disabled.
- `--profile-dir` captures one training epoch with `torch.profiler`, exports a
  Chrome trace, and writes a ranked operator summary.

## Evidence

- Relevant token/runtime/replay/Triton tests pass on CPU.
- The Triton VQ and incidence kernels pass small CUDA equivalence tests against
  their Torch implementations.
- A one-case CPU contract smoke completes training, validation, checkpointing,
  and asynchronous verifier shutdown.
- A one-case `cuda:0` contract smoke exercises bf16 autocast, Triton kernels,
  async verifier replay, asynchronous device assertions, and checkpoint output:
  `outputs/smoke/gpu_runtime_cuda_contract_v2/report.json`.
- The 32-case bounded profile completes without the dynamic-padding validation
  failure: `outputs/train/floorflow_runtime_profile_32_v2/report.json`. Its
  three-step trace is 122 MB instead of the 1.2 GB profile produced by the
  earlier unbounded capture.
- `torch.compile` training is compatible with conflict-aware backward after
  disabling AOTAutograd donated buffers. Training uses the Sinkhorn room
  alignment, while validation temporarily bypasses the compiled call to preserve
  exact Hungarian semantics: `outputs/smoke/floorflow_runtime_compile_final_v3/report.json`.
- Boundary-edge features and MiTA mask checks no longer extract GPU scalars to
  Python, removing the known Dynamo graph breaks in those paths.
- Effective settings are persisted under `runtime_acceleration` and
  `verifier_reward_replay` in `report.json`.

## Remaining Measurement

The CPU profile is useful for checking host-side regressions, but it is not a GPU
speed claim. Run a short `cuda:0` profile before choosing a production budget:

```bash
uv run python scripts/train_rplan_full_cad.py \
  --config configs/rplan_generation_floorflow_v4_token_stages.json \
  --output-dir outputs/train/floorflow_runtime_profile_v1 \
  --train-cases 32 --validation-cases 8 --workers 4 \
  --profile-dir outputs/profile/floorflow_runtime_profile_v1
```

The main generator still uses exact Hungarian assignment in validation for
room-mask, polygon, wall, and opening supervision. Training-time typed-state
alignment uses Sinkhorn when compile is active. Replacing the remaining training
loss matchers with Sinkhorn or a GPU assignment solver remains a separate
model-behavior arm because it can change supervision and must pass the matched
screen.

## Selected Triton Policy

Triton is enabled in the active training config for the two implemented kernels.
It is intentionally not used for masked cross entropy: PyTorch already provides
fused, stable forward and backward kernels, and a forward-only Triton replacement
would break gradient semantics.
