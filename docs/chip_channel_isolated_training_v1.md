# Chip Channel-Isolated Training v1

Status date: 2026-11-01. This is an implementation and CPU profiling result,
not a trained quality claim.

## Direct answer

Different typed channels and the cross-attention fusion can now be parameter-
isolated and trained in separate stages:

- block/local message passing;
- global specification and complexity;
- structural tokens;
- constraint tokens;
- port tokens;
- semantic and degree experts;
- explicit global-to-structure and constraint-to-port bridges;
- cross-attention plus output norm.

The encoder has also been split into `encode_channels()` and `fuse_channels()`.
When no pre-fusion channel parameter requires gradients, channel encoding runs
under `torch.no_grad()` and only the cross-attention stage builds an autograd
path.

This is not yet full representation independence. Structural, constraint and
port tokens are still pooled from the shared block representation. Separate
optimizers can isolate their trainable parameters, but a learned structural
state can still inherit block-feature errors. A future independent-channel
encoder must duplicate those input paths rather than merely freezing prefixes.

## Critical gradient repair

Four zero-gated branches had permanently zero gate gradients because both the
gate and the final projection were initialized to zero:

- global-spec conditioner;
- semantic node/port encoder;
- netlist hyperedge conditioner;
- structural token residual.

Their outputs remain exact no-ops at initialization. The final projections now
retain normal initialization, so the scalar gate receives a nonzero first-step
gradient and can open. Regression tests pin this contract in
`tests/test_chip_gate_gradient_contracts.py`.

## Training scopes

Examples:

```bash
# Train one channel while the rest of the network is fixed.
uv run python scripts/train_chip_slot.py ... \
  --training-scope channel:local+downstream

# Fine-tune only cross-attention and let the flow/backbone adapt.
uv run python scripts/train_chip_slot.py ... \
  --training-scope fusion+downstream

# Freeze channel encoders, train fusion, then unfreeze everything.
uv run python scripts/train_chip_slot.py ... \
  --training-scope-schedule \
  '0:channels+downstream,1000:fusion+downstream,2000:all'
```

When changing the scope of an existing arm, resume weights with
`--resume-weights-only`; the old optimizer param groups intentionally do not
match a different isolated scope. A scheduled run keeps all parameters in one
stable optimizer group set so it can reopen exact scopes at transitions.

The schedule is manifest-visible and requires step-zero initialization. It is
incompatible with `--freeze-existing`; a static non-full scope is incompatible
with the old backbone-unfreeze switch because that switch could silently reopen
parameters outside the declared scope.

## Research basis

### HGT

Hu et al., *Heterogeneous Graph Transformer*, WWW 2020,
<https://arxiv.org/abs/2003.01332>. HGT uses node- and edge-type-dependent
projections and relation attention. This supports keeping the typed channel
contract explicit instead of treating every token as one homogeneous space.

### GraphGPS

Rampášek et al., *Recipe for a General, Powerful, Scalable Graph Transformer*,
NeurIPS 2022, <https://arxiv.org/abs/2205.12454>. GraphGPS decouples local
message passing from global attention. Its modular layer directly supports the
new channel-encoding / cross-attention boundary.

### AdapterFusion

Pfeiffer et al., *AdapterFusion: Non-Destructive Task Composition for Transfer
Learning*, <https://arxiv.org/abs/2005.00247>. AdapterFusion separates knowledge
extraction in individual adapters from a later knowledge-composition stage.
This is the main basis for the `channels` to `fusion` schedule.

### GraphMAE

Hou et al., *GraphMAE: Self-Supervised Masked Graph Autoencoders*, KDD 2022,
<https://arxiv.org/abs/2205.10803>. GraphMAE supports target-free channel warm
starts through masked feature reconstruction and scaled cosine error. This is a
future warm-start option, not part of the current training claim.

### Perceiver IO

Jaegle et al., *Perceiver IO: A General Architecture for Structured Inputs &
Outputs*, ICLR 2022, <https://arxiv.org/abs/2107.14795>. The input-to-latent
cross-attention is separable from later latent computation. This supports
precomputing frozen channel tokens and training only the fusion consumer.

## Pinned open-source references

The following trees were cloned under `third_party/reference/` and recorded in
`third_party/README.md`:

| Repository | Commit | License | Relevant mechanism |
|---|---|---|---|
| `rampasek/GraphGPS` | `2801570` | MIT | Local MPNN plus global attention |
| `THUDM/GraphMAE` | `b14f080` | MIT | Masked feature reconstruction |
| `adapter-hub/adapters` | `53a1ea1` | Apache-2.0 | Independent adapters plus fusion |
| `acbull/pyHGT` | `85eaccd` | MIT | Relation-specific typed attention |
| `lucidrains/perceiver-pytorch` | `202cd4e` | MIT | Separate input cross-attention |

No upstream source was copied into the model. The local implementation preserves
the existing typed encoder interfaces and only separates its stages.

## CPU profile

Command:

```bash
uv run python scripts/profile_chip_channel_training.py --device cpu \
  --scope all --scope channels --scope fusion \
  --scope channel:local --scope channel:structural \
  --output outputs/reports/chip_channel_training_scope_profile_cpu_v2.json
```

Synthetic setting: hidden 64, 8 samples, 48 blocks, 64 pins, two typed graph
layers. Five measured repeats after one warm-up.

| Scope | Trainable parameters | Median typed forward+backward |
|---|---:|---:|
| all | 288,517 | 20.91 ms |
| channels | 106,762 | 19.59 ms |
| fusion | 16,777 | 14.21 ms |
| local | 1,267 | 16.77 ms |
| structural | 576 | 12.83 ms |

The fusion-only stage is about 32% faster than full typed backward and reduces
optimizer-tracked parameters by 94.2%. This profile measures the typed
conditioning subsystem only and must not be presented as full training speedup.

The same script on `cuda:0` with twelve measured repeats gives 21.81 ms for
`all`, 21.21 ms for `channels`, 18.23 ms for `fusion`, 21.75 ms for `local`,
and 26.18 ms for `structural`. Fusion-only is about 16% faster than the full
typed stage in this synthetic GPU profile. The individual channel probes are
noisy because forward cost is unchanged and only the backward graph shrinks.
The GPU result is a short operator-budget measurement, not a trained model
result.

A second `--mode full` run includes the complete synthetic `supervised_loss`
forward/backward rather than only typed conditioning:

| Scope | Trainable parameters | `cuda:0` median |
|---|---:|---:|
| all | 288,517 | 58.64 ms |
| fusion+downstream | 181,755 | 40.50 ms |
| local+downstream | 166,245 | 37.38 ms |
| structural+downstream | 165,554 | 32.71 ms |

In this short synthetic profile, fusion+downstream is about 30.9% faster than
the all-parameter path. The model is deliberately small and the batch is
synthetic. Real training speedup, optimizer-state memory and data-loader overlap
must be measured on the actual 5k-subset arm before making an iteration-cost
claim.

## Required validation sequence

1. Use an existing matched checkpoint; do not compare independently initialized
   channel stages.
2. On the frozen 5k subset, compare `all` against
   `channel:local+downstream`, `channel:structural+downstream`, and
   `fusion+downstream` with identical data order and evaluation budget.
3. Confirm the scheduled route `channels -> fusion -> all` has no layer-wise
   dead branches and no non-finite gradients at each transition.
4. Promote only if legalized cost and V_rel pass jointly; parameter savings or
   step-time savings alone are not a model-quality result.
5. Use the full dataset only for a finalist plus matched control.
