# Predicted-Channel Dynamic Graph v1

Date: 2026-11-01

Superseded by `docs/predicted_stage_mita_graph_v2.md`; retained as the
matched control contract.

## Audit Result

The current promoted room model is not encoder-starved:

| Component | Parameters | Share |
|---|---:|---:|
| ResNet18/FPN room encoder | 12,394,860 | 94.0% |
| Query mask decoder | 691,738 | 5.2% |
| Query set/count head | 104,242 | 0.8% |
| Total | 13,190,840 | 100% |

Its successful five-seed confirmation used 5,122 train floors, eight epochs,
batch eight, approximately 40,976 sample exposures and 5,128 optimizer updates
per seed. That budget was sufficient to select an epoch before the endpoint;
more epochs alone are not the next room-quality lever.

The predicted-channel model has a different problem. Its prior screen used:

```text
676,049 parameters
256 train cases x 8 epochs = 2,048 sample exposures
512 optimizer updates
```

That is underpowered for a dynamic graph claim. The previous three-epoch probes
and even the eight-epoch static result cannot distinguish architecture quality
from insufficient exposure.

## Model Upgrade

The following prediction path is now implemented:

```text
room / structure / furniture / support predicted-control fields
-> separate field projections + footprint
-> fused spatial field
-> per-source mean/max evidence context
-> target-channel queries + evidence context
-> masked channel-to-pixel cross-attention
-> image-conditioned typed channel-to-channel attention bias
-> channel logits
-> ordered CAD action graph
-> independent room / wall / opening / furniture stage heads
```

### Evidence Context

`PredictedControlFieldEncoder.encode_with_evidence` now preserves a compact
summary of each predicted-control source instead of discarding per-source
evidence after concatenation. The context is projected into every semantic
channel query.

### Dynamic Typed Graph

`TypedChannelDistributionHead` previously used only:

```text
static family prior + one global learned residual
```

The dynamic arm adds a per-image low-rank channel graph:

```text
A_dynamic = Q(channel queries, field context) K^T / sqrt(d)
bias = static typed prior + residual + dynamic typed bias
```

This graph changes with the current predicted controls rather than only with
the dataset taxonomy. It is an additive attention bias, so hard structural
rules remain a separate safety mechanism.

Both additions are disabled by default for old checkpoints. The new run
contract enables both.

## Added Budget

The new screen contract is:

| Stage | Train cases | Epochs | Batch | Exposures | Updates |
|---|---:|---:|---:|---:|---:|
| Screen | 512 | 12 | 4 | 6,144 | 1,536 |
| Promotion | 1,024 | 16 | 8 | 16,384 | 2,048 |
| Confirmation | 5,126 | 16 | 16 | 82,016 | 5,136 |

The dynamic model is approximately 2.23M parameters with field width 64,
channel width 128, three typed-attention layers, a 24x24 attention memory, and
four dedicated dynamic stage heads. This is still a compact model but more than
triples the previous capacity and triples the screen update budget.

The weak stage heads are now balanced instead of sharing tiny 1x1 classifiers:

| Stage head | Parameters |
|---|---:|
| Rooms | 243,738 |
| Walls | 241,699 |
| Openings | 241,421 |
| Furniture | 246,717 |
| Action graph | 99,593 |

Each stage combines field features, its prior layer, a stage token, and
predecessor evidence. The wall gate is conditioned on room probability, opening
on wall probability, and furniture on room probability. The spatial gate and
CAD stage link bias are image-conditioned rather than scalar broadcast rules.

## Required Ablations

- dense channel attention without evidence context;
- static typed graph without evidence context;
- typed graph with evidence context;
- dynamic typed graph without evidence context;
- full evidence-context + dynamic typed graph.

The full arm must exceed both static typed and dense controls on calibration and
disjoint holdout. The model must also retain the CAD stage heads and explicit
layer contract; a flat all-channel prediction is not an acceptable endpoint.

## Files

- `src/floorflow/model/predicted_channel_model.py`
- `src/floorflow/heads/masked_channel_distribution.py`
- `src/floorflow/model/predicted_channel_checkpoint.py`
- `src/floorflow/training/typed_channel_cli.py`
- `configs/predicted_channel_dynamic_graph_v1.json`

## Verification

```text
13 focused typed-channel tests passed
Ruff passed
checkpoint loader remains backward-compatible by default
```

No GPU run was performed. Dynamic graph quality remains an open empirical gate.
