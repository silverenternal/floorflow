# FloorFlow Token Training Protocol v3

## Scope

This revision connects the canonical token branch to the real RPLAN data and
training path without running a GPU experiment. It does not claim token
generation SOTA and does not replace the existing vector-CAD or FEVP evidence.

The downstream consumer, token readout, conflict projection, and dynamic
batching follow-up are documented in
`docs/token_downstream_optimization_v4.md`.

## Data Contract

`floorplan_token_targets.py` converts one canonical target plan into a
train-only token target:

- case ids and source hashes become stable placeholders;
- operation targets/details become placeholders;
- geometry-heavy token fields use `<REF>` in the default `reference` mode;
- `exact` and `quantized` geometry modes remain available for controlled
  ablations;
- room/wall/opening/case identity leakage is therefore removed from the
  structural generation objective;
- overlong sequences raise `TokenTargetOverflow` instead of being silently
  truncated;
- each sequence carries ids, attention mask, loss mask, CAD-stage ids, length,
  and OOV count.

Exact geometry remains the responsibility of the vector heads. The token
branch learns the canonical structural order and CAD-stage sequence rather than
memorizing an unbounded coordinate vocabulary.

## Vocabulary

The vocabulary is serialized with a version, special tokens, and SHA-256.
Checkpoint loading rejects a vocabulary checksum mismatch. Candidate training
uses the first 4096 train cases for vocabulary construction and hard-fails on
any OOV batch.

Observed target statistics:

| Split | Cases | Median | p95 | Max | OOV |
| --- | ---: | ---: | ---: | ---: | ---: |
| train, vocabulary source | 4096 | 193 | 242 | 319 | built-in |
| validation audit | 256 | - | 239 | 261 | 0 |

The reference-mode vocabulary has 78 content tokens, 79 output classes, rather
than 29,480 classes from raw exact-coordinate tokens on only 512 cases.

## Model Chain

```text
boundary tokens + query-aware room pool
  -> CadActionGraph stage tokens
  -> vector decoder stage summaries
  -> TypedTokenMemory
  -> causal stage-conditioned token decoder
  -> grammar-masked token CE
  -> token/vector structure consistency
  -> joint weighted loss + gradient balancer
```

`QueryAwarePool` replaces the previous mean over all room queries, including
inactive slots. `TypedTokenMemory` keeps boundary, room, CAD action, and vector
stage tokens distinct with type embeddings. The token head consumes this
multi-token memory, applies stage embeddings, masks context padding, and retains
BOS-shifted causal decoding.

## Objective

The token objective includes:

- grammar-masked next-token CE;
- token-kind balancing;
- CAD-stage balancing;
- room/wall/opening cardinality consistency with vector heads;
- a soft room-partition excess penalty as a differentiable validity proxy;
- optional online gradient-norm balancing and one-step pairwise gradient
  diagnostics.

These structure terms do not replace exact validity or watertightness
evaluation. They only prevent the token branch and vector heads from optimizing
contradictory structural counts during training.

## Runtime and Checkpoints

`floorplan_token_training_runtime.py` owns vocabulary loading, model-size
synchronization, forward/loss arguments, CLI overrides, OOV policy, gradient
balancing, and checkpoint metadata. Checkpoints record:

- vocabulary JSON and SHA-256;
- target protocol, max length, geometry mode, and OOV policy;
- token and structure weights;
- kind/stage weights;
- gradient-balancer state.

Older checkpoints without token parameters can initialize the vector model when
the token branch is enabled; token-only missing weights are explicitly allowed.
A checkpoint with a different token vocabulary checksum is rejected.
Resuming also restores the gradient-balancer EMA/scales and global training
step.

## Reproduction

Build the vocabulary:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_rplan_token_vocabulary_v1.py \
  --dataset .data/external/CE2EPlan/dataset/dataset_json/data_train.json \
  --output-dir outputs/token_vocabulary/rplan_train4096_v1 \
  --limit 4096 \
  --geometry-token-mode reference
```

Run the CPU contract smoke:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_rplan_full_cad.py \
  --config configs/rplan_generation_floorflow_v4_token_stages.json \
  --output-dir outputs/train/floorflow_v4_token_contract_cpu_v2 \
  --device cpu --contract-smoke --epochs 1 --workers 0
```

The contract report records token CE, grammar violations, OOV count, token
lengths, structure consistency, and per-term gradient norms/cosines.

## Verification

- Non-chip pytest suite: `872 passed`.
- All changed files pass Ruff.
- All non-chip Python files remain below 600 lines.
- `uv build` succeeds.
- The repository-wide package-layout line guard remains red only for three
  chips files outside this task's allowed edit scope.

## Claim Boundary

The current evidence establishes an executable, auditable token-training
interface and a bounded CPU contract result. It does not establish that token
decoding improves Room IoU, GED, RPLAN SOTA, or external full-CAD SOTA. Those
claims require controlled CPU/GPU training and held-out evaluation with this
contract frozen.
