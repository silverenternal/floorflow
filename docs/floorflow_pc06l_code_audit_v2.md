# FloorFlow PC-06L Code and Architecture Audit v2

Date: 2026-10-30

## Scope

This audit covers the residential predicted-control path only. It does not
modify or evaluate chips/FloorSet work. The goal is to distinguish claims that
are actually wired through train, inference, vector readout, and CAD consumers
from modules that exist only as names or offline helpers.

## Confirmed Assets

- The ordered CAD action graph is
  `envelope -> program -> rooms -> walls -> junctions -> openings -> furniture
  -> validation`.
- Rooms, walls, openings, and furniture are independent output layers.
- Candidate-wall scoring is anchored to deterministic wall evidence and uses a
  learned filtration descriptor, candidate-graph attention, and phased budgets.
- Opening host routing supports multiple openings on one wall and is retained
  as a typed candidate mechanism; the detector remains the production fallback.
- Program-conditioned rooms and junction-stage injection remain opt-in because
  their matched screens were mixed rather than uniformly positive.

## Gaps Found and Closed

### Vector furniture contract

Before this pass, furniture was a raster layer with no final vector object and
no CAD consumer. The following path now exists:

```text
room/program/action evidence
  -> CadFurnitureStage
  -> dynamic furniture queries
  -> room-attached polygons
  -> RasterWallPlan / vector plan JSON
  -> CanonicalFurniture-compatible evaluator state
```

The existing `DynamicFurnitureInstanceHead` is reused. The new stage adds
program/action conditioning and is disabled by default so old checkpoints are
bit-compatible.

### Validation repair consumer

The CAD action graph produced a validation token but no downstream network
consumed it. `CadTopologyRepairHead` now consumes wall, junction, opening-host,
validity, and validation-token state. It emits:

- wall keep/delete logits;
- bounded endpoint deltas;
- opening-host logits.

Execution is fail-closed: endpoint displacement has a hard maximum and each
sample may apply at most a configured number of graph edits. At initialization
the repair is an identity map. The candidate-plan composer now consumes repair
outputs, so the head is not an orphan module.

### Missing contract test

`fptodo.json` referenced `tests/test_cad_action_layer_contract.py`, but the file
did not exist. The test now checks stage order, furniture gradients, repair
boundedness, and actual candidate-plan consumption.

### Audit schema mismatch

The v2 audit report was written with a v1 schema string. The report, default
path, and schema are now consistently v2.

### Checkpoint loader drift

Six candidate/opening/vector scripts independently duplicated the model
constructor. A checkpoint trained with furniture queries would therefore fail
to reload because these scripts omitted the new architecture fields.
`build_predicted_channel_model` is now the single config-to-model contract and
is used by those scripts.

### Dead room-stage path

With `preserve_room_stage=True`, the room prior is intentionally preserved and
the learned room stage is not used. This is safe for old checkpoints but means
the room stage parameters are a dead branch. A `--learned-room-stage` training
option and a live-gradient test now define the matched alternative; the
production default remains unchanged.

### Training and environment hygiene

- `train_typed_channel_attention_v1.py` had crossed the 600-line limit; stage
  validation helpers were moved to `training/typed_channel_eval.py`.
- Re-running training appended stale rows to `train.jsonl`; the log is now
  reset at run start.
- Evaluation imported helpers from a CLI training script; it now imports the
  reusable training module.
- Query rendering accepted only a four-dimensional footprint even though the
  real validation path supplies the standard `[B,H,W]` contract. The smoke
  exposed the resulting 384-vs-batch broadcast error; both `[B,H,W]` and
  `[B,1,H,W]` are now tested and accepted.
- Two executable paths defaulted to GPU 1. They now default to GPU 0.

## Initialization and Safety Properties

- Furniture-query branch: opt-in; absent from old checkpoint state dicts.
- Room-stage learning: opt-in; old checkpoints preserve room evidence.
- Junction stage: opt-in; old checkpoints keep the wall token in that slot.
- Repair head: identity at initialization; hard endpoint and edit budgets.
- Furniture vectors: attached to a room by majority room-label overlap;
  uncontained components are dropped.
- No target-derived input is used by any new inference path.

## Verification

- Contract and architecture tests cover furniture gradients, vector readout,
  checkpoint reconstruction, repair bounds, and candidate-plan consumption.
- The full pre-change suite passed: `907 passed`; the post-change suite passed:
  `962 passed`.
- A real-data 16/8 one-epoch integration smoke passed on `cuda:0`. The
  furniture-query loss was finite (`6.5818`), the checkpoint reloaded through
  the shared builder with no missing/unexpected keys, and one-case vector
  readout completed with schema validity and conditional watertightness `1.0`.
- The smoke reported furniture F1 `0.0`. This is an underfit integration result,
  not a mechanism failure or promotion result.
- Ruff, JSON parsing, the 600-line guard, and the v2 cad-action contract audit
  all pass.

## Remaining Scientific Work

The code contracts are now explicit, but they are not scientific successes by
themselves. Required matched screens are:

1. learned room stage versus preserved room prior;
2. furniture-query stage versus furniture stage map;
3. bounded learned repair versus no repair;
4. optional junction stage after layer retraining;
5. optional program-conditioned room bias at its no-regression setting.

Each screen must use an identical split, seed, decoder, and visibility mask.
Any promoted mechanism must then run on the established 256/64 mechanism split
before full-split evidence.

The immediate matched task is `PC-06M`. The 16/8 smoke only certifies that the
data, gradient, checkpoint, and vector-serialization contracts execute.
