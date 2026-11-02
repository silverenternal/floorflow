# Floorplan Token Layer Optimization v2

## Scope

This change upgrades the unified floorplan token layer from a deterministic
codec to an optionally trainable and grammar-constrained output path. It is
CPU-validated and does not modify chips/FloorSet code.

The follow-up training integration, typed structural memory, vocabulary
protocol, and runtime contract are documented in
`docs/token_training_protocol_v3.md`.

## Trainable Token Head

`FloorplanTokenHead` performs causally masked, BOS-shifted next-token prediction
from the room and boundary context. Padding is masked in both attention and the
sequence loss. `UnifiedFloorplanGenerator` exposes this as an opt-in
`token_logits` branch, and `unified_generator_loss` includes token CE when token
targets are available.

The default vector-head path is unchanged.

## Grammar and Constrained Decoding

`token_grammar.py` provides:

- a stable token vocabulary;
- allowed-next-kind rules;
- grammar masks;
- constrained argmax;
- prefix violation diagnostics;
- next-token CE.

This catches invalid prefix transitions before semantic decoding.

The tokenizer also supports a reversible `VD` delta-vertex mode. Subsequent
ring vertices are encoded relative to the previous vertex; absolute `VERTEX`
tokens remain supported for backward compatibility. Explicit `DELTA` and
`ABSOLUTE` streams fail closed if their coordinate token kinds are mixed.

Language-level token primitives live in `floorplan_token_io.py`. The canonical
validation and tokenization pipeline remains in
`unified_floorplan_tokens.py`, with the established public imports preserved.

## Quantizer Contract

The token stream records the coordinate quantizer scale. Decoding restores that
scale, and junction validation now uses the same quantizer instead of a hidden
fixed scale of 16.

## Raster Diagnostics

Canonical plans now render:

- room labels;
- wall labels;
- opening labels;
- furniture labels;
- composed visible labels;
- footprint and overlap/boundary/outside diagnostics.

## Consistency Metrics

Canonical round-trip reports fingerprint equality, entity-count equality,
adjacency, junction degree, quantization error, and grammar prefix violations.
The canonical evaluator exposes these through `--canonical-token-readout`.

## Claim Boundary

The trainable token head is implemented and tested as an opt-in interface. No
GPU training result or token-generation SOTA claim is produced by this change.

## Verification

- Non-chip tests including the repository-wide line-count guard:
  `869 passed`.
- Twenty-four targeted token tests cover causal attention, padding, grammar
  masks, constrained argmax, adaptive quantizer round trips, coordinate-mode
  rejection, rendering, and canonical prefix diagnostics on CPU.
- Changed files pass Ruff and all non-chip Python files remain below 600 lines.
- `uv build` succeeds.
