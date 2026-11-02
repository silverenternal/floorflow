# Room Owner Readout and Runtime v1

Date: 2026-11-01

## Why

The previous readout collapsed all query masks of the same class before the
wall/topology consumers. Two adjacent bedrooms could therefore become one
semantic component even when the query decoder represented them separately.
The count gate was also hard and non-differentiable, and screening repeatedly
ran expensive Shapely verification for every arm.

## Owner-Preserving Readout

`src/floorflow/heads/room_owner_readout.py` now exposes:

- `query_mass`: query masks after optional existence and count-selection gating;
- `owner_map`: the selected query at every pixel;
- `owner_confidence`: the selected query mask probability;
- `owner_foreground`: owner confidence inside the footprint;
- `owner_boundary`: adjacent pixels owned by different queries;
- `owner_preserving_logits`: the selected query's class probability, with the
  previous class-mass readout as fallback outside confident owner regions.

`RasterInstanceControl` publishes these fields without changing the default
`room_logits`, `dense_logits`, or `fused_logits`. Existing checkpoints therefore
remain behavior-compatible for previous artifacts.

The wall fusion API now accepts `owner_boundary` and supports:

- `owner_boundary`;
- `owner_wall_gated`.

This allows same-class room instances to create a wall seam before semantic
class marginalization.

## Soft-Count Training Interface

`RoomQuerySetHead` returns both:

- a hard top-k selection for inference;
- an expected-top-k soft selection for training.

For sorted query rank `r`, the soft inclusion probability is
`P(count > r)` under the predicted count distribution. Its sum equals expected
cardinality. `RasterInstanceControl` uses the soft mask in train mode and the
hard mask in eval mode when `query_count_gate` is enabled.

## Engineering Changes

- Sinusoidal query position embeddings are cached by shape/channels/device.
- The 1x1 mask projection is computed once at the finest decoder level and
  average-pooled for coarse levels. This is equivalent for complete pooling
  windows and removes two repeated 1x1 convolutions.
- `RasterInstanceControl.module_contract()` reports the exact active forward
  order and train/eval count-routing mode.
- The page-floor wall evaluator defaults to cheap raster structural proxies.
  Shapely/fpeval runs only with `--exact-structural` for finalist arms.

## Verification

Focused tests cover:

- same-class owner boundary preservation and gradients;
- footprint masking and fallback behavior;
- expected-cardinality identity and gradients for soft top-k;
- train-soft/eval-hard count routing;
- cached position embeddings and projection/pooling equivalence;
- owner-boundary wall fusion.

```text
31 passed
968 non-chip tests passed with the package-wide line guard excluded
Ruff passed
uv build passed
all non-chip Python files remain <600 lines
```

No GPU training or evaluation was run in this pass. Metric promotion remains
open until the owner-boundary and soft-count mechanisms clear matched screens.

## v342 Class-Consumer Repair

A later tensor-contract audit found that the selected-query gather expanded its
class-index dimension to one rather than the number of room classes. PyTorch
accepted the narrower index, returned the first class only, and broadcast that
single value across the class axis in `owner_preserving_logits`.

The gather now carries all room classes. Historical `True`/`False` owner
configuration values retain absolute-confidence behavior, while the explicit
`"relative"` mode is available only through
`configs/predicted_channel_v6_owner_relative_only.json`. Deterministic tests
cover all-class preservation, low-absolute/high-relative ownership, config
round-trip, and rejection of unknown modes.

The repository-wide line-count test currently reports one pre-existing
out-of-scope chips file, `src/floorflow/chip_flow/hierarchical_flow.py`, above
600 lines. That file was not modified in this pass.
