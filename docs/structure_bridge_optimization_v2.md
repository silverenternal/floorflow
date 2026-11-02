# Structure Bridge Optimization v2

The typed v2 contract remains the matched baseline. Its stronger multi-scale
MIta successor is defined in `docs/structure_bridge_mita_v3.md`.

## Scope

This change targets the layer immediately after the residential room learning
head. It is opt-in, keeps the legacy checkpoint contract, and does not modify
chip/FloorSet code or run a GPU experiment.

## Previous Contract

The canonical evaluator blended dense and query room logits, resized them to
224, passed them through `JointControlRepair` with `repair_room=False`,
max-pooled room probabilities to 128, and ran a structure U-Net that computed
an internal 256 map before returning it at 128.

The bridge checkpoint was trained on a different room checkpoint than the one
used at evaluation, and it did not receive query-readout distribution changes.

## Typed v2 Contract

`structure_contract="typed_v2"` adds:

- bilinear probability-space room conditioning instead of max-pooled class maps;
- room-boundary, owner-uncertainty, second-owner, wall-proximity,
  footprint-boundary, and opening-wall-affinity channels;
- direct 128-control-resolution outputs, avoiding an unused 256 computation;
- opening presence and type conditioned on the predicted wall host;
- junction presence and five-class degree heads;
- an auxiliary room-boundary head;
- wall/opening/room-edge relation losses;
- optional room repair through the existing `CoarseLayoutRefiner`;
- training-time room-blend jitter around the deployed dense/query blend.

The legacy path remains the default. New projections that alter predictions are
zero-initialized where identity initialization is possible.

## Reproducible Contract

See `configs/predicted_control_structure_typed_v2.json`.

The trainer now supports:

```text
--room-query-blend
--room-blend-jitter
--structure-contract typed_v2
--structure-relation-weight
--structure-opening-host-weight
--structure-room-edge-weight
--structure-junction-weight
--structure-boundary-weight
```

The canonical evaluator supports `--repair-room`, records the structure
contract, and consumes predicted junction support when constructing the wall
layer.

## Reused Components

- Existing `JointControlRepair` and `CoarseStructureRefiner`.
- Existing room repair through `CoarseLayoutRefiner`.
- Existing room/wall/opening affinity from `structural_consistency.py`.
- Existing skeleton/degree utilities from `geometry_heads.py`.
- Existing dense/query blend adapter from `raster_control_bank.py`.
- Existing `CadStageRuntime` structure consumer.

## Claim Boundary

No new GPU metric result exists yet. Promotion requires the implementation-ready
screen, disjoint holdout, two-seed promotion, and five-seed confirmation.

## CPU Verification

- Targeted structure suite: `29 passed`.
- Full non-chip test set: `853 passed`.
- Changed files pass Ruff.
- All non-chip Python files remain below 600 lines.
- The legacy structure bridge checkpoint loads through the shared resolver on
  CPU and preserves its legacy single-resolution output.
- `uv build` succeeds.
