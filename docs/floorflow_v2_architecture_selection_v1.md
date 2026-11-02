# FloorFlow Architecture Selection v1

## Scope

This note audits the local CE2EPlan source, evaluates a topology-native successor
without copying its implementation, and records which modules survive matched
small-budget screens. It is internal experiment evidence, not a paper draft and
not an external leaderboard claim.

The RPLAN test split remains closed. All GPU runs use `cuda:0`.

## CE2EPlan Reference

The local source is CE2EPlan commit
`774e670941eeda3e6677303ff79e8d517ae0d1c0`, corresponding to the IEEE TVCG 2026
paper. The released architecture has:

- a flattened 80-point boundary and eight-point entrance condition;
- eight fixed six-value room tokens `[exists, category, cx, cy, w, h]`;
- four Transformer self-attention layers;
- one graph-attention layer over a complete room-pair graph;
- 1,000 diffusion sampling steps; and
- axis-aligned boxes followed by heuristic alignment and vector decoration.

Its practical strengths are short sequences, a mature 300k-step checkpoint, and
strong post-processing. Its representation does not natively contain variable
room polygons, room masks, wall junctions, host-conditioned openings, or an
audited CAD layer sequence.

## FloorFlow v2/v3 Contract

The candidate remains a strict representational superset rather than a port of
CE2EPlan:

| Capability | CE2EPlan | FloorFlow candidate |
| --- | --- | --- |
| Boundary memory | flattened condition vector | boundary segment graph |
| Room count | fixed maximum eight | twelve set queries with predicted count |
| Room geometry | axis-aligned box | mask plus 24-point variable polygon |
| Per-vertex validity | no | yes |
| Room adjacency | fixed complete graph | predicted adjacency and mask-derived query graph |
| Walls/openings | post-processing | typed CAD outputs |
| Sampling | 1,000 diffusion steps | target NFE 1 |
| Audit output | post-processed boxes | one CanonicalFloorplan plus operation log |

### Closed-Loop Query Graph

`RoomQueryGraphRefiner` adds a graph path from predicted room masks to geometry:

```text
boundary-edge memory
  -> room mask and class queries
  -> mask-derived adjacency
  -> room-query message passing
  -> polygon, adjacency, wall, and opening decoding
```

Unlike CE2EPlan's complete graph, this adjacency depends on the predicted spatial
support of each room. The graph layer is optional so old checkpoints remain
loadable.

## Engineering Defects Found

The audit found and fixed three issues that could invalidate comparisons:

1. Training reports recorded CLI defaults rather than the effective model and
   loss configuration. Reports now include both.
2. Padded boundary-edge tokens participated in room-mask cross-attention. The
   decoder now masks invalid memory tokens and has a regression test proving
   that changing padding content cannot change the output.
3. Vector validation recorded an invalid polygon and then continued into GEOS
   operations that could abort the whole evaluation. Invalid geometry is now
   rejected deterministically with a typed validation error.

## Matched 256/64 Screens

Budget: 256 train cases, 64 validation cases, eight epochs, batch size two, seed
2905, and validation-only checkpoint selection. These are single-seed mechanism
screens.

| Arm | Room IoU | Instance F1 | Type accuracy | Count exact |
| --- | ---: | ---: | ---: | ---: |
| Original ResNet18 control | **0.3901** | **0.1388** | **0.4401** | 0.5156 |
| Compact full v2 | 0.2432 | 0.0802 | 0.1406 | 0.5156 |
| Compact v2 minus boundary graph | 0.3630 | 0.1227 | 0.3359 | 0.5156 |
| Compact v2 minus program set | 0.3056 | 0.0960 | 0.3568 | 0.5156 |
| Compact v2 minus point validity | 0.2616 | 0.0916 | 0.2813 | 0.5156 |
| ResNet18 plus full v2 modules | 0.3853 | 0.1333 | 0.3906 | 0.3594 |
| ResNet18 boundary graph only | 0.3824 | 0.1248 | 0.3203 | 0.2031 |
| ResNet18 point validity only | 0.3740 | 0.1240 | 0.4063 | 0.5156 |
| ResNet18 boundary graph plus point validity | 0.3734 | 0.1355 | 0.2500 | 0.5156 |
| ResNet18 mask-derived query graph | 0.3478 | 0.1081 | 0.2344 | 0.5156 |

### Decision

- Retain the explicit boundary representation and point validity as architecture
  capabilities, but do not claim a quality win from the current small screen.
- Reject the compact encoder at this budget because it loses to ResNet18.
- Reject the room-program set head in the current loss composition because it
  consistently slows identity learning.
- Reject separate vector identity heads: two matched epochs produced zero Room
  IoU, reproducing the earlier negative result.
- Keep the mask-derived query graph as an experimental capability only until a
  larger-budget screen or causal attribution establishes a gain.

## Mask-Partition Readout

Checkpoint-level decomposition found that the identity-mask branch is stronger
than the independently regressed vector loop. On the same 1024/256 best
checkpoint, mask matching reaches `0.5384` while polygon Room IoU is `0.4916`.

The decoder now supports an explicit `mask_partition` geometry source. Selected
query masks compete only above the confidence threshold; connected room cells are
converted into variable polygons, clipped to the footprint, and passed through
the same canonical assembler and token codec.

| Readout | Room IoU | Instance F1 | Type accuracy | Successful assembly |
| --- | ---: | ---: | ---: | ---: |
| Polygon loop, epoch 2 | 0.5051 | 0.2613 | 0.6628 | 143/256 |
| Mask partition, epoch 2 | **0.5780** | **0.2625** | 0.6527 | **255/256** |
| Polygon loop, epoch 7 | 0.5112 | 0.3075 | 0.6532 | - |
| Mask partition, epoch 7 | 0.5092 | 0.2836 | 0.5968 | - |

The epoch interaction means the former polygon-IoU checkpoint selection is no
longer aligned with the deployable output. The promoted experiment trains 12
epochs with `mask_partition` validation and checkpoint selection. These values
are single-checkpoint validation evidence, not a test or external SOTA claim.

The 12-epoch, batch-8, seed-2905 mask-selected run reaches training-time
mask-partition Room IoU `0.6104`, instance F1 `0.3480`, and type accuracy
`0.6721`. Re-decoding that checkpoint through the complete canonical assembly
and token round trip gives Room IoU `0.5860`, instance F1 `0.3523`, and type
accuracy `0.6600`. It completes 256/256 cases with exact token round trip.

The directly comparable canonical polygon readout from the same checkpoint is
Room IoU `0.5354`, instance F1 `0.3208`, and type accuracy `0.6659`, but only
111/256 cases finish canonical assembly. The mask-partition gain is therefore
not a training-metric artifact: it survives vectorization, validation, and
serialization while making the output path substantially more robust.

### Five-Seed Canonical Evidence

Five mask-selected seeds (`2905, 19, 29, 37, 43`) use identical 1024/256,
12-epoch, batch-8 training and validation-only checkpoint selection. Canonical
mask-partition results are:

- Room IoU `0.6122`, 95% t-CI `[0.5891, 0.6352]`, range `0.5860-0.6337`;
- instance F1 `0.3749`, 95% t-CI `[0.3505, 0.3993]`;
- matched type accuracy `0.6726`, 95% t-CI `[0.6483, 0.6969]`;
- assembly success `256/256` for every seed; and
- exact token round trip mean `0.9977`, minimum `0.9922`.

Against the polygon readout from the same five checkpoints, mask partition adds
Room IoU `+0.0669` (95% t-CI `[0.0316, 0.1021]`, paired `p=0.0062`) and
instance F1 `+0.0610` (95% t-CI `[0.0180, 0.1040]`, paired `p=0.0169`). The
polygon path completes only `107.4/256` cases on average. Type-accuracy change
has a positive mean but its interval crosses zero and is not claimed as a
significant improvement.

A validation-only confidence sweep selected `0.3` on the raw label map, but the
canonical five-seed mean did not improve: Room IoU was `0.6117` versus `0.6122`
at the original `0.5`, and token round trip fell from `0.9977` to `0.9953`.
The deployable threshold therefore remains `0.5`; the sweep is retained as a
diagnostic that raw-label tuning does not automatically transfer to vector output.

### Mask-Only Objective

Because polygon geometry is no longer the deployable output, the polygon-loop
loss was removed while identity, count, wall, opening, and adjacency losses were
retained. Five-seed canonical mask output improves to Room IoU `0.6413`,
instance F1 `0.4671`, and type accuracy `0.7199`; every seed assembles 256/256
plans. Relative to polygon-loss-on checkpoints, F1 improves by `+0.0922`
(paired `p=0.0013`) and type accuracy by `+0.0473` (`p=0.0337`). Room IoU
improves by `+0.0291` with a mean-positive interval that crosses zero, so it is
reported as a trend rather than a significant paired result.

## Next Gate

The promoted experiment keeps the strict no-module ResNet18 backbone and replaces
only the final geometry readout with mask partition. Promotion requires a
stable Room IoU gain over the checkpoint-aligned polygon baseline with valid
canonical assembly across five seeds. Type, count, and F1 must be reported
separately because the current gain is geometry-specific.

At 4096 training cases, held-out validation confirms the gain: two seeds reach
mean canonical Room IoU `0.6674`, instance F1 `0.5028`, and type accuracy
`0.7573` on 4096 cases with no canonical failures. On the preceding matched
512-case slice, CE2EPlan's released pretrained anchor reports `0.6597`, `0.4814`,
and `0.7470`. FloorFlow uses one forward NFE versus CE2EPlan's 1,000 and 32,768
versus a disclosed 153.6M sample-exposure estimate. The claim remains a
validation-only, budget-disclosed comparison rather than a matched-training
victory.

The seed-37 checkpoint was then evaluated on all 12,018 validation cases:
Room IoU `0.66445`, instance F1 `0.50438`, type accuracy `0.74644`, count MAE
`0.91654`, canonical success `12,018/12,018`, and token round trip `0.99384`.
This is the current frozen quality reference for the 4096 mask-only candidate.

CE2EPlan remains a pretrained external anchor with a declared 300k-step and
1,000-NFE budget. No matched-budget or leaderboard victory is claimed.
