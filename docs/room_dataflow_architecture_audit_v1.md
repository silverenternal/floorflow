# Room Dataflow and Architecture Audit v1

Date: 2026-11-01

## Active Forward Path

The promoted room checkpoint contains `13,190,840` parameters:

| Module | Parameters | Share |
|---|---:|---:|
| ResNet18 + FPN room encoder | 12,394,860 | 94.0% |
| Query mask decoder | 691,738 | 5.2% |
| Query set/count head | 104,242 | 0.8% |

The active path is:

```text
page-floor RGB + footprint
-> ResNet18/FPN hidden map [B,128,H,W]
-> dense room logits
-> avg_pool2d(hidden, 2)
-> 3-layer room query decoder
-> class logits + 24 query masks + query features
-> RoomQuerySetHead (presence, count, class counts)
-> hard top-k count selection
-> class-mass room fusion
-> interpolation to full resolution
-> optional dense/query probability blend
```

If "the layer after the learning head" means after the class/mask query heads,
the next learned module is `RoomQuerySetHead`. It is a shallow global-evidence
head: a 1x1 projection, adaptive pooling, two 64-wide MLPs, and a global
count/class-count readout. Its discrete top-k result is the only path by which
count prediction changes the fused map.

## Training Consumers

The forward path is followed by sibling losses rather than one additional
feature layer:

1. exact Hungarian class + mask supervision on query outputs;
2. matched query mask BCE/Dice/Tversky and optional same-class separation;
3. dense fused partition CE/Dice;
4. full-resolution fused identity CE/Dice/boundary;
5. query presence, global count, and per-class count CE;
6. optional teacher, program, or boundary auxiliary losses.

The active promoted configuration enables the first five through room-mask,
partition, fused-identity, and query-set terms. Teacher, program, adaptive
reference, slot state, mask pooling, full-resolution sampler, and boundary
refiner remain off.

## What Is Good

- The encoder/decoder separation is clear and reuses one shared feature map.
- Hungarian supervision preserves a permutation-invariant set formulation.
- Dense and instance outputs are explicitly separated and evaluated separately.
- Count prediction is connected to a downstream consumer. Before this fix the
  head was trained but ignored.
- Worker-side target extraction and pinned collation remove a full
  H2D -> D2H -> H2D training round trip.

## Structural Weaknesses

### 1. The readout is much weaker than the encoder

The query decoder is only about 5% of parameters and runs on a half-resolution
hidden map. The dominant encoder does not have a correspondingly expressive
room-identity and owner readout. More encoder training alone is therefore an
inefficient way to improve instance identity.

### 2. Class-mass fusion erases same-class instance identity

`fused_room_mask_logits` multiplies every query mask by its class probability,
sums all query mass into the same class, then normalizes by class. Two adjacent
bedrooms can therefore collapse into one semantic component even when the query
decoder separated them. Room F1, Node F1, and wall provenance cannot fully
recover from that information loss.

The next substantive architectural change should expose an owner map or an
owner-boundary map before class marginalization. Wall seams should be derived
from query-owner changes, not only from class-label changes.

### 3. Count selection is global and discrete

The count head sees only global average/max evidence plus aggregate query
statistics. Its count is `argmax`, its query selection is hard `topk`, and the
selected boolean mask has no differentiable path into mask fusion. Count
calibration therefore depends on auxiliary CE and works empirically, but the
query-to-map interface is still discontinuous.

A stronger design is a set-cardinality state that exposes soft query ownership
during training and a single discrete top-k only at inference.

### 4. The query decoder is dense and resolution-split

At 256 input, the half-resolution decoder memory has
`128^2 + 64^2 + 32^2 = 21,504` tokens for 24 queries. Three full cross-attention
layers repeatedly attend to all tokens, while mask logits are produced at
128x128 and only interpolated to 256. This is both expensive and weak at room
and wall boundaries.

The local Deformable-DETR reference is the relevant replacement candidate:
sample a small set of learned points per query and scale instead of attending
to every token.

### 5. Optional-branch sprawl

`RasterInstanceControl` exposes many semantically overlapping switches:
existence gate, adaptive reference, mask-pooled class, full-resolution class
sampler, owner reassignment, boundary owner refiner, query set, slot state,
count gate, dense refiner, program head, and program prior. Most are inactive,
several are closed negative arms, and their ordering is encoded procedurally in
one `forward`.

This is flexible for research but not yet elegant as a final architecture.
The promoted core should be explicit, with closed mechanisms moved to adapters
or ablation implementations.

## Fast Engineering Wins

These do not change supervision:

- cache sinusoidal position embeddings by `(H,W,C,device)`;
- compute the 1x1 mask feature at full decoder resolution once and average-pool
  the projected features for coarse levels;
- compute class fusion only for the final layer and retained auxiliary layers;
- use cheap raster room/wall/continuity metrics for arm screening and invoke
  Shapely/fpeval exact checks only for the top one or two arms;
- cap screen evaluation to one representative readout per mechanism.

## Highest-Impact Model Upgrade

The next model change should be:

```text
query features -> mask logits
-> query-owner posterior
-> owner map + owner-change boundaries
-> class map and wall seams
```

This directly addresses same-class room merging, count-to-query coupling,
instance F1, Node/Edge/GED, and wall provenance in one consumer chain. It is
more valuable than another independent residual head.

## Verdict

The architecture is a solid modular research scaffold, but it is not yet
"perfect" or final-paper elegant. The main imbalance is a 94%-parameter encoder
feeding a compact semantic readout that discards query ownership before the
wall/topology consumers. The owner-preserving readout is the strongest next
architectural step; deformable/sparse decoding and protocol-specific wall
adaptation follow after that.

## Implemented Follow-up

The first owner-preserving readout, soft-count training interface, cached
position embedding, single-projection decoder path, module-order contract, and
cheap structural screening switch are implemented in the next pass. See
`docs/room_owner_readout_and_runtime_v1.md`. No metric claim is attached until
matched offline evaluation runs.
