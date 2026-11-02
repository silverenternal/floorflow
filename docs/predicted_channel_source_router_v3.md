# Predicted Channel Source Router v3

Date: 2026-11-01

## Audit

The v2 prediction head still had three bottlenecks:

| Component | v2 behavior | v3 behavior |
|---|---|---|
| Channel queries | one static learned vector per class | sample-conditioned dynamic offset |
| Predicted-source interaction | one shared global evidence vector | per-channel room/structure/furniture/support routing |
| Pixel cross-attention | unconditional residual | query/context-conditioned gate |

The channel-to-pixel attention itself was structurally valid, but every channel
read the same field representation in the same way. This was too simple for
channels with different semantics.

## Source-Conditioned Queries

The field encoder now preserves six source tokens from the predicted room,
structure, furniture, support, room-program, and learned envelope fields. The
program and envelope tokens make the CAD schedule and exterior boundary explicit
instead of leaving them inside a pooled fusion vector. A low-rank hypernetwork
generates a channel-specific query offset from the flattened source state.

Every channel then performs typed source attention over the four tokens. A
structural prior favors:

- room channels toward room, structure, and support;
- wall and opening channels toward structure and room;
- furniture channels toward furniture and room.
- room channels additionally toward the room-program token;
- wall/opening channels toward the learned envelope token.

The prior is an additive bias; the learned attention remains sample-dependent.

## Gated Pixel Cross-Attention

At every masked channel layer, the pixel-attention residual is multiplied by a
gate predicted from the current channel query and the attended spatial context.
The gate therefore controls both sample and semantic-channel behavior instead
of applying the same unconditional update to every channel.

## Capacity

The v3 candidate with room-instance and furniture stages has about 6.22M
parameters:

| Group | Parameters |
|---|---:|
| Shared field | 278,976 |
| Typed channel head | 1,711,371 |
| Dynamic query adapter | 565,569 |
| Source attention | 91,138 |
| Cross-attention gates | 99,075 |
| Explicit envelope source | 256 |
| Program instance router | 23,066 |
| Room instance stage | 636,022 |
| Furniture query stage | 539,152 |
| Furniture semantic router | 25,562 |
| Channel-family adapters | 51,523 |
| Cross-attention LoRA | 18,507 |
| Spatial Conv-Adapter | 4,401 |
| CAD action graph | 253,010 |
| Dynamic stage heads | 2,799,793 |

The shared field remains unchanged. The added capacity is concentrated in
channel identity, source selection, and cross-attention control.

## Controls

Component-isolated arms are required. The full model must beat static queries,
ungated cross-attention, and source attention alone. A shuffled-source control
tests whether predicted-source semantics are actually consumed.

Validation logs report source-attention entropy, source gate, and cross-gate
strength.

## Files

- `src/floorflow/heads/predicted_channel_adapters.py`
- `src/floorflow/heads/masked_channel_distribution.py`
- `src/floorflow/model/predicted_channel_model.py`
- `configs/predicted_channel_source_router_v3.json`
