# Predicted-channel and typed cross-attention audit

## Direct answer

Yes, predicted semantic channels exist. `MaskedChannelDistributionHead` gives
every target semantic class its own query token and uses channel-to-pixel
cross-attention. It can therefore predict room categories, wall, openings,
furniture, and other target channels jointly.

No, that head is not yet connected to the non-uniform typed edit graph.
`typed_edit_graph.py` currently builds anchor, affected-room, adjacent-wall,
global-outline, and weak-link regions in NumPy and applies them only during
post-hoc composition. It is not a learned attention component.

The relevant components currently exist in three disconnected places:

| Component | Learned | Predicted channels | Typed graph |
| --- | --- | --- | --- |
| `MaskedChannelDistributionHead` | yes | yes, one query per channel | no |
| `SemanticFamilyStreamMixer` | yes | family streams | no typed adjacency |
| `GraphEditEncoder` | yes | graph nodes | adjacency messages, not spatial channels |
| `typed_edit_graph.py` | no | routing masks | typed non-uniform regions |

## Why the previous probe is not decisive

The existing masked-channel probe used contour-only features, only three
epochs, and 256 training cases. Its validation Room IoU was about `0.061`.
That is an underbudgeted, weakly conditioned probe, not a fair test of an
all-channel predicted-control model.

## Architecture to build

The new head should preserve one first-class query token per semantic target
channel and add two learned interactions:

1. Channel-to-pixel cross-attention, reusing the masked spatial memory path.
2. Channel-to-channel typed attention, where the link weight comes from a
   learnable residual over a family prior:

```text
room <-> room        adjacency / room-boundary evidence
room <-> wall        boundary ownership
wall <-> opening     host-wall attachment
room <-> opening     access relation
furniture <-> room   containment
same family          weak structural prior
unrelated families   suppressed
```

The typed relation is an additive bias on attention logits, not a hard mask.
This keeps hard validity rules outside the model while allowing the model to
learn deviations inside the feasible envelope.

Predicted-control reliability and edit masks should gate the channel queries.
The gate must be produced from predicted inputs and must not use target values.

## Required arms

- Dense channel self-attention.
- Typed channel links.
- No-link ablation.
- No edit/control gate ablation.

All arms use the same encoder, split, seed, loss weights, NFE, and decode.

## Immediate budget

Start with train256/cal64 for eight epochs. Extend to 24 epochs or a documented
plateau only if room/wall/opening and graph metrics improve together. A
three-epoch diagnostic cannot reject the architecture.

## Eight-epoch implementation audit

The first full implementation was intentionally rejected as a final model:

- Typed channel head, eight epochs, 256 train cases: Room IoU `0.0575`.
- Dense channel head, matched budget: Room IoU `0.0524`.
- Moving the mask output from `16x16` to the `128x128` field and normalizing
  CE weights changed the loss from roughly `30-50` to roughly `3-4`, but did
  not solve instance fragmentation.

The data-chain audit found two further structural mistakes:

1. Predicted room, wall, opening, and furniture evidence was compressed into a
   shared feature before prediction, discarding the useful per-channel logits.
2. The flat predictor had no room-instance or stage handoff, so predicted room
   evidence fragmented into hundreds of connected components.

The corrected connection keeps predicted channel logits as a prior and learns a
bounded residual. More importantly, it moves typed attention into the staged
CAD action graph described in
`docs/predicted_control_cad_action_map_v1.md`; a flat 65-channel output is no
longer treated as the endpoint.

## Dynamic typed graph follow-up

The static typed graph has now been upgraded with two learned image-conditioned
mechanisms:

- compact per-source predicted-control evidence is injected into channel
  queries;
- a per-image low-rank channel graph adds a dynamic attention bias on top of
  the family prior and global residual.

The old checkpoint contract remains valid because both mechanisms default off.
The stronger budget and ablation matrix are defined in
`configs/predicted_channel_dynamic_graph_v1.json` and
`docs/predicted_channel_dynamic_graph_v1.md`.
