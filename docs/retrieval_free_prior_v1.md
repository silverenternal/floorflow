# Retrieval-free contour prior

The main inference path no longer reads a train-bank nearest neighbour.
`decode_overlay` accepts `memory=None` when `retrieval_prior_weight=0`.
Contour information is encoded by `FootprintCrossChannelPrior`:

```text
footprint + feature tokens
  -> wall queries
  -> opening queries (wall context)
  -> room queries (wall + opening context)
  -> furniture/support queries (room context)
```

Each route uses the pinned MiTA route attention. Residual scales are zero
initialized, so old checkpoints gain the module without changing their output
until the prior is trained. The learned prior is saved under
`cross_channel_prior` in distribution checkpoints.

`ContourPartitionMemory` remains available as an explicitly weighted
diagnostic. It is not constructed for the main retrieval-free evaluation.

The learned recursive gate is a separate mechanism. `RoomGraphTrustGate`
predicts whether graph geometry should be trusted from target-free logits,
support agreement, entropy and confidence. Its threshold is selected on val,
not test.
