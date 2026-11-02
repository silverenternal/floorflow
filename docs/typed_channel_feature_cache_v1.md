# Typed-Channel Frozen Feature Cache v1

Date: 2026-11-01

## Why

Partial training still paid for predicted-control stage inference and field
encoding on every epoch. Those tensors are invariant whenever the room,
structure, centerline, opening checkpoints and field configuration are frozen.

## Usage

```bash
uv run python scripts/train_typed_channel_attention_v1.py \
  ... \
  --train-scope group_ssf \
  --feature-cache-root outputs/cache/typed_channel_frozen_v1
```

The first encounter with each case writes one compressed NPZ containing:

- post-stage predicted controls;
- field features;
- evidence context;
- source tokens.

All later epochs and validation passes read the same entry. The cache is
rejected for `full` and `field` scopes because field weights can change.

## Integrity

`manifest.json` stores a SHA-256 fingerprint over:

- image size and room blend;
- predicted-control root;
- room, structure, centerline, and opening checkpoint paths, sizes, and mtimes;
- model options that affect field evidence or source tokens.

A changed fingerprint fails closed and requires a new cache root.

## Numeric Contract

Controls are stored in float32 and field features in float16. A CPU test
verifies that full and cached model outputs agree within tolerance on finite
values. Training metrics still evaluate every output channel.
