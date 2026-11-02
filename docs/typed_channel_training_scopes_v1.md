# Typed Channel Training Scopes v1

Date: 2026-11-01

## Answer

Different channel families and cross-attention can now be trained separately.
This is possible at two levels:

1. module scope: field, complete channel head, cross-attention, MIta routes,
   stage heads, or all parameters;
2. channel-family scope: room, wall, opening, or furniture adapter plus its
   downstream stage components.

The shared channel queries remain one tensor, so family isolation is provided
by zero-initialized family adapters rather than toggling slices of an ordinary
`requires_grad` tensor.

## Scopes

Run the typed-channel trainer with:

```bash
--train-scope cross_attention
--train-scope opening_adapter
--train-scope furniture_adapter
```

| Scope | Trainable elements |
|---|---:|
| full | 6,218,329 |
| field | 278,976 |
| channel_head | 1,711,371 |
| cross_attention | 561,411 |
| cross_attention_lora | 18,507 |
| adalora | 18,507 |
| lora_ranks | 72 |
| spatial_conv_adapter | 4,401 |
| bias_only | 39,962 |
| channel_queries | 59,843 |
| group_adapters | 51,523 |
| group_ssf | 16,128 |
| group_delta | 32,323 |
| stage_heads | 3,052,807 |
| mita_routes | 1,352,115 |
| room_adapter | 1,082,996 |
| wall_adapter | 824,670 |
| opening_adapter | 832,690 |
| furniture_adapter | 1,392,226 |

The fastest mechanisms are `group_ssf` (16,128 elements) and
`spatial_conv_adapter` (4,401 elements), followed by `cross_attention_lora`
(18,507). All keep the full backbone and base
attention weights fixed.

For repeated epochs, add:

```bash
--feature-cache-root outputs/cache/typed_channel_frozen_v1
```

This skips stage-runtime and field encoding after the first case encounter.
See `docs/typed_channel_feature_cache_v1.md`.

## Contract

- The forward graph and losses remain identical.
- Frozen parameters are verified to retain their original values.
- Every scope still evaluates room, wall, opening, furniture, graph, and safety
  metrics across the full output.
- Group scopes require `channel_group_adapters=true`.
- Optimizer construction receives only parameters with `requires_grad`.

## Files

- `src/floorflow/training/typed_channel_scopes.py`
- `src/floorflow/heads/predicted_channel_adapters.py`
- `src/floorflow/heads/masked_channel_distribution.py`
- `src/floorflow/training/typed_channel_cli.py`
- `configs/typed_channel_training_scopes_v1.json`
