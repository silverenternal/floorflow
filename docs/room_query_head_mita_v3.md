# Room Query Head MIta v3

Date: 2026-11-01

## Why This Is the Next Room Lever

The confirmed room model improved paired Room IoU, instance F1, count MAE, and
type accuracy, but absolute val491 Room IoU remains 0.31. The old checkpoint is
still heavily encoder-dominated:

| Component | Parameters | Share |
|---|---:|---:|
| Room encoder | 12,385,240 | 75.8% |
| Query decoder | 2,423,450 | 14.8% |
| Query-set head | 1,525,142 | 9.3% |
| Total | 16,333,832 | 100% |

The v3 candidate does not enlarge the encoder. It moves capacity into the
query-set head, which controls room existence, cardinality, and class-count
consistency.

## Hybrid Routing

The confirmed dense set attention remains present. Each query-set layer adds a
causal MIta branch over the same room queries:

```text
query tokens
-> dense set attention
-> MIta router selection
-> sample-conditioned top-k budget
-> gated sparse residual
-> presence / count / class-count heads
```

The MIta branch is hybrid rather than a replacement. This protects the already
confirmed dense identity path and gives the model a second, routed mechanism
for deciding which proposed rooms mutually inform count and existence.

## Capacity and Budget

The query-set hidden width rises from 128 to 192 and the routed set depth from
one to two layers. New training defaults use 512 cases for 16 screen epochs,
1,024 cases for 20 promotion epochs, and five seeds for full confirmation.

Every validation row records route entropy, budget, effective expert fraction,
and gate mean. This makes it possible to distinguish a genuine sparse
mechanism from a route branch that collapsed to a constant or never activated.

## Acceptance

The route branch is rejected if it only lowers count MAE while Room IoU or
instance F1 falls. It must beat an equal-parameter dense query-set control and
a static-budget MIta control on calibration and disjoint holdout.

The existing five-seed room checkpoint remains canonical until the v3 screen
and promotion close.

## Files

- `src/floorflow/heads/room_query_set_head.py`
- `src/floorflow/heads/raster_control_bank.py`
- `src/floorflow/training/raster_instance_cli.py`
- `configs/room_query_head_mita_v3.json`
