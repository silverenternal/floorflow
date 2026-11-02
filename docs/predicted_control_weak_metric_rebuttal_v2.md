# Predicted-Control Weak-Metric Rebuttal Audit v2

Date: 2026-11-02

## Scope

This note answers the current weak-metric challenge with code-level causes and
implemented architecture repairs. It is an audit record, not a result claim.
The full491 values below are the pre-repair baseline:

| Metric | Baseline | Gate |
| --- | ---: | ---: |
| Room IoU | 0.327585 | 0.45 |
| Instance F1 | 0.281759 | 0.50 |
| Type accuracy | 0.673841 | 0.75 |
| Wall F1 | 0.487405 | 0.58 |
| Opening F1 | 0.087022 | 0.35 |
| Type-node F1 | 0.727429 | 0.82 |
| Type-edge overlap | 0.361825 | 0.50 |
| Type-graph GED | 7.682281 | <= 6.0 |

The values come from
`outputs/reports/predicted_control_full_stage_val491_gedfixed_v2.json`.

## What The Metrics Do And Do Not Mean

1. Room support is not room identity. Room support IoU is `0.774252`, while
   semantic Room IoU is `0.327585`. The dense model finds the floor area but
   often assigns the wrong room instance or type.
2. HouseMind-style Node F1 is a unique room-type set metric, not instance F1.
   Its precision `0.706797` and recall `0.792457` show that missing types are
   not the only problem; the model also predicts extra types.
3. HouseMind-style Edge overlap is a unique room-type edge metric. Its
   precision is `0.440659` and recall is `0.516361`. It is not the query
   adjacency proxy and must not be defended with that proxy.
4. Type-graph GED follows node substitution plus edge insertion/deletion.
   Baseline edge deletion `2.421589` and insertion `2.279022` are larger than
   either node insertion `0.376782` or deletion `1.156822`.
5. Wall and opening errors are independent: their casewise correlation is
   `-0.001535`. A shared repair knob cannot be claimed as a joint mechanism.

## Implemented Repairs

### Unique query presence

The previous query existence target used independent argmax IoU per query.
Several queries could therefore be marked positive for one target room, after
which the top-k count gate could select duplicate masks. Query existence now
uses exact one-to-one assignment, matching the mask and adjacency losses.

Code:

- `src/floorflow/heads/room_query_set_head.py`
- `_one_to_one_query_targets`
- `room_query_set_loss`

### Presence-consistent room fusion

Room fusion previously multiplied masks by the count-selection mask but not by
query presence. A weak selected slot could still contribute all its semantic
mass. Final query fusion now uses `sigmoid_existence`, so count selection and
query presence are both required.

Code:

- `src/floorflow/model/room_instance_identity_stage.py`
- `src/floorflow/heads/room_mask_fusion.py`

### Eval-space node supervision

The evaluated node set consists of unique room types. A new differentiable
fuzzy-OR projection converts query presence and class probabilities into type
presence and supervises the exact node set.

Code:

- `src/floorflow/objectives/room_adjacency.py`
- `room_type_node_consistency_loss`

### Eval-space edge supervision

The previous graph loss supervised query-mask contact. The canonical metric
collapses instances to unique types and records whether any pair of distinct
types touches. A new projection converts query adjacency and class
probabilities into type-edge probabilities and supervises that exact space.

Code:

- `src/floorflow/objectives/room_adjacency.py`
- `type_adjacency_probability`
- `room_type_adjacency_consistency_loss`

The query-space version above did not survive a matched v3 screen. It improved
the internal adjacency proxy while increasing instance class loss and leaving
final Room/Instance quality unchanged. It is therefore disabled by default.
The active candidate instead projects the final dense room probability into
type-node and type-edge space:

- `dense_room_type_graph_losses`
- `--dense-type-adjacency-weight`
- `--dense-type-node-weight`

### Atomic loss controls

The trainer now exposes independent controls for instance adjacency, canonical
type adjacency, type nodes, and query-set cardinality:

- `--room-instance-adjacency-weight`
- `--room-type-adjacency-weight`
- `--room-type-node-weight`
- `--room-query-set-loss-weight`

## Claims Deferred

The implementation repairs close identified model-to-metric disconnects. They
do not establish an improvement until a matched predicted-control screen
passes:

- Room IoU and instance F1 must improve without type-node regression.
- Canonical type-node and type-edge metrics must be measured from the final
  rendered plan, not from the new training proxies.
- Edge gain must be paired on identical cases and pass the frozen deterministic
  GED evaluator.
- The wall portfolio must be compared against boundary-only control.
- Opening v3 must be compared against the v2 ranker at predicted-wall hosts.

Until then, the full491 table above remains the authoritative baseline.
