# Predicted-Control Metric-Bound Architecture Audit v294

Date: 2026-11-02

Scope: residential predicted-control generation only. No chip/FloorSet code and
no test-split run is included.

## Metric-to-Network Diagnosis

The authoritative full491 baseline is Room IoU `0.32759`, instance F1 `0.28176`,
Wall F1 `0.48741`, Opening F1 `0.08702`, Node F1 `0.72743`, Edge overlap
`0.36183`, and GED `7.68228`. The previous loss registry did not expose those
bottlenecks at the same granularity as the model.

Three concrete defects were found:

1. The dynamic balancer treated wall, opening, and furniture as one
   `stage_family` scalar even though the metrics and parameters are independent.
2. Dense type-node and type-edge losses were recomputed inside one
   `dense_type_graph` term, and room-query existence, count, class-count, and
   cardinality were hidden inside one query-set term.
3. Class-cardinality consistency was an unnormalized L1 sum across every room
   class. On the one-case CPU contract it reached `352.8` while CE was about
   `1.6`, so it dominated early optimization before the first dynamic update.

The dynamic balancer also called `autograd.grad` on constant zero terms. A
disabled or inapplicable objective could therefore crash the first balancing
update. Constant terms now report zero norm and zero touched parameters.

The first real smoke also exposed an uninitialized optional dense-loss bucket
when both dense graph weights were zero. The atomic-term lists are now
initialized before the conditional branches.

The matched screen then showed that the dense Edge objective was effectively
flat: BCE remained around `1.002-1.010` while the target edge mean was about
`0.0175`. BCE alone does not place enough priority on the sparse positive edge
set. Soft Dice overlap terms are now added to both dense type-node and type-edge
losses, with zero-default behavior for old configurations.

## Implemented Repairs

### Atomic loss ownership

The following objectives now expose live tensors independently:

- `stage_wall`
- `stage_opening`
- `stage_opening_type`
- `stage_furniture`
- `query_existence`
- `query_count`
- `query_class_count`
- `query_class_cardinality`
- `dense_type_adjacency`
- `dense_type_node`

The registry uses the same effective coefficient as the non-dynamic path,
including the outer room-instance scale. Wall/opening/furniture no longer share
one gradient norm.

Disabled furniture-containment and room-instance diagnostics previously logged
the active furniture-query scalar by default. Missing objectives now log zero,
so a disabled path cannot appear live in reports.

### Cardinality stability and prior

Class-cardinality consistency is now a bounded relative L1 distance. The query
class-count head is initialized from taxonomy source counts, allocating the
initial room budget sparsely across classes instead of assigning an expected
count near `16` to every class. This adds no parameters and uses dataset
provenance already present in the taxonomy.

Room-query class and partition cross-entropy now use inverse-square-root
taxonomy frequencies, normalized to mean one and bounded to `[0.25, 4.0]`.
This targets rare room types without reweighting from each batch's labels.

### Relational room edge head

Canonical Edge overlap is a room-type edge metric, while the previous adjacency
head used cosine query similarity plus a mask-contact scalar. The v5 candidate
adds a symmetric pair residual over:

- normalized query pair features;
- absolute and multiplicative pair differences;
- 2 px mask contact;
- count-selected endpoint presence.

The pair residual starts small but nonzero, so both its hidden and output layers
receive first-step gradients. It is independently switchable and preserves the
mask-only v4 control. Room-graph parameters rise from `77,145` to `143,196`;
the full candidate is `6,946,647` parameters.

Construction of the optional pair MLP consumes random initialization draws.
Those draws are isolated and the global RNG is restored before later shared
modules are built. A same-seed contract verifies that control and candidate have
identical weights everywhere outside the two relational-only tensors.

## External Architecture Check

Recent released work supports the chosen separation rather than a wholesale
backbone replacement:

- FMLM (arXiv `2604.04859`) uses a structured markup grammar for heterogeneous
  floorplan tasks.
- TLC-Plan (arXiv `2602.07100`) separates global room layout from polygon-level
  geometry.
- DPLAN (arXiv `2606.21159`) makes connectivity constraints explicit.
- RenoLayout (arXiv `2610.21366`) separates immutable structure from editable
  design points.

FloorFlow already has the corresponding stage and layer boundaries. The current
bottleneck is not another monolithic backbone; it is binding each evaluated
weak atom to a live, comparable consumer. Existing FloorFlow modules are reused
rather than copying these external systems.

## Verification

- Thirty-seven focused room, graph, query, stage, balancer, and joint-training
  tests pass.
- The one-case contract verifies nonzero parameter reach for wall, opening,
  opening-type, dense Edge, dense Node, query presence, count, class-count, and
  cardinality terms.
- Ruff passes for all touched Python modules.
- All touched Python files remain below 600 lines.

These are architecture and gradient contracts, not score evidence.

## Matched Screen Result

The first matched 256/64 seed19 full-model screen did not promote the v5
bundle. Both arms selected epoch0. Joint minus control produced:

| Metric | Delta | 95% CI | Decision |
| --- | ---: | ---: | --- |
| Room IoU | +0.000353 | [-0.000154, 0.000935] | no effect |
| Instance F1 | +0.001095 | [-0.002277, 0.004570] | no effect |
| Count MAE | +0.140625 | [-0.25, 0.5625] | no improvement |
| Wall F1 | -0.003094 | [-0.003977, -0.002244] | significant regression |
| Edge overlap | -0.007579 | [-0.022703, 0.005131] | no improvement |
| GED | -0.15625 | [-0.453125, 0.15625] | no effect |

The control's best checkpoint was epoch0; later full-model updates reduced
Instance F1 from `0.21493` to as low as `0.17560` and raised count MAE from
`4.14063` toward `7.45313`. The next training policy should therefore preserve
the residual prior by training room/graph adapters first, rather than
unfreezing every channel and stage simultaneously.

The final adjacency head used count-selected query presence, while the
adjacency and type-graph losses still used raw query existence. All graph
consumers now share one `room_graph_query_presence` tensor. A hard intersection
made validation signal disappear entirely in the adapter contract, so
target-mask edge supervision now keeps a `0.05` presence floor as a sample
weight instead of dropping the pair. The previous arbitrary `0.05` Hungarian
IoU cutoff is also removed. The adapter contract changes from zero to `16`
average valid pairs per two cases, and the adjacency objective has a non-zero
gradient norm.

The graph-focused adapter configs use `train_scope=room_graph_adapter`,
initialize from the matched control's epoch-0 checkpoint, and lower the
learning rate to `1e-4`. They freeze the room decoder and all unrelated
channels; only adjacency, graph conditioning, query-set, count, and three
bounded fusion scalars train. The sparse class-count prior is a non-persistent
architectural buffer, so checkpoint loading cannot silently overwrite it.
Control, relational-only, prior-only, and joint configs share the same frozen
backbone.

The first adapter CPU contract exposed that the gradient balancer still passed
frozen tensors to `autograd.grad`. It now filters parameters internally. The
adapter contract executes a real balancing update over roughly `575k`
trainable graph/query elements while the room decoder and all unrelated
channels remain frozen.

The count-selected second graph pass now reuses the same adjacency head and
graph conditioner after the final query-count decision. Its class and mask
residuals update the room output before owner readout, so the final audited
adjacency is no longer loss-only. The adapter contract verifies nonzero
adjacency and type-edge gradients plus finalized class/mask tensors with no
additional parameters.

Auditing the six-epoch graph-only control exposed a first-step dead zone:
loading the legacy checkpoint restored both graph-to-output residual heads to
zero, even after source initialization was made nonzero. Count-aware
second-graph models now re-warm-start those heads after loading. The one-case
CPU contracts change Room IoU from `0.3354` to `0.4187-0.4278` after one epoch
and restore a nonzero graph-message gradient path.

Sparse graph BCE and Dice are now computed per case and then averaged, matching
the canonical case-mean Node/Edge protocol. The previous pooled loss allowed a
large plan with many edges to dominate a batch regardless of case count.

Wall supervision was recall-heavy: full-val precision is `0.4091` versus recall
`0.6759`. A precision-oriented Tversky term with `alpha=0.7`, `beta=0.3` is now
registered as its own `stage_wall_precision` objective. Family validation also
logs wall/opening/furniture precision and recall, which were previously hidden
behind F1.

Opening has the same asymmetry in the CPU diagnostic: precision `0.1292`
versus recall `0.4962`. A footprint-masked precision Tversky term is now an
independent `stage_opening_precision` atom, while opening foreground geometry
and opening-type classification remain separate objectives. The existing
opening-wall affinity loss is also exposed as `stage_opening_host`, penalizing
opening probability farther than two pixels from the target wall band.

The canonical CAD consumer exposes a larger Opening issue: on one real case,
pixel Opening F1 is `0.2049` while the raster-to-wall-plan vector Opening F1 is
`0.0`. Pixel F1 and vector F1 are therefore not interchangeable. The new
`opening_vector_proxy` keeps both metrics in validation, but vector F1 still has
to become a selection gate and the interval/host decoder must consume the final
stage opening maps during training.

Wall/opening/furniture focal loss now also normalizes per case before batch
averaging. Together with Dice and Tversky, every CAD stage objective now matches
the case-mean family metric instead of allowing a large drawing to dominate all
gradient from a batch.

## Next Decision Gate

Run the v5 control matrix on identical cases and seeds:

| Arm | Relational pair | Sparse count prior | Rare-class weighting |
| --- | --- | --- | --- |
| v5 control | no | no | no |
| pair only | yes | no | no |
| prior only | no | yes | no |
| rare-class only | no | no | yes |
| joint | yes | yes | yes |

Promote only if instance F1, Node F1, Edge overlap, and GED improve together
without lowering Room IoU, validity, watertightness, or outside-footprint
exactness. The first run is a mechanism screen, not a publication result.
