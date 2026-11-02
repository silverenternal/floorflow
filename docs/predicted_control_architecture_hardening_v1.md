# Predicted-Control Architecture Hardening v1

## Root cause

The strongest currently evaluated structure checkpoint is the 774,261-parameter
legacy raster bridge. It has no typed junction/room-boundary output, no global
MIta structure route, and no centerline path. More capable checkpoints existed in
code but could not be consumed reliably because:

- evaluation scripts reconstructed `JointControlRepair` as legacy instead of
  loading its checkpoint configuration;
- the all-channel field encoder fused all controls with one static convolution;
- CAD action tokens were precomputed from raw controls, so downstream stage
  tokens did not consume upstream room/wall/output predictions;
- zero-initialized action and opening-host residual scales blocked the first
  gradient into their context networks;
- routing diagnostics were logged without any loss pressure;
- real float16 predicted controls failed in the float32 control-prior path;
- adaptive CUDA pooling broke deterministic backward.

## Implemented architecture

### Checkpoint-aware deployment

Predicted-control fidelity, plausibility, and raster materialization now load
`JointControlRepair` through its checkpoint config. Legacy checkpoints remain
loadable; typed checkpoints no longer get silently reconstructed as legacy.

### Typed source-spatial router

`TypedSourceSpatialRouter` routes room, structure, furniture, and support fields
at every spatial location for each CAD family:

- rooms;
- walls;
- openings;
- furniture.

Each family owns separate source logits and projection heads. The shared fusion
convolution receives the concatenated family deltas, so family-specific evidence
is retained. The promoted source-router config has 6,259,579 parameters,
including 32,673 source-router parameters.

Training scopes now expose the whole router and each family-specific route.

### Causal CAD action refresh

The all-channel model now rebinds predicted room, wall, and opening outputs into
`CadActionGraph` before downstream stages:

    rooms -> refreshed wall token -> refreshed opening token -> furniture

Action update, dynamic-link, and MIta route residual scales start at `0.01`
instead of zero. This removes the dead-gradient initialization while retaining a
small residual.

### Action-conditioned channel queries

CAD stage tokens are pooled into a control context and projected into the
channel-query evidence before residual prediction. The information path is now
one-way:

    predicted controls -> CAD action graph -> typed channel queries

Later stage refreshes still update the causal stage heads, so this introduces no
query/action feedback loop.

### Room query residual initialization

The query room stage previously multiplied its residual by a zero-initialized
scale, making the instance decoder irrelevant to the dense room output at the
first step. The scale now starts at `0.1`, preserving a bounded residual while
allowing query masks and classes to affect Room IoU immediately.

CAD stage readout now consumes the current channel-query residual rather than
only the raw prior. This restores a live graph from stage losses to family
adapters and query residuals while room-stage preservation remains explicit.

### Query count gate

The reusable `RoomQuerySetHead` is now part of the query room stage. It outputs
query presence, a room-count distribution, per-class counts, and candidate/soft
selection masks. Fusion uses soft selection during training and deterministic
candidate selection during validation. Count and class-count losses are included
in instance-room training, directly targeting the audited instance-F1/count gap.

The trainer defaults are aligned with the promoted architecture: room instance
hidden width `128`, room query-set width `96`, and furniture query width `128`.
This prevents a nominally full run from silently training smaller weak-metric
branches than the configuration under audit.

The dynamic gradient balancer now registers every independently weighted
objective, including room instance, room partition, furniture assignment, and
containment. Previously enabling dynamic loss weights replaced the aggregate
loss with an incomplete term set and silently discarded those weak-metric
supervision signals.
An exact model-config parity test now covers every architecture switch and
width, including the 32-query furniture budget.

Predicted per-class counts now also calibrate foreground room class logits via a
bounded learned prior. This gives the class-count head a live downstream
consumer and targets the audited room-type bottleneck without changing room
support pixels.

Query existence now uses the same exact one-to-one assignment as the mask and
adjacency losses. The previous independent argmax target could mark several
queries positive for one room. Final room fusion also multiplies selected
queries by their existence probability, so the count gate cannot admit a weak
slot at full mask mass.

### Query-room adjacency

The query room stage now exposes a symmetric `[B,Q,Q]` room adjacency matrix.
Adjacency logits use dual-query dot products, shared with
`VectorPrimitiveHead` through `heads/room_adjacency.py`. A differentiable
mask-contact target supervises the head, so graph output is connected to the
training objective rather than being a detached diagnostic. Validation reports
adjacency precision, recall, F1, and MAE, and model selection includes the
adjacency F1. A graph conditioner propagates adjacency messages into query
features, room class logits, and spatial mask logits before partition fusion.
During training, predicted query masks are IoU-matched to GT room masks and the
GT mask-contact adjacency is gathered into query slots, providing direct edge
supervision. Matching is one-to-one Hungarian, so duplicate queries cannot be
assigned the same target node and taught false inter-query edges. Inference
remains target-free.

The evaluated HouseMind-style graph collapses room instances into unique room
types. Instance-edge supervision alone therefore did not constrain the measured
consumer. The room stage now also projects query adjacency and class
probabilities into type-edge space and supervises type-edge F1 directly. A
second projection supervises the unique-type node set from query presence and
class probabilities. These losses target the canonical graph contract rather
than a merely self-consistent mask property.

### Room-local furniture stage

The furniture stage now consumes the updated room-stage logits and pools room
tokens from program/action-conditioned features. Detached furniture ablations
detach both room features and room logits, preventing gradient leakage into the
room/wall backbone. The typed trainer enables furniture queries and room
assignment by default.

Dynamic furniture budgets now activate the top target-free class/mask/room
scores instead of the first N unordered query slots. Vector legality also
resolves the predicted room-class assignment to the best-overlapping concrete
room instance before clipping, preventing same-class rooms from being confused.

### RPLAN adapter contract

RPLAN boundary and room polygons that exceed fixed tensor limits are now
perimeter-resampled instead of vertex-prefix-truncated. Zero-area boundaries and
duplicate source room IDs fail explicitly. Openings are marked valid only when
their referenced host wall remains inside the tensorized wall budget.

RPLAN verifier target keys now hash all structural targets, including geometry
that normal training tokens replace with `<REF>`. Collation also rejects mixed
grammar vocabulary widths and partially present target keys.

### Predicted-control typed edit bridge

The predicted CAD output now exposes the fields required by the shared editable
state: room identity tokens, existence/class logits, adjacency, and ordered stage
tokens. `predicted_control_typed_state` maps these into `TypedFloorplanState`
without target labels, providing the input contract for the next predicted-edit
and strict-local projection run.

### Dense type-graph consumer bridge

The first query-space type-node/type-edge losses improved only the internal
query-adjacency proxy and did not improve final rendered metrics. In a matched
v3 screen, epoch 0 changed Room IoU by `+0.001305`, Instance F1 by `-0.000210`,
and count MAE by `+0.296875`; instance class loss increased from `2.2306` to
`2.4065`. Query-space type losses now default to zero.

The replacement loss consumes the final dense room probability and supervises
type presence and pairwise class contact after resizing to the target grid. It
is opt-in through `--dense-type-adjacency-weight` and
`--dense-type-node-weight`.

The runtime adapter now exposes `room_instance_probability` and owner
map/boundary fields. Without that wiring the dense loss silently skipped and
logged zero. A joint backward test locks the contract.

A first 128/32 matched screen showed that supervising the pre-owner probability
did not move final Node/Edge/GED. The candidate loss now uses softmax of the
final owner-preserving `room_instance_logits`, so the objective and the rendered
consumer share the same tensor contract.

The zero-effect root cause was the owner gate: during early training, soft count
selection confidence is below the inference threshold, so every query fell back
to dense class mass. Dense losses received no path into the query parameters.
Owner foreground now uses confidence-weighted blending in training and the hard
threshold only at inference.

Dense type-edge contact now derives its neighborhood from the canonical 2 px
room-adjacency tolerance. The previous 3×3-only contact could label a true
evaluator edge as negative when room boundaries were separated by the expected
wall thickness.

### Owner-preserving room readout

The room stage previously collapsed query masks back into dense class mass
before the wall/structure consumer, discarding the query-owner boundary learned
by adjacency conditioning. The shared `room_owner_readout` now supplies the
final room logits and can be disabled for matched ablation with
`--no-owner-preserving-room-readout`.

### Opening and wall consumer contracts

Opening candidate identity now uses object identity rather than dataclass
equality, so duplicate centers/scores cannot share the wrong score rank or
interval-neighbor gap. Center-width predictions are bounded by host-wall length,
and image-space family typing aggregates the same interval evidence used by
wall-space decoding.

Wall candidates are stably sorted by layer evidence before `max_walls`
truncation. Previously the proposal-source prefix controlled which candidates
survived budgeting, so high-evidence walls could be discarded silently.

Wall candidate fingerprints now bind a v7 evidence-ranked generator contract,
so old source-prefix/order caches cannot be reused after the ordering change.
Batched opening attachment also fails closed when the decoding host plan and
the destination plan do not share wall identity; the old path could preserve a
`wall_index` from a different wall ordering.

Routed opening rows now reject zero-length hosts, and final vector projection
clamps opening width to a positive interval bounded by the destination wall.

### Predicted edit-state and add/delete projection

The predicted-control edit adapter now applies the query-count selection mask
to room existence and validates token, existence, class, adjacency, and stage
token shapes before constructing `TypedFloorplanState`.

`project_vector_plan` no longer assumes equal pre/post room, wall, and opening
counts. Objects outside the edit region are preserved from the before plan;
objects touching the edit region are accepted from the after plan. This permits
room/wall/opening addition and deletion while preserving zero outside changes.

Wall/region intersection samples at every wall pixel rather than nine fixed
fractions, so one-pixel edits cannot be missed. Outside-change auditing moves
its region masks to the accelerator holding the tensor differences.

Typed edit-family inference now unions before and after labels. A conversion
from room to wall/opening therefore opens the destination channel instead of
being classified only by the source family.

### Routing objective

`typed_routing_regularization` now applies loss pressure to:

- source entropy;
- family-source usage diversity;
- channel-source attention entropy;
- CAD-stage routing entropy;
- effective MIta expert fraction;
- LoRA rank budgeting.
- room query-set MIta routing and expert usage.

The trainer exposes `--routing-regularization-weight`; the promoted source-router
config uses `0.01`.

### Typed opening host

The typed-v2 structure refiner can consume wall, junction, and room-boundary
evidence jointly through small nonzero 1x1 presence/family host heads. The real
`24/38/52/28` field smoke has nonzero both host gradients and nonzero MIta route
gradient with route entropy `0.9950`.

### Spatial gap MIta router

The center-width opening head can now route four gap tokens through sparse MIta
attention over a deterministic `16x16` spatial evidence memory. The existing
self-token fusion remains the control path, so promoted center-width checkpoints
stay loadable. The gap residual starts at `0.01` rather than zero.

### Opening host feature contract

An opt-in v3 opening-ranker contract adds nine host-aware features: wall
continuity, visibility gap, thickness, endpoint margin, same-wall candidate
coverage, endpoint distance, collinear neighbor count/length, and parallel-wall
fraction. The promoted 21-feature v2 ranker remains backward compatible. Both
joblib and torch persistence paths reconstruct the v3 feature contract.

### Source-aware wall candidate portfolio

Wall candidates now preserve proposal provenance: room boundary, footprint,
predicted host wall, or one of three skeleton readout scales. Candidate caches
store source IDs separately from the boundary control stream, and the wall
scorer applies a trainable source embedding before graph and geometry scoring.

### Wall observation candidates

Raster wall readout is parameterized by close kernel, simplification fraction,
and minimum segment length. A matched val128 diagnostic found:

| Readout | Wall F1 | Precision | Recall |
| --- | ---: | ---: | ---: |
| k3/s0.006 control | 0.523933 | 0.445203 | 0.702087 |
| k5/s0.006 | 0.531671 | 0.458523 | 0.694394 |
| k7/s0.006 | 0.531340 | 0.463392 | 0.680578 |
| k5/s0.010 | 0.567422 | 0.529699 | 0.659460 |

This remains diagnostic only. The candidate cache now supports a
`skeleton_portfolio` proposal stream combining room/footprint boundaries with
k3-s0.006, k5-s0.006, and k5-s0.010 skeleton candidates. Boundary-only control
proposals remain isolated.

### Deterministic GPU paths

- action pooling uses fixed-kernel avg/max pooling;
- channel-grid pooling uses a deterministic fixed-grid average;
- partition-consistent owner attention reshapes level tokens to 4D before bias
  construction;
- float16 control caches produce finite float32 priors.

## Verification

- 18 focused GPU/contract tests pass for one-to-one query presence,
  presence-consistent fusion, canonical type-node/type-edge gradients, and the
  joint predicted-control backward contract.
- Real-field all-channel smoke on `cuda:0`: logits `[1,65,64,64]`, source routing
  `[1,4,4,64,64]`, three causal action refreshes.
- Real-field typed bridge smoke on `cuda:0`: both opening-host gradients and the
  MIta route gradient are nonzero.
- `1120 passed, 1 skipped` for the scoped floorplan suite.

The unfiltered suite also contains separately maintained chip shape-head/file
size failures and an NLP todo-count assertion; none are modified here.

## Next measurement order

1. Train the promoted source-router 512/64 screen with a disjoint 64-case
   holdout; reject collapsed source or stage routing before reading metrics.
2. Run static-field, global-source-attention, spatial-router, and causal-refresh
   controls under equal parameters and updates.
3. Train the typed-v2 host-conditioning bridge under the aligned structure
   budget and compare against the legacy bridge on the same room distribution.
4. Train the wall scorer on the isolated skeleton candidate portfolio and
   evaluate Wall, Opening, junction, continuity, Room, Node, Edge, GED, and
   safety together.
5. Promote one full 1024/128 confirmation arm only after holdout gates pass;
   keep test splits closed until freeze.

## Full-Val Gap Diagnosis

The reproducible audit in
`outputs/reports/predicted_control_metric_gap_audit_v1.json` shows:

- opening F1 has the largest absolute gap (`0.263`);
- instance F1 (`0.218`) and edge overlap (`0.138`) dominate the room/graph gap;
- Room support IoU is `0.774` while semantic Room IoU is `0.328`, so the room
  failure is identity/count/type, not missing room support;
- Node F1 `0.727` is much higher than edge overlap `0.362`; 75 cases have strong
  nodes but weak edges;
- Room IoU correlates `0.687` with edge overlap and edge overlap correlates
  `-0.533` with GED;
- Wall/opening F1 correlation is `-0.0015`, supporting independent layer
  contracts;
- cases with 13+ target rooms fall to Room IoU `0.248` and GED `14.82`.

These observations set the priority order: room identity/count/type, typed edge
consistency, wall precision, then opening host geometry.

The promoted model allocates approximately `25.1%` of parameters to the typed
channel head, `17.3%` to the room instance stage, `11.7%` each to wall and
opening stages, `7.9%` to furniture, and only `1.1%` to the room graph
conditioner. Wall and opening are therefore not generically under-sized; adding
uniform width is not the next justified action. The actionable imbalance is the
graph consumer and metric-specific readout semantics.
