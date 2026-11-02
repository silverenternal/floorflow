# FloorFlow Entrance and Partition-Dual Architecture Audit

## Scope

This note records the architecture-first response to the local CE2EPlan
implementation. No RPLAN test data is used. The comparison is against
CE2EPlan commit `774e670941eeda3e6677303ff79e8d517ae0d1c0` as an
architecture reference, not as code to copy.

## CE2EPlan Capability Audit

The released implementation uses:

- an 80-coordinate exterior condition repeated across every room token;
- an eight-coordinate entrance condition repeated across every room token;
- eight fixed room tokens containing existence, class, center, width, and height;
- four Transformer layers and a complete room graph;
- 1,000 diffusion steps; and
- box alignment and room decoration after sampling.

FloorFlow should not adopt this bottleneck. Its stronger representation is a
variable set of room queries, per-query masks, variable polygons, and one
canonical CAD object. The goal of the entrance work is to match CE2EPlan's
explicit condition discipline while retaining that stronger output contract.

## Defects Found

### Missing entrance condition

Canonicalization rotates the footprint ring to an order-invariant start vertex.
As a result, the first canonical boundary segment is not the supplied entrance.
The previous generator did not receive the original `entrance_expand` geometry,
even though RPLAN and CE2EPlan define it as an input condition.

The new path extracts the conditioned opening interval, anchors it to the
nearest exterior boundary edge, and supplies it to:

- a typed entrance geometry token;
- room-query cross-attention;
- boundary-global and room-count context;
- the staged vector decoder; and
- canonical opening readout.

The token is opt-in so old checkpoints retain their exact state-dict contract.

### Pixel-staircase wall loss

Mask-partition room polygons encode pixel cells. Their room boundaries contain
one-pixel staircases. The previous wall-candidate reader split these staircases
into sub-threshold fragments, so complete exterior walls could disappear while
the room polygons remained valid.

Room boundary candidates are now simplified with a bounded polygon
approximation before junction splitting. Final room polygons are unchanged.
This reduced the dangling-endpoint rate on the 22 historical canonical failures
from `0.2181` to `0.0225`; all 22 cases became parsable with validity `1.0`.

### Fixed-point containment

Complex diagonal footprints can move outside a footprint by a few square pixels
when mask-cell intersections are rounded to the token grid. Room clipping now
includes a `0.1`-pixel precision inset and records
`room_clip_precision_guard`. This preserves the evaluator tolerance and fixes
the observed containment failure without weakening validation.

### Dead learned CAD heads

The 4096 mask-only checkpoint predicted zero active wall queries out of 128 on
256 validation cases, despite a mean target count of `27.9`. Opening queries
also predicted zero against a target count of `1.0`.

The loss used a hard distance threshold during Hungarian matching. At
initialization, no query passed the threshold, so no wall/opening query received
positive existence supervision and no endpoint/center loss was applied. The
head therefore converged to the all-negative solution.

Training now uses fixed-cardinality Hungarian assignment: every valid target is
assigned to one query before negative classification. Existence losses use a
dynamic positive weight bounded by `20`. Matching is unchanged at evaluation.

### Query fragments inflated room count

The room-count head itself had lower error than the canonical output because a
single room query could contribute several disconnected threshold fragments.
Each query now contributes only its largest legal connected component; secondary
fragments are logged and dropped. On the common five-seed quality protocol this
reduced mean count MAE from `1.05195` to `0.88672` while preserving the positive
mean Room IoU and increasing mean instance F1.

### Junction ordering broke token idempotence

Full-validation token round-trip exactness initially fell to `0.8971`. Junction
IDs were assigned from unquantized cluster centers and then sorted again from
quantized points on the next canonicalization pass. Nearby junctions could
therefore swap tuple order without changing geometry. Junction sorting now uses
the fixed-point coordinates before IDs are assigned. The full 12,018-case
round-trip is back to `1.0000`.

## New Screen Contract

The matched screen uses:

- train1024, validation256, batch 8, 12 epochs, seed 2905;
- the same ResNet18 backbone and mask-only objective;
- geometry-only control without the entrance token;
- learned-token arm with the entrance token;
- separate raw learned wall/opening diagnostics; and
- one-forward NFE.

The deterministic entrance/readout repair is evaluated separately from the
learned token. A conditioned opening recall must not be reported as evidence for
the learned opening head.

## Acceptance

The learned-token arm must preserve Room IoU and instance F1 within `0.01`,
retain validity `1.0`, reach conditional watertightness at least `0.95`, keep
the dangling-endpoint rate at most `0.05`, and improve at least one predeclared
model metric over the geometry-only control.

Raw wall and opening heads must report predicted counts and matched F1 without
the canonical room-derived fallback. A head that predicts all negatives is a
failure even if the canonical fallback remains valid.

## Two-Seed Entrance Result

Seeds `2905` and `37` compare the learned entrance token against the
geometry-only control under the same `1024/256`, 12-epoch budget.

- Room IoU mean delta: `+0.00105` (one seed positive, one negative);
- instance F1: `+0.00575` on both seeds;
- type accuracy: `+0.01852` on both seeds;
- count MAE: `-0.06836` on both seeds;
- raw wall F1: `-0.03995` on both seeds.

The learned token is therefore retained only as an optional room
identity/type/count conditioner. It is rejected as a wall/opening mechanism.

## Compact CAD and Partition Edge

Wall targets in 4,096 training cases have mean `27.79`, p99 `40`, and maximum
`47`, so the wall-query bank is reduced from 128 to 64 and openings from eight
to four. A differentiable partition-edge map is computed from competing room
masks and injected into wall-decoder memory.

On seed 2905, the compact control reaches raw wall F1 `0.3500`; the
partition-edge arm reaches `0.4143`. Canonical effects are:

| Metric | Compact control | Partition edge |
| --- | ---: | ---: |
| Room IoU | 0.64168 | **0.64370** |
| Instance F1 | 0.46953 | **0.49465** |
| Type accuracy | **0.71523** | 0.70626 |
| Count MAE | **0.88281** | 0.94141 |
| Node F1 | 0.88587 | **0.92314** |
| Edge overlap | 0.54631 | **0.63052** |
| GED | 3.90625 | **3.22266** |
| Conditional watertightness | **0.99609** | 0.99219 |

Validity is `1.0` for both. The mechanism passes the first-seed geometry and
raw-wall gates but needs the seed-37 confirmation before promotion. Learned
opening F1 remains near zero and is excluded from the main output claim.

### 4096 Promotion Result

At train4096/validation512, the compact control is stronger on identity:

| Metric | Compact control | Partition edge |
| --- | ---: | ---: |
| Canonical Room IoU | **0.66108** | 0.65797 |
| Instance F1 | **0.50686** | 0.48368 |
| Type accuracy | **0.76090** | 0.75607 |
| Count MAE | 0.94531 | **0.91406** |
| Node F1 | 0.90003 | **0.90893** |
| Edge overlap | 0.60175 | **0.60641** |
| GED | 3.27734 | **3.22852** |
| Conditional watertightness | 0.99023 | 0.99023 |
| Raw wall F1 | **0.41967** | 0.40392 |

The partition-edge branch therefore does not survive as a general room
identity improvement at 4096. It is retained as a topology-specialized variant:
graph metrics improve slightly while identity F1 falls by `0.0232` and raw wall
F1 falls. The compact 64-query bank itself improves the training-mask
Room IoU/F1 relative to the 128-query mask-only candidate and is promoted to a
five-seed quality run. The edge variant cannot enter the main claim without a
quality-preserving isolation/arbitration mechanism.

