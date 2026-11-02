# FEVP v1.0-Hardened Metric Contracts

This document defines the executable contract in
`configs/fevp_metric_contract_v2.json`. The reference implementation is
`floorflow.evaluation.fevp_metric_contracts`.

## Global Rules

- Labels are canonical `uint8` values on a 256 x 256 grid.
- Pixel exactness has zero tolerance.
- Walls in the target before/after label maps are excluded from the valid outside
  domain after one-pixel dilation.
- Room matching is one-to-one Hungarian matching with the same semantic label and
  intersection-over-union at least 0.5.
- No metric permits many-to-one matches or target-informed selection.
- Floating results are reported to six decimals, but comparisons use the declared
  geometric tolerances rather than display rounding.

## Outside Exact

`outside_pixel_exact` is the fraction of valid outside pixels whose label is
unchanged. Its tolerance is exactly zero.

`outside_room_level_exact` requires all untouched before-edit room instances to:

1. match one-to-one to the after-edit room of the same semantic type;
2. preserve their raster mask;
3. preserve a polygon boundary within 0.5-pixel Hausdorff distance;
4. preserve all edges among untouched rooms with boundary contact within two
   pixels.

`locality_violation_rate` is changed valid outside pixels divided by all valid
outside pixels. It is reported separately from the binary exactness rate.

## Validity

Validity is the conjunction of five checks:

| Check | Failure |
|---|---|
| Outer boundary closed | Footprint has no non-degenerate closed exterior ring. |
| Room polygon closed | A room ring is empty, open, or below four pixels. |
| No self-intersection | A room polygon is invalid. |
| No room overlap | Pairwise room intersection exceeds 4 px2 (a 2x2 grid-snap sliver). |
| Semantic complete | A room has an illegal or missing label. |

The report must include all five subchecks and the error-code list. A validity of
1.0 without the five checks is not a complete FEVP row.

## Conditional Watertightness

Watertight diagnostics report:

- watertight rate on eligible targets;
- hole count and hole-area ratio;
- disconnected component count;
- dangling endpoint rate;
- issue codes.

Raster holes are computed by filling the room-support mask and measuring enclosed
unlabelled components above 0.25 px2. Wall endpoints are clustered within two
pixels; an endpoint in a degree-one cluster is dangling.

The local topology scaffold is evaluated before and after repair. Repair claims
must report repairs, regressions, and unresolved failures separately.

## Graph Metrics

Node matching is a semantic room-type set comparison. Node precision, recall, and
F1 are always reported together.

Edge matching uses undirected canonical semantic-type pairs. Adjacency means that
room polygon boundaries are within two pixels. Edge precision, recall, F1, and
Jaccard overlap are reported together. Edge shifts beyond two pixels and edge
insertions/deletions are protocol attacks.

## GED

The fixed GED evaluator is
`floorflow.evaluation.housemind_structure.housemind_style_ged`.

- Node substitution: zero for equal room types, otherwise one.
- Node insertion/deletion: one.
- Edge insertion/deletion: one.
- Edge substitution: zero.
- Timeout: 0.5 seconds; timeouts and fallbacks are recorded.

Only paired differences under this evaluator are interpreted. Absolute values
are not compared to CE2EPlan or another paper's GED.

## Delta Metrics

Delta-IoU and Delta-MSE are reported as full distributions, per operation,
room-count strata, edit-area strata, and hardest-decile cases. Means alone are
not accepted as complete evidence.

## Mandatory Negative Controls

| Metric | Controls |
|---|---|
| Outside exact | 1-pixel untouched-room move; label change; adjacency-only change. |
| Validity | Open outer ring; open room ring; self-intersection; overlap; illegal label. |
| Watertight | One-pixel hole; disconnected room; open wall cycle. |
| Node F1 | Delete node type; add spurious type; relabel room. |
| Edge F1 | Delete edge; insert spurious edge; shift edge beyond tolerance. |

The attack suite records sensitivity and false-pass/false-fail behavior for every
control. A control that does not change its intended metric is a protocol failure.
