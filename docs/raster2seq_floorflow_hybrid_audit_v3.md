# Raster2Seq + FloorFlow Structural Audit v3

## Scope

This is a clearly labeled hybrid on the Raster2Seq val503 split. Raster2Seq
provides room proposals; FloorFlow adds room seams, supported-openings
filtering, exterior composition, wall-gap reconciliation, and bounded
open-endpoint repair. It is not standalone FloorFlow.

Official Raster2Seq Room, semantic Room, Corner, Angle, and Opening values are
read directly from the upstream `result_jsons`. FloorFlow adapter and
structural values are reported separately.

## Accepted Configuration

- 256 x 256 audit resolution;
- 32 px fragment merge;
- one-pixel room seam;
- unsupported-opening removal;
- boundary width 7;
- endpoint-gap fraction 0.06;
- no IP reconciliation;
- prune a short spur only when one endpoint is open and its length is at most
  0.02 of the frame;
- connect open wall endpoints that lie on the same frame boundary;
- preserve the explicit room polygons.

The bounded endpoint repair never changes the room raster or adapter fidelity.
On the same 503 cases it changes no official Raster2Seq metric and no case-level
wall-area fraction.

## Results

| Metric | v5, gap 0.06 | v6, bounded repair |
|---|---:|---:|
| Official Room IoU | 0.63228 | 0.63228 |
| Official semantic Room F1 | 0.64711 | 0.64711 |
| Official Corner F1 | 0.59368 | 0.59368 |
| Official Opening F1 | 0.34026 | 0.34026 |
| Adapter instance F1 | 0.83585 | 0.83585 |
| Adapter type accuracy | 0.58870 | 0.58870 |
| Adapter count error | 1.24851 | 1.24851 |
| Validity | 1.00000 | 1.00000 |
| Functional watertightness | 0.98807 | 0.98807 |
| Wall continuity mean | 0.98770 | 0.99480 |
| Wall-continuity case pass | 0.96024 | 0.98608 |
| Plan plausibility | 0.94831 | 0.97416 |
| Wall count mean | 113.19 | 112.95 |

`Wall count` is the number of graph segments, not wall area. The accepted
repair reduces the mean wall count and leaves wall-area fraction exactly
`0.1608764815860483` unchanged.

## Mechanism Screen

The fixed tail contained images `4, 26, 54, 55, 73, 76, 96, 107`. Before the
repair, all eight failed. After the repair, images `4, 26, 54, 55` pass.
Images `73, 76, 96` retain a multi-component wall-continuity failure, while
image `107` retains a room-face collapse.

The adjacent 18-case control set had zero regressions, zero official-metric
changes, zero wall-area changes, and a net wall-count reduction of four.

The full run reduces the failure count from 26 to 13. The remaining cases are:

| Mode | Cases |
|---|---:|
| Multi-component wall continuity | 7 |
| Multi-component room-face collapse | 2 |
| Multi-component partition protocol boundary | 3 |
| Single-component room-face collapse | 1 |

These are retained as protocol boundaries and prototype failures. No threshold
was changed after full evaluation.

## Rejected Candidates

- Global gap `0.08`: no recovery of the isolated failing case in the 20-case
  screen, while mean wall segments rose to `126.45`.
- IP reconciliation on the fixed tail: every tail case failed, wall continuity
  fell to `0.73140`, and raster plausibility fell to `0.875`.

## Decision

Promote v6 only as a bounded reasonableness repair for the clearly labeled
hybrid. Do not present it as standalone FloorFlow, a Room IoU improvement, or a
learned-model contribution.

The next work item is the 2026 Raster2Seq proposal-teacher corpus followed by
native proposal-identity distillation into FloorFlow.
