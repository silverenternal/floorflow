# Strict-Locality Positioning Evidence v1

This is an internal evidence brief, not manuscript prose.

## Verified conference context

The official CVPR 2026 technical-program announcement reports:

- `16,092` submissions
- `4,089` accepted papers
- derived acceptance rate `25.41%`

The previously supplied `4,090` is off by one. Use the official number and link:
<https://cvpr.thecvf.com/Conferences/2026/News/Technical_Program>.

This statistic provides field context only. It does not imply an acceptance
probability for this paper.

## Why locality needs its own axis

HouseMind, ChatHouseDiffusion, RenoLayout, and ReLayout all discuss localized or
structure-preserving editing in some form. Their interfaces and reported metrics
differ, and the screened material does not provide the exact four-factor
outside-preservation protocol used here.

The defensible argument is therefore:

> Generated layout fidelity and edit-region safety are separate evaluation axes.
> FloorFlow makes the edit region an explicit architectural and measurement
> contract, reporting outside pixel exactness, outside room-label exactness,
> validity, and conditional watertightness independently.

This supports a protocol and evaluation contribution. It does not support
“FloorFlow is the first” or “other methods do not preserve locality.”

## Current quantitative anchor

The current submission table uses the oracle-control CubiCasa test503 protocol,
not a deployable predicted-control claim:

| Metric | FloorFlow | HouseMind-E public anchor |
|---|---:|---:|
| Delta-IoU, higher is better | 0.956703-0.999801 | 0.608 |
| Delta-MSE, lower is better | 0.000019-0.000096 | 0.0019 |
| Node F1 | 0.982250 | 0.998 |
| Edge overlap | 0.929475 | 0.934 |
| GED | 0.869544 | 0.467 |

The first two rows are the positive empirical headline. The graph rows are
reported as a limitation. Exact outside pixel, room, polygon, wall, opening, and
edge preservation is `1.0`; FP-Eval validity is `1.0` and conditional
watertightness is `0.9579-0.9675`.

This supports the following scoped claim:

> FloorFlow establishes a strong edit-detection and locality anchor under a
> declared oracle-control protocol. It does not claim complete-method HouseMind
> superiority or a same-benchmark leaderboard rank.

## Reviewer boundary

Use the related-work section to establish why edit safety deserves measurement.
Do not convert missing competitor metrics into zeros or failures. Missing
metrics remain N/A.

