# HouseMind Audit Positioning

| Method | Task | Delta-IoU | Node F1 | Edge overlap | GED | Outside exact | Violation | Negative controls | Attacks | Paired stats |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| HouseMind-E | natural-language floorplan editing | 0.608 | 0.998 | 0.934 | 0.467 | — | — | — | — | — |
| HouseMind-O | unified Omni model; editing evaluation | 0.598 | 0.997 | 0.908 | 0.653 | — | — | — | — | — |
| HouseMind-G | generation-only model | — | — | — | — | — | — | — | — | — |
| FloorFlow | CubiCasa test503, declared oracle controls | 0.956703-0.999801 | 0.982250 | 0.929475 | 0.869544 | 1.0 | 0.0 | Reported | Reported | Reported |

HouseMind-E/O values are public Table 3 context, not a same-dataset ranking.
HouseMind-G is generation-only and is not placed on the editing axis.

FloorFlow numerically exceeds the Delta-IoU anchor but not the Node, Edge, or GED
anchors. Its locality, negative-control, attack, and paired-statistics cells are
FloorFlow-reported evidence rather than claims about missing HouseMind values.

We do not claim that FloorFlow beats HouseMind or that HouseMind lacks locality. We claim that the public HouseMind evaluation does not expose the corresponding audit evidence, so no matched locality ranking is possible.

A dash or null means not reported or not applicable. It does not mean zero, failure, or inferiority.
