# HouseMind Positioning v2

HouseMind addresses unified tokenization and multimodal instruction editing. FEVP addresses whether edit locality is independently auditable. HouseMind-E/O publish task-level edit and graph metrics but do not publish outside exactness, violation rate, negative controls, protocol attacks, or paired locality statistics.

| Method | Evaluation context | Delta-IoU | Delta-MSE | Node F1 | Edge overlap | GED |
|---|---|---:|---:|---:|---:|---:|
| HouseMind-E | RPLAN natural-language editing, published | 0.608 | 0.0019 | 0.998 | 0.934 | 0.467 |
| FloorFlow | CubiCasa test503, declared oracle controls | 0.956703-0.999801 | 0.000019-0.000096 | 0.982250 | 0.929475 | 0.869544 |

FloorFlow numerically exceeds the published Delta-IoU and Delta-MSE anchors but
does not exceed the published Node F1, edge overlap, or GED anchors. The
datasets, inputs, and task interfaces differ, so these rows are metric context,
not a unified leaderboard.

We do not claim that FloorFlow beats HouseMind or that HouseMind lacks locality. We claim that the public HouseMind evaluation does not expose the corresponding audit evidence, so no matched locality ranking is possible.

A dash or null means not reported or not applicable. It does not mean zero, failure, or inferiority.
