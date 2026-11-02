# FloorFlow Mask-Only Objective: Five-Seed Evidence

## Motivation

The mask-partition readout already bypasses the polygon-loop geometry head. Keeping
the polygon regression loss therefore spends optimization capacity on an output
that is not consumed. The mask-only objective sets the polygon-loop weight to zero
while retaining identity, count, wall, opening, and adjacency supervision.

## Protocol

Five seeds (`2905, 19, 29, 37, 43`) use the same 13.61M-parameter ResNet18
backbone, 1024 training cases, 256 validation cases, 12 epochs, batch eight,
threshold `0.5`, and validation-only checkpoint selection. RPLAN test remains
closed.

## Canonical Result

| Metric | Mean | 95% t-CI | Seed range |
| --- | ---: | ---: | ---: |
| Room IoU | **0.6413** | [0.6261, 0.6565] | 0.6243-0.6544 |
| Instance F1 | **0.4671** | [0.4586, 0.4757] | 0.4605-0.4786 |
| Type accuracy | **0.7199** | [0.6937, 0.7462] | 0.6963-0.7429 |
| Count MAE | 1.0953 | [0.8895, 1.3012] | 0.9063-1.3203 |

All five checkpoints assemble 256/256 canonical plans. Token round-trip exactness
has mean `0.9953` and minimum `0.9922`.

## Paired Objective Comparison

Compared with polygon-loss-on checkpoints at the same seeds:

| Metric | Mask-only minus polygon-loss-on | 95% t-CI | Paired t-test |
| --- | ---: | ---: | ---: |
| Room IoU | **+0.0291** | [-0.0030, 0.0612] | p=0.0654 |
| Instance F1 | **+0.0922** | [0.0603, 0.1242] | p=0.0013 |
| Type accuracy | **+0.0473** | [0.0059, 0.0887] | p=0.0337 |
| Count MAE | -0.0891 | [-0.3473, 0.1692] | p=0.3926 |

The objective change significantly improves F1 and type accuracy. Room IoU has a
positive mean and positive mean CI evidence but the 95% interval crosses zero;
it is reported as a strong trend, not a significant paired effect.

## Interpretation

The current optimal objective for the FloorFlow mask-partition path does not
train the unused polygon loop. This supports the architectural decision to make
the mask partition the primary output. The values remain single-prefix,
single-checkpoint validation evidence and do not establish an external SOTA.

Removing wall, opening, and adjacency losses as well as polygon regression
short-circuits more compute, but two 1024/256 seeds did not show stable quality
superiority. Their canonical Room IoU was `0.6432/0.6424` with F1
`0.4703/0.4807`, while type accuracy fell relative to the mask-only suite. That
`mask-core` branch remains diagnostic; the five-seed candidate retains the
wall/opening/adjacency auxiliary losses.

## Canonical Structural Audit

On the seed-43 checkpoint, the quality-first threshold readout has validity `1.0`
but conditional watertightness `0.102`, Node F1 `0.923`, Edge overlap `0.604`,
and fixed-evaluator GED `3.332`. The low watertightness comes from holes and
disconnected room support, not invalid polygons.

A declared safety-first `partition_fill` readout uses nearest-room assignment
inside the footprint and one-primary-component cleanup. It reaches
conditional watertightness `1.0`, validity `1.0`, Node F1 `0.923`, Edge overlap
`0.640`, and GED `3.059`, while Room IoU decreases to `0.631` and instance F1 to
`0.389`. The two modes are reported as a quality/safety Pareto pair rather than
mixed into one headline.

The generalization split at validation offset 256 and size 1024 confirms both
regimes: quality-first mask output has Room IoU `0.6402` and F1 `0.4561`;
safety-first partition fill has validity `1.0`, conditional watertightness
`0.9971`, Node F1 `0.9189`, Edge overlap `0.630`, and GED `3.045`.
