# FloorFlow Mask-Partition Readout: Five-Seed Evidence

## Protocol

Five seeds (`2905, 19, 29, 37, 43`) use the same 13.61M-parameter ResNet18
backbone, 1024 training cases, 256 declared validation cases, 12 epochs, batch
eight, and validation-only checkpoint selection. RPLAN test remains closed.

The output path is:

```text
selected room masks
  -> per-query mask competition at threshold 0.5
  -> exact 4-connected cell vectorization
  -> footprint clipping
  -> CanonicalFloorplan + reversible token codec
```

Every report in this note was generated on the same 256-case validation prefix.
No external leaderboard or test result is implied.

## Main Result

| Metric | Mean | 95% t-CI | Seed range |
| --- | ---: | ---: | ---: |
| Room IoU | **0.6122** | [0.5891, 0.6352] | 0.5860-0.6337 |
| Instance F1 | **0.3749** | [0.3505, 0.3993] | 0.3523-0.4061 |
| Type accuracy | **0.6726** | [0.6483, 0.6969] | 0.6454-0.6942 |
| Count MAE | 1.1844 | [1.0548, 1.3140] | 1.0586-1.3320 |

All five checkpoints assemble 256/256 canonical plans. Token round-trip exactness
has mean `0.9977` and minimum `0.9922`.

## Paired Comparison Against Polygon Readout

The polygon and mask readouts use the same checkpoint for each seed.

| Metric | Mask minus polygon | 95% t-CI | Paired t-test |
| --- | ---: | ---: | ---: |
| Room IoU | **+0.0669** | [0.0316, 0.1021] | p=0.0062 |
| Instance F1 | **+0.0610** | [0.0180, 0.1040] | p=0.0169 |
| Type accuracy | +0.0548 | [-0.0208, 0.1303] | p=0.1145 |
| Count MAE | +0.0996 | [-0.0176, 0.2168] | p=0.0778 |

Positive count MAE means mask partition is worse on count. Type and count changes
are not claimed as significant. The polygon path assembles only `107.4/256`
cases on average, while mask partition assembles all cases.

## Threshold Check

A threshold sweep over `0.2-0.7` was run on the same declared validation prefix.
Raw-label scoring selected `0.3`, but canonical five-seed Room IoU was `0.6117`
at `0.3` versus `0.6122` at `0.5`. The `0.3` round-trip mean was also lower
(`0.9953` versus `0.9977`). The deployable threshold is therefore frozen at
`0.5`.

## Interpretation Boundary

- This establishes a readout bottleneck and a robust geometry improvement.
- It does not establish a new RPLAN test result or SOTA claim.
- CE2EPlan remains a pretrained 300k-step/1000-NFE external anchor.
- The next test is data/objective scale-up followed by full validation.
