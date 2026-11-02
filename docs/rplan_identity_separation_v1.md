# RPLAN Identity Separation Review v1

## What Improved

Class-agnostic active-query separation is a real mechanism gain.

At 256/64:

- baseline Room IoU `0.3410` -> `0.4493`;
- instance F1 `0.1197` -> `0.1842`;
- type accuracy `0.4010` -> `0.5299`.

At 1024/256:

- baseline Room IoU `0.4775` -> `0.5042`;
- instance F1 `0.2391` -> `0.2696`;
- type accuracy `0.5806` -> `0.6231`;
- count MAE `1.1953` -> `0.9336`.

The 1024 result is still below the Room IoU `0.55` promotion gate and cannot
support a SOTA claim.

## Why It Works

The separation term penalizes soft Dice overlap between active room queries.
It removes duplicate hypotheses and improves one-to-one assignment, reflected by
higher matched Room IoU and instance F1.

## Why It Is Not Sufficient

The same term shrinks or suppresses masks:

- mask IoU falls `0.5610 -> 0.5334`;
- rendered macro IoU falls `0.3284 -> 0.2598`;
- unique predicted room classes fall `5.06 -> 4.00`.

This is an alignment/precision tradeoff, not a pure improvement in dense room
decoding.

## Rejected Variants

Class-balance only at 256/64 reaches Room IoU `0.3949`, F1 `0.1417`; combined
reaches `0.4116/0.1538`; class-aware separation reaches `0.4242/0.1722`. All
underperform class-agnostic separation at `0.4493/0.1842`.

The class-weight implementation uses per-batch inverse frequency. A batch has
roughly 72 valid rooms at this budget, so weights are nonstationary and can
amplify rare-class noise. Class-aware overlap also weakens duplicate suppression
for uncertain cross-class queries.

Decision:

- retain class-agnostic separation;
- reject batch-level class balance;
- reject predicted-class-aware separation;
- do not promote combined losses.

## Budget Comparison

The current comparison is not budget matched.

| Model | Parameters | Training exposures used by screen | Training exposures declared/released | NFE |
| --- | ---: | ---: | ---: | ---: |
| FloorFlow 1024 screen | 13.61M | 8,192 | - | 1 |
| FloorFlow 4096 screen | 13.61M | 32,768 | - | 1 |
| CE2EPlan | 9.64M | - | 153.6M (README batch 512 × released step 300k) | 1000 |

Even using the conservative README batch size, the 4096 FloorFlow screen sees
about `0.021%` of CE2EPlan's training exposures, while using `1.41x` as many
parameters. Absolute comparison against the pretrained external checkpoint is
therefore an anchor only, not a matched training comparison.

## Next Review

The next separation loss should use Hungarian-positive query assignments and
target classes rather than batch class frequencies or predicted class affinity.
This keeps duplicate suppression but avoids suppressing legitimate room classes.
