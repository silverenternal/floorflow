# Training-dynamics data maps

FloorFlow adapts the region definitions and training-dynamics coordinates from
[Dataset Cartography](https://github.com/allenai/cartography), pinned at
`3df3438fcc7324e706dee2e787426389bbd8fb1a` (Apache-2.0).

The upstream implementation labels NLP examples from the mean, standard
deviation and correctness of per-epoch true-class probabilities. FloorFlow
keeps those coordinates but computes them inside the rasterized footprint and
adds family-specific directions needed by CAD reconstruction:

| Coordinate | FloorFlow definition |
| --- | --- |
| confidence | mean true-class probability inside the footprint |
| variability | epoch-to-epoch std of confidence |
| correctness | pixel accuracy inside the footprint |
| room precision | correct pixels among predicted room-class pixels |
| room recall | correct pixels among GT room pixels |
| opening/furniture accuracy | accuracy restricted to GT pixels of that family |
| forgetfulness | transitions from mostly correct to mostly incorrect |

Training data remains visited uniformly. The resulting weights multiply the
loss only after a configurable warm-up, which preserves the empirical
`p(x)` while changing the optimization emphasis. This is deliberate: direct
stratum resampling was already measured as a regression.

The data map is generated per epoch under the requested training output
directory. Join it to the fine-grained data index with
`scripts/analyze_data_dynamics.py`.

## Full-data study

The 5,126-case pool was visited once from the retained parent checkpoint, with
and without dynamic loss weights under the same optimizer and augmentation
protocol. On val491, dynamic weighting changed Room IoU from `0.12762` to
`0.12805` and balanced CAD from `0.28344` to `0.28522`, while improving
instance IoU by `0.00702`, room-polygon F1 by `0.00267`, watertight by
`0.00204`, and room coverage by `0.00332`.

It also reduced opening IoU by `0.01049`, furniture IoU by `0.00293`, and the
`rooms0` Room IoU by `0.01856`. A train-only guard that disables dynamic
weights on low-room-count and low-instance-density samples was tested as
`outputs/reports/datamap_full1_guard_val491.json`; it regressed Room IoU,
instance IoU, opening IoU and furniture IoU, so it was rejected.

The mechanism is retained as an isolated experimental option. It is not the
production default until a candidate improves the primary metric and every
declared weak stratum.
