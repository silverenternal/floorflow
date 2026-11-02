# Opening Consumer Promotion v1

Date: 2026-10-31

## Decision

Promote the learned opening-only consumer as a predicted-control mechanism.
This is not a full-CAD SOTA claim and does not close the final Opening F1 target.

Evidence:

- `outputs/reports/opening_parity_v2/opening_interval_queryblend_fiveseed_v1.json`
- seeds: 19, 20, 21, 22, 23
- splits: cal384, holdout107, val491
- paired bootstrap iterations: 5,000
- query readout blend: checkpoint-selected `0.75`

The earlier `outputs/reports/opening_interval_queryblend_fiveseed_v1.json`
aggregate used a stale `0.5` readout and is superseded.

## Paired Results

| Split | Opening F1 absolute delta | SD | Range | Improved seeds | CI above zero | Non-regression |
|---|---:|---:|---:|---:|---:|---:|
| cal384 | 0.17593 | 0.00705 | 0.16699–0.18374 | 5/5 | 5/5 | 5/5 |
| holdout107 | 0.15991 | 0.00525 | 0.15214–0.16629 | 5/5 | 5/5 | 5/5 |
| val491 | 0.17244 | 0.00565 | 0.16540–0.17829 | 5/5 | 5/5 | 5/5 |

Room IoU, support IoU, instance F1, count MAE, wall F1, Node F1, edge overlap,
exact-GED intersection, validity, watertightness, wall continuity, and outside
footprint did not regress in any seed.

The val491 baseline Opening F1 is approximately 0.095 and the mean promoted
consumer is approximately 0.267. The next Opening F1 target remains 0.35; final target
remains 0.75.

## Runtime Contract

`CadStageRuntime` now exposes `use_opening_consumer`.

- Enabled: replaces only the opening-family and opening-type channels.
- Disabled: returns a bit-exact copy of the pre-consumer structure tensor.
- The learned opening predictor is not loaded when the consumer is disabled.

Regression tests:

- `tests/test_cad_stage_runtime.py`
- `tests/test_opening_consumer_comparison.py`

The consumer remains confined to predicted-control evidence and must not be
reported as full CAD or free-text editing SOTA.
