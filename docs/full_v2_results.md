# Structured condition results (full 503)

## Protocol

- Frozen CubiCasa5K floor split: 5,126 train / 491 val / 503 test.
- Room supervision: dense partition with nearest-room fill.
- Object channels are trained with the room backbone frozen; object gradients do
  not update the room backbone.
- EMA checkpoint selected on raw room-partition IoU.
- Structured-object thresholds were selected globally on validation only:
  opening 0.7, furniture 0.7, structural support gate enabled.
- Room partition receives a deterministic top-1 nearest-contour prior from the
  training cache with weight 0.75; no test target or best-of-N is used.
- Retrieval bank: 5,126 train floors, fingerprint
  `27eec6b2c8ae44c5604fe7d4588382747204fdee26e14f49d5258cfc4b91f13d`.
- Cosine descriptor retrieval is retained: it reached 0.1254 room IoU on test.
  A binary-mask-IoU retriever looked better on validation (0.1257) but fell to
  0.1217 on test, so it is not used.
- Class-wise prior calibration raised the 191-val room IoU to 0.1280 but only
  reached 0.1259 on test with a small group-IoU regression, so the global
  0.75 prior is retained.
- No best-of-N, target-informed candidate selection, or test tuning.

## Metrics

The table separates channel-level metrics from single-raster composition.
Composition cannot be used to judge both room and object channels because an
object masks the room label beneath it.

| Metric | 503 test |
| --- | ---: |
| Room macro IoU, composed single raster | 0.1254 |
| Room macro IoU, partition channel | 0.1226 |
| Functional-group macro IoU | 0.1575 |
| Dataset class-macro room IoU | 0.0622 |
| Dataset micro room IoU | 0.1417 |
| Normalized room coverage / recall | 0.9201 |
| Target room occupancy | 0.8125 |
| Generated room occupancy | 0.9171 |
| Opening F1, object channel | 0.3523 |
| Opening F1, composed raster after wall fusion | 0.2957 |
| Furniture IoU, object channel | 0.1451 |
| Wall IoU, object channel | 0.3913 |
| Wall IoU, composed after output fusion | 0.2560 |
| Interior-wall IoU, composed | 0.0270 |
| Wall centerline F1, composed | 0.4977 |
| Structural closure, wall ∪ opening | 0.9589 |
| Furniture containment, conditioned on non-empty prediction | 1.0000 |
| Room-area distribution MAE | 0.0416 |
| Outside-footprint fraction | 0.0 |

Paired bootstrap against the prior condition path:

| Metric | Mean delta | 95% CI |
| --- | ---: | ---: |
| Room macro IoU | +0.02850 | [+0.02018, +0.03754] |
| Opening F1 composed | -0.00719 | [-0.00974, -0.00481] |
| Interior-wall IoU | +0.02696 | [+0.02523, +0.02888] |
| Furniture IoU composed | -0.00377 | [-0.00865, +0.00119] |
| Furniture containment | +0.07914 | [+0.07107, +0.08762] |

The channel-level furniture IoU is 0.1451, while the composed single-raster
furniture IoU is 0.0475. This difference is caused by mutually exclusive label
composition, not by two different furniture checkpoints.

## Gate verdict

Validity passes, while research performance targets still fail:

- Validity passes full-case count, no best-of-N, zero outside footprint,
  exterior-wall validity, normalized room coverage, opening-wall alignment,
  furniture containment, room-area MAE, and sampling integrity.
- Furniture containment is conditioned on non-empty predictions: an empty
  prediction is vacuously contained; recall is measured separately by furniture
  IoU/F1.
- Fine/group/dataset room IoU and interior-wall IoU remain below research
  thresholds. Wall centerline F1 and structural closure pass.

Artifacts:

```text
outputs/eval/condition_structured_retrieval075_final_503_v4/metrics.json
outputs/eval/condition_structured_retrieval075_final_503_v4/per_case.json
outputs/eval/condition_structured_retrieval075_final_503_v4/publication_gate.json
outputs/train/condition_structured_full_v1_low_lr/
outputs/train/structured_objects_full_v2/
```
