# Opening detection decoder pre-execution review

## Decision

Train a frozen-feature opening detector rather than extending the existing
one-interval-per-wall `WallIntervalHead`.

The current target builder stores one interval for each wall. A diagnostic on
the first 64 validation plans found 53 plans (82.8%) with at least one wall
carrying multiple openings, including walls with up to six. The representation
therefore collapses repeated openings before optimization starts, so more
epochs cannot recover the missing instances.

## Empirical baseline

- The best existing threshold-only wall-interval readout reaches opening F1
  `0.11262` on calibration64.
- The existing structured opening evidence lowers end-to-end opening F1 to
  `0.05325` in the canonical CAD composition.
- The structured-object detector itself reports a much stronger pixel opening
  F1 (about `0.35`) in its native validation report, so the dominant failure is
  in how openings survive vectorization and host-wall association, not evidence
  that the raster contains no opening signal.

## External design evidence

The 2026 fpvec-lab study, "When Should a Network Emit Geometry, and When Should
It Detect It?", compares sequence emission and dense detection on the same
network. On real CubiCasa5K scans, detection is stronger for walls; reading a
trained opening heatmap raises opening F1 from `0.246` to `0.644`, and to
`0.799` when the family is ignored. The paper also reports a small standalone
door/window family head with `0.988` family accuracy.

This supports a local detection head whose output is vectorized only after
reading the heatmap. It does not support another global opening threshold
sweep.

## Matched budget

| Item | Screen | Promotion |
| --- | ---: | ---: |
| Train / calibration cases | 256 / 64 | 256 / 64 then offset128 |
| Batch size | 8 | 8 |
| Updates per epoch | 32 | 32 |
| Epochs | 8 | 24 or first documented plateau after epoch 8 |
| Optimizer updates | 256 | 768 |
| Input pixels at 384px | 302,231,454 | 906,694,362 |
| Trainable scope | opening head only | opening head only |

The encoder is frozen. A one-step or one-epoch run is only an integration
smoke. The eight-epoch arm is retained for promotion if the opening vector F1
is still improving over its final checkpoints. Otherwise it may be rejected
only after the pixel detector, component vectorizer, and host-wall assignment
are separately audited.

## Matched control and integration contract

The control is the frozen structured-opening checkpoint under the same 384px
frozen encoder, host wall plan, component vectorizer, threshold grid, and
`fpeval` decode. Room polygons and the wall layer are unchanged. The learned
opening layer can only replace openings attached to the existing wall plan, so
room and instance metrics are non-regression guards rather than targets.

Metrics are reported in three layers:

1. Pixel opening precision, recall, and F1.
2. Oracle-host vector opening F1, a diagnostic only.
3. End-to-end predicted-host `fpeval` opening F1 with unchanged room/wall
   validity and watertightness checks.

No test503 case is opened during this screen.
