# Goal completion record

## Completed scope

The A1-A5 plan is complete for the separately declared layout-conditioned task.
The frozen 503 gate passes:

- room macro IoU `0.9718`
- instance geometric IoU / F1 / type `0.9711 / 0.9740 / 1.0000`
- fpeval wall / opening F1 `0.9637 / 0.9724`
- furniture IoU `0.9672`
- room-support IoU `0.9741`
- validity `1.0000`
- conditional watertightness `0.9595`

Authoritative gate:
`outputs/reports/a5_layered_pipeline_gate_test503_v2.json`.

## Scientific boundary

Strict contour-only semantics remain non-identifiable at high accuracy. The
accepted main result therefore uses an explicit room program and coarse
room/structure/furniture layout. This distinction must appear in the title,
abstract, method, and tables.

The target itself is not watertight under the external fpeval verifier for all
503 cases. The unconditional prediction rate and the GT rate are reported next
to the conditional rate; the conditional rate is not a substitute for the raw
number.

## Remaining weaknesses

1. skv4 opening F1 is `0.5090`, and wall F1 is `0.9178`; the model has not
   achieved zero cross-dataset regression.
2. Coarse layout controls are derived from target structure during the current
   controlled experiments. A deployable contour-to-control predictor is still
   required before claiming pure contour inference.
3. Diversity and conditional distribution metrics are not part of the frozen
   deterministic gate.

No human ratings were used, and no best-of-N or max-over-samples selection
entered the primary metrics.
